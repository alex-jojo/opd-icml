"""Two-rank grouped graph checks with nested FSDP, checkpoints and mixed precision.

CPU: python tests/check_parameter_graph_reuse_distributed.py --fsdp --checkpoint
GPU: python tests/check_parameter_graph_reuse_distributed.py --cuda --fsdp --checkpoint --mixed-precision
"""
import argparse
import copy
from datetime import timedelta
from functools import partial
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_parameter_tokens import audit_module, make_actor, make_data, optimizer_update


def worker(rank, address, options):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if options.cuda else torch.device('cpu')
    if options.cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if options.cuda else 'gloo', init_method=address, rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        torch.manual_seed(97)
        config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             vocab_size=64, max_position_embeddings=64, attention_dropout=0.,
                             tie_word_embeddings=True)
        config._attn_implementation = 'eager'
        template = LlamaForCausalLM(config)
        if options.checkpoint:
            template.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        def actor_copy():
            actor = make_actor()
            model = copy.deepcopy(template).to(device)
            if options.fsdp:
                precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                           buffer_dtype=torch.bfloat16) if options.mixed_precision else None
                model = FSDP(model, device_id=device, use_orig_params=False, mixed_precision=precision,
                             auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                                      transformer_layer_cls={LlamaDecoderLayer}))
            else:
                model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[rank] if options.cuda else None)
            actor.actor_module = model
            actor.param_dtype = torch.bfloat16 if options.mixed_precision else torch.float32
            actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            return actor

        actor, baseline = actor_copy(), actor_copy()
        data = make_data()
        data.batch['opd_top16_ids'] = torch.arange(48).reshape(1, 3, 16).flip(-1).repeat(2, 1, 1)
        data.batch = {k: v.to(device) for k, v in data.batch.items()}
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        state = copy.deepcopy(actor.actor_optimizer.state_dict())
        captures = []
        original_norm = audit_module.GradientEnergy.squared_norm

        def capture(layout):
            gradients = []
            for parameter in layout.layout.params:
                assert parameter.grad is not None
                assert parameter.grad.shape == parameter.shape  # reduced shards, not full gradients
                gradients.append(parameter.grad.detach().cpu().clone())
            captures.append(gradients)
            return original_norm(layout)

        audit_module.GradientEnergy.squared_norm = capture
        sequential = audit_module.FullParameterTokenAudit(actor, dict(graph_group_size=1)).measure(data, 1, 0)
        references = captures[:]
        captures.clear()
        grouped = audit_module.FullParameterTokenAudit(actor, dict(graph_group_size=4)).measure(data, 1, 0)
        assert grouped['graph_group_size'] == 4 and grouped['graph_reuse_fallback'] is None
        assert grouped['forward_calls'] == 2 and sequential['forward_calls'] == 5
        assert grouped['backwards'] == sequential['backwards'] == 5
        tolerance = .03 if options.mixed_precision else 5e-5
        np.testing.assert_allclose(grouped['energy'], sequential['energy'], rtol=tolerance, atol=1e-7)
        np.testing.assert_allclose(grouped['diagnostic_kl'], sequential['diagnostic_kl'], rtol=tolerance, atol=1e-7)
        for actual, reference in zip(captures, references):
            for gradient, expected in zip(actual, reference):
                torch.testing.assert_close(gradient, expected, rtol=tolerance,
                                           atol=3e-4 if options.mixed_precision else 1e-6)
        # Repeat at the same weights to catch stale hook or accumulated-gradient state.
        repeated = audit_module.FullParameterTokenAudit(actor, dict(graph_group_size=2)).measure(data, 1, 0)
        assert repeated['forward_calls'] == 3
        np.testing.assert_allclose(repeated['energy'], sequential['energy'], rtol=tolerance, atol=1e-7)
        for index, values in state['state'].items():
            for key, value in values.items():
                assert torch.equal(value, actor.actor_optimizer.state_dict()['state'][index][key])
        assert all(p.grad is None for p in actor.actor_module.parameters())
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        for actual, expected in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for index, values in baseline.actor_optimizer.state_dict()['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, actor.actor_optimizer.state_dict()['state'][index][key], rtol=0, atol=0)
        if rank == 0:
            print(f'PASS: two-rank {"nested FSDP" if options.fsdp else "DDP"}; '
                  f'checkpoint={options.checkpoint}, mixed_precision={options.mixed_precision}; '
                  '5 -> 2 forwards, equal token gradients and unchanged AdamW trajectory')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    if args.cuda and torch.cuda.device_count() < 2:
        parser.error('--cuda requires two CUDA GPUs')
    if args.mixed_precision and not args.fsdp:
        parser.error('--mixed-precision requires --fsdp')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}', args), nprocs=2, join=True)
