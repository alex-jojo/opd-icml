"""CPU FSDP check of real Llama causal-prefix audits and the next AdamW update."""
import copy
from datetime import timedelta
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig,LlamaForCausalLM

from test_parameter_tokens import make_actor,make_data,optimizer_update,reference_energy,audit_module


def worker(rank,address):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=address,rank=rank,world_size=2,timeout=timedelta(seconds=120))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        torch.manual_seed(99)
        actor,data=make_actor(),make_data()
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,attention_dropout=0.)
        config._attn_implementation='eager'
        actor.actor_module=LlamaForCausalLM(config)
        actor.actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
        baseline=copy.deepcopy(actor)
        actor.actor_module=FSDP(actor.actor_module,device_id=torch.device('cpu'),use_orig_params=False)
        actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
        optimizer_update(actor,data);optimizer_update(baseline,data)
        expected=reference_energy(baseline,data)
        result=audit_module.FullParameterTokenAudit(actor,dict(mode='exact_all')).measure(data,1,0)
        assert result['causal_prefix']
        assert result['forward_input_tokens']<result['full_suffix_forward_input_tokens']
        np.testing.assert_allclose(result['energy'],expected,rtol=3e-5,atol=1e-6)
        optimizer_update(actor,data);optimizer_update(baseline,data)
        with FSDP.summon_full_params(actor.actor_module):
            for actual,ordinary in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
                torch.testing.assert_close(actual,ordinary,rtol=3e-5,atol=1e-6)
        if rank==0:
            print('PASS: two-rank CPU FSDP real Llama prefix audit, energy and next AdamW update',flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    mp.spawn(worker,args=(f'tcp://127.0.0.1:{port}',),nprocs=2,join=True)
