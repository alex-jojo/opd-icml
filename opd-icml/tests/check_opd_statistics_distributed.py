"""Two-rank CPU DDP: shared full/sample groups, entropy and unchanged gradients."""
from datetime import timedelta
from pathlib import Path
import socket
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_parameter_tokens import make_actor, make_data, objective, opd, study_module


def worker(rank, address, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        for mode in ('exact_all', 'kl_stratified'):
            torch.manual_seed(86)
            actor, data = make_actor(tied=True), make_data()
            actor.actor_module = torch.nn.parallel.DistributedDataParallel(actor.actor_module)
            actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.001)
            with torch.no_grad():
                # Use the same cached logits that determine the sampled population.
                _, cached, _, _ = opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False)
            data.batch['opd_student_top16_logits'] = cached
            directory = Path(output)/mode
            cfg = actor.config.policy_loss
            cfg['replication17'].update(mode='full', output_dir=str(directory), rank=2,
                freeze_step=10, interval=5, rounds=6, reuse=1, tokenizer_path=None)
            cfg['hidden_projection'] = dict(asset_path=str(directory/'basis.npz'))
            cfg['gradient_study'] = dict(enabled=True,
                parameter_tokens=dict(enabled=True, mode=mode, sample_size=4))
            calls = []
            original_prepare = study_module.prepare_groups

            def prepare(*args):
                calls.append(1)
                return original_prepare(*args)

            study_module.prepare_groups = prepare
            try:
                runtime = study_module.StudyReplication(actor)
                runtime.begin(data, None, 1)
                runtime.before_inner(0)
                part = data.select_idxs([rank])
                with runtime.watch_gradient_scopes(part.batch, 0, {}), runtime.watch_training(part.batch, 0, rank):
                    (objective(actor, part)*part.batch['response_mask']).sum().mul(2/5).backward()
                gradients = [p.grad.clone() for p in actor.actor_module.parameters()]
                runtime.before_update(0)
                assert calls == ([1] if rank == 0 else [])
                for parameter, reference in zip(actor.actor_module.parameters(), gradients):
                    torch.testing.assert_close(parameter.grad, reference, rtol=0, atol=0)
                dist.barrier()
                with np.load(directory/'token_audit/step_001_inner_1.npz') as saved:
                    teacher = data.batch['opd_teacher_top16_logits'][data.batch['response_mask'].bool()]
                    log_q = teacher.double().log_softmax(-1)
                    np.testing.assert_array_equal(saved['teacher_entropy'], (-(log_q.exp()*log_q).sum(-1)).numpy())
                    assert len(saved['row']) == 5
                if rank == 0:
                    print(f'PASS: two-rank {mode}; one shared grouping, identical FP64 entropy and unchanged gradients', flush=True)
            finally:
                study_module.prepare_groups = original_prepare
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='opd-statistics-') as output:
        mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}', output), nprocs=2, join=True)
