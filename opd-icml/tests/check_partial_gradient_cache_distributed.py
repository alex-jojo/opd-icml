"""Two-rank CPU FSDP: partial-cache/replay equivalence and unanimous capacity."""
import copy
from datetime import timedelta
import importlib
import os
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_parameter_tokens import audit_module, make_actor, make_data, optimizer_update


def worker(rank, address):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        study = importlib.import_module(audit_module.__package__ + '.opd_gradient_study')
        torch.manual_seed(212)
        actor, data = make_actor(tied=True), make_data()
        baseline = copy.deepcopy(actor)
        actor.actor_module = FSDP(actor.actor_module, device_id=torch.device('cpu'), use_orig_params=False)
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.001)
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        state = copy.deepcopy(actor.actor_optimizer.state_dict())
        batches = []
        for i in range(4):
            batch = copy.deepcopy(data)
            batch.batch['opd_teacher_top16_logits'] = torch.roll(batch.batch['opd_teacher_top16_logits'], i, -1)
            batches.append(batch)
        layout = study.ParameterLayout(actor.actor_module)
        basis = torch.eye(4)[:, :2]
        reference = None
        counts = {}
        vector_bytes = sum(p.numel() * 4 for p in layout.params)
        for mode in ('replay', 'partial', 'cache', 'mixed_capacity', 'one_rank_low_memory'):
            units = {'replay': 0, 'partial': 4.1, 'cache': 5.1,
                     'mixed_capacity': 4.1 if rank == 1 else 5.1,
                     'one_rank_low_memory': 0 if rank == 1 else 5.1}[mode]
            os.environ['OPD_GRADIENT_CACHE_MAX_GIB'] = str(units * vector_bytes * 2 / 1024**3)
            gram, calls = study.exact_gram(lambda i: study.gradient(actor, batches[i], basis)[0], layout, 4)
            counts[mode] = calls
            assert calls == {'replay': 10, 'partial': 5, 'cache': 4, 'mixed_capacity': 5, 'one_rank_low_memory': 10}[mode]
            if reference is None: reference = gram
            for group in reference: np.testing.assert_array_equal(gram[group], reference[group])
            assert all(p.grad is None for p in actor.actor_module.parameters())
        for index, values in state['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, actor.actor_optimizer.state_dict()['state'][index][key], rtol=0, atol=0)
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        with FSDP.summon_full_params(actor.actor_module):
            for actual, expected in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        if rank == 0: print('PASS: CPU FSDP; identical Gram matrices, unanimous fallback, unchanged optimizer; calls='+str(counts), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}',), nprocs=2, join=True)
