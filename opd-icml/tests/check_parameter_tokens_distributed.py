"""Two-rank norm/trajectory check: CPU DDP, or --fsdp on two CUDA GPUs."""
import argparse
import copy
from datetime import timedelta
import socket
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_parameter_tokens import make_actor, make_data, reference_energy, optimizer_update, audit_module


def worker(rank, address, fsdp):
    torch.set_num_threads(1)
    if fsdp == 'cuda':
        torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if fsdp == 'cuda' else 'gloo', init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(20)
        actor, data = make_actor(tied=True), make_data()
        if fsdp == 'cuda':
            actor.actor_module.cuda(rank)
            data.batch = {k: v.cuda(rank) for k, v in data.batch.items()}
        baseline = copy.deepcopy(actor)
        if fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            actor.actor_module = FSDP(actor.actor_module,
                device_id=rank if fsdp == 'cuda' else torch.device('cpu'), use_orig_params=False)
        else:
            actor.actor_module = torch.nn.parallel.DistributedDataParallel(actor.actor_module)
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.001)
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        expected = reference_energy(baseline, data)
        state_before = copy.deepcopy(actor.actor_optimizer.state_dict())
        measured = audit_module.FullParameterTokenAudit(actor, {'mode': 'exact_all'}).measure(data, 1, 0)
        np.testing.assert_allclose(measured['energy'], expected, rtol=.02 if fsdp else 2e-5, atol=1e-6)
        assert measured['backwards'] == 5
        for index, values in state_before['state'].items():
            for key, value in values.items():
                assert torch.equal(value, actor.actor_optimizer.state_dict()['state'][index][key])
        assert all(p.grad is None for p in actor.actor_module.parameters())
        optimizer_update(actor, data)
        optimizer_update(baseline, data)
        from contextlib import nullcontext
        with FSDP.summon_full_params(actor.actor_module) if fsdp else nullcontext():
            for actual, ordinary in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
                torch.testing.assert_close(actual, ordinary, rtol=.02 if fsdp == 'cuda' else 2e-5, atol=1e-6)
        if rank == 0:
            print('PASS: two-rank ' + ('FSDP' if fsdp else 'DDP') + ' full-token energy and unchanged AdamW trajectory')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--fsdp-cpu', action='store_true', help='Exercise FSDP sharding on CPU if supported by PyTorch')
    args = parser.parse_args()
    if args.fsdp and torch.cuda.device_count() < 2:
        parser.error('--fsdp requires two CUDA GPUs')
    if args.fsdp and args.fsdp_cpu:
        parser.error('select one FSDP device')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mode = 'cuda' if args.fsdp else ('cpu' if args.fsdp_cpu else None)
    mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}', mode), nprocs=2, join=True)
