"""Four CPU/Gloo FSDP ranks: overlap equivalence and worker-error propagation."""
from datetime import timedelta
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_parameter_tokens import make_actor, make_data, study_module


def worker(rank, address, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        torch.manual_seed(996)
        actor = make_actor()
        model = actor.actor_module
        model.model.embed_tokens = torch.nn.Embedding(16, 8)
        model.model.layers[0] = torch.nn.Linear(8, 8)
        model.model.norm = torch.nn.LayerNorm(8)
        model.lm_head = torch.nn.Linear(8, 16, bias=False)
        actor.actor_module = FSDP(model, device_id=torch.device('cpu'), use_orig_params=False)
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.001)
        data = make_data().select_idxs([0, 1]*8)
        basis = torch.eye(8)[:, :2]
        study = study_module.GradientStudy(actor, dict(batches=4, overlap_cpu_geometry=False), output)
        expected = study.cross_batch(data, basis, 5)
        study.cfg['overlap_cpu_geometry'] = True
        actual = study.cross_batch(data, basis, 5)
        assert actual == expected
        assert all(p.grad is None for p in actor.actor_module.parameters())
        namespace = study.cross_batch.__func__.__globals__
        original = namespace['scope_geometry']
        if rank == 0:
            def fail(*args, **kwargs):
                raise ValueError('four-rank CPU worker failure')
            namespace['scope_geometry'] = fail
        caught = False
        try:
            study.cross_batch(data, basis, 10)
        except Exception as error:
            caught = 'four-rank CPU worker failure' in str(error)
        finally:
            namespace['scope_geometry'] = original
        assert caught
        dist.barrier()
        if rank == 0:
            print('PASS: four-rank CPU FSDP overlap; identical statistics; CPU-worker failure reaches all ranks', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='opd-overlap-') as output:
        mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}', output), nprocs=4, join=True)
