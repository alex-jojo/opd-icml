"""Four real Gloo ranks + CPU FSDP: padding, reductions, archive I/O failures."""
from datetime import timedelta
import importlib
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_parameter_tokens import make_actor, study_module


def legacy_dot(layout, a, b):
    values = {}
    for i, start, n, groups in layout.slices:
        value = 0.
        for offset in range(start, start+n, 262144):
            end = min(offset+262144, start+n)
            value += float(torch.dot(a[i][offset:end].double(), b[i][offset:end].double()))
        for group in groups:
            values[group] = values.get(group, 0.) + value
    ranks = study_module.dp.gather(values)
    return {key: sum(rank.get(key, 0.) for rank in ranks) for key in set().union(*(r.keys() for r in ranks))}


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        module = importlib.import_module(study_module.__package__+'.opd_gradient_study')
        storage = importlib.import_module(study_module.__package__+'.opd_diagnostic_storage')
        torch.manual_seed(67)
        actor = make_actor(tied=True)
        # An odd coordinate count requires padding; the shared head counts once.
        actor.actor_module.extra = torch.nn.Parameter(torch.tensor([.75]))
        count = sum(p.numel() for p in actor.actor_module.parameters())
        actor.actor_module = FSDP(actor.actor_module, device_id=torch.device('cpu'), use_orig_params=False)
        layout = module.ParameterLayout(actor.actor_module)
        before = layout.capture(gradient=False)
        with torch.no_grad():
            for p in layout.params:
                p.add_(.125 if rank % 2 else -.25)
        delta = layout.delta(before)
        pairs = [(delta, delta), (before, delta), (before, before)]
        expected = [legacy_dot(layout, a, b) for a, b in pairs]
        actual = layout.dot_many(pairs)
        assert actual == expected, (rank, actual, expected)
        report = layout.update_summary(before, delta, actual[0])
        assert report['all']['count'] == count
        assert report['all']['changed'] == count
        assert report['all']['positive'] + report['all']['negative'] == count
        assert report['all']['update_norm'] == actual[0]['all']**.5
        assert 'shared_embedding_head' in report and 'head' not in report
        assert all(item == report for item in study_module.dp.gather(report))
        repeated = layout.capture(gradient=False)
        for a, p in zip(repeated, layout.params):
            assert torch.equal(a, p.detach().reshape(-1).cpu())

        directory = Path(directory)
        matrix = np.arange(25, dtype=np.float32).reshape(5, 5)
        matrix += matrix.T.copy()
        study_module.atomic_moment(directory/'moment.npz', dict(second=matrix))
        dist.barrier()
        np.testing.assert_array_equal(storage.load_moment(directory/'moment.npz')['second'], matrix)
        if rank == 0:
            (directory/'blocked').write_text('a file, not a directory')
        dist.barrier()
        failed = False
        try:
            study_module.atomic_moment(directory/'blocked'/'moment.npz', dict(second=matrix))
        except RuntimeError:
            failed = True
        assert all(study_module.dp.gather(failed)), 'write failure must reach every rank'
        if rank == 0:
            print('PASS: 4 CPU FSDP ranks; exact legacy dots; global counts exclude padding and tied duplicates; '
                  'identical summaries; lossless archive; all-rank I/O error propagation.', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='opd-storage-four-rank-')
    Path(directory).mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(f'tcp://127.0.0.1:{port}?use_libuv=0', directory), nprocs=4, join=True)
