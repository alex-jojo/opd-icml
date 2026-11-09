"""Four-process CPU check of the rank-independent OPD collective math."""
import importlib.util
import json
from pathlib import Path
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]


class Batch:
    def __init__(self, value):
        self.value = value

    def __len__(self):
        return len(self.value)

    def select_idxs(self, indices):
        return Batch(self.value[indices])

    @classmethod
    def concat(cls, batches):
        return cls(torch.cat([batch.value for batch in batches]))


def worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=45))
    try:
        path = ROOT / 'runtime/verl/workers/actor/opd_distributed.py'
        spec = importlib.util.spec_from_file_location('opd_dp_cpu', path)
        dp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dp)
        torch.manual_seed(42)
        data = torch.randn(8, 4)
        teacher = torch.randn(8, 16)
        local = data[rank * 2:rank * 2 + 2]
        batch = dp.global_batch(Batch(local))
        torch.testing.assert_close(batch.value, data)
        shard, offset = dp.shard(batch)
        assert offset == rank * 2
        torch.testing.assert_close(shard.value, local)
        second, count = dp.sum_gram(local.T @ local, len(local))
        assert count == len(data)
        torch.testing.assert_close(second, data.T @ data)
        assert dp.total(float(rank)) == 6.
        model = torch.nn.Linear(4, 16, bias=False)
        expected = torch.nn.Linear(4, 16, bias=False)
        expected.load_state_dict(model.state_dict())

        def reverse_kl(student, target):
            lp, lq = student.log_softmax(-1), target.log_softmax(-1)
            return (lp.exp() * (lp - lq)).sum(-1)

        loss = reverse_kl(model(local), teacher[offset:offset + len(local)]).sum() * world / len(data)
        loss.backward()
        dist.all_reduce(model.weight.grad)
        model.weight.grad.div_(world)
        reverse_kl(expected(data), teacher).mean().backward()
        torch.testing.assert_close(model.weight.grad, expected.weight.grad, atol=1e-7, rtol=1e-5)
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        torch.optim.AdamW(expected.parameters(), lr=1e-4).step()
        torch.testing.assert_close(model.weight, expected.weight, atol=1e-7, rtol=1e-5)
        if rank == 0:
            print(json.dumps(dict(cpu_ranks=world, global_batch=True, hidden_moments=True,
                                  reverse_kl_gradient=True, adam_update=True)))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    rendezvous = f'tcp://127.0.0.1:{port}?use_libuv=0'
    mp.spawn(worker, args=(4, rendezvous), nprocs=4, join=True)
    print('PASS: four CPU ranks; no CUDA or model downloads.')
