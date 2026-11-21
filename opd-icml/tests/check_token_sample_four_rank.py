"""CPU/Gloo FSDP check, including ranks with no valid sampled tokens."""
from datetime import timedelta
import copy
import json
import socket
import tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_parameter_tokens import make_actor, make_data, objective, study_module
from test_parameter_sampling import cache_logits
from test_rq2_motion import configure


def worker(rank, address, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=4,
                            timeout=timedelta(seconds=150))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        torch.manual_seed(391)
        actor=make_actor();baseline=copy.deepcopy(actor)
        data=make_data().select_idxs([0,1]*160)
        data.batch['response_mask'][80:]=0  # ranks 1--3 have empty local samples.
        cache_logits(baseline,data)
        configure(actor,__import__('pathlib').Path(output),rounds=106)
        actor.config.policy_loss['replication17']['diagnostic_interval']=5
        actor.config.policy_loss['gradient_study']['parameter_tokens']=dict(enabled=True,mode='kl_stratified',sample_size=64,seed=42)
        actor.actor_module=FSDP(actor.actor_module, device_id=torch.device('cpu'),use_orig_params=False)
        actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
        observer=study_module.StudyReplication(actor)
        observer.begin(data,None,1);observer.before_inner(0)
        local=data.select_idxs(list(range(rank*80,(rank+1)*80)))
        with observer.watch_gradient_scopes(local.batch,0,{}),observer.watch_training(local.batch,0,rank*80):
            (objective(actor,local)*local.batch['response_mask']).sum().mul(4/200).backward()
        saved=[p.grad.clone() for p in actor.actor_module.parameters()]
        observer.before_update(0)
        for p,g in zip(actor.actor_module.parameters(),saved):assert torch.equal(p.grad,g)
        (objective(baseline,data)*data.batch['response_mask']).sum().div(200).backward()
        actor.actor_optimizer.step();baseline.actor_optimizer.step()
        with FSDP.summon_full_params(actor.actor_module):
            for p,q in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
                torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-7)
        if rank==0:
            from pathlib import Path
            record=json.loads((Path(output)/'training_records/rq1_step_0001.json').read_text())
            assert record['token_statistics_measurement']['sample_count']==128
            assert record['parameter_gradient_measurement']['token_count']==64
            assert observer.pending_gradient['count']==200
            with np.load(Path(output)/'training_records/rq1_hidden_sample_0001.npz') as f:
                assert len(f['index'])==128 and len(set(f['index']))==128
            print('PASS: four-rank FSDP, 128 hidden / 64 parameter, empty local shards, unchanged AdamW',flush=True)
        dist.barrier()
        actor.actor_optimizer.zero_grad(set_to_none=True)
        observer.begin(data,None,2);observer.before_inner(0)
        with observer.watch_gradient_scopes(local.batch,0,{}),observer.watch_training(local.batch,0,rank*80):
            (objective(actor,local)*local.batch['response_mask']).sum().mul(4/200).backward()
        original=study_module.atomic_npz
        if rank==0:
            def fail(path,arrays):
                if 'rq1_hidden_sample' in str(path):raise OSError('sample output write failed')
                return original(path,arrays)
            study_module.atomic_npz=fail
        caught=False
        try:observer.before_update(0)
        except Exception as error:caught='sample output write failed' in str(error)
        finally:study_module.atomic_npz=original
        assert caught
        dist.barrier()
        if rank==0:print('PASS: sampled-record write failure reaches all four ranks',flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='opd-token-sample-') as output:
        mp.spawn(worker,args=(f'tcp://127.0.0.1:{port}',output),nprocs=4,join=True)
