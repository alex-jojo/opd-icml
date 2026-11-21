"""Single-rank nested CPU FSDP + Qwen3 + diagnostic selective checkpointing."""
import copy
from datetime import timedelta
from functools import partial
import socket
import numpy as np
import torch
import torch.distributed as dist
from test_diagnostic_checkpoint_cache import make_checkpoint_actor, cache, study
from test_parameter_tokens import audit_module,make_data,optimizer_update


def main():
    torch.set_num_threads(1)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=0,world_size=1,
                            timeout=timedelta(seconds=60))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        for mixed in (False,True):
            torch.manual_seed(88)
            actor,data=make_checkpoint_actor(),make_data()
            template=copy.deepcopy(actor.actor_module)
            def wrap(model):
                return FSDP(model,device_id=torch.device('cpu'),use_orig_params=False,
                    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.bfloat16) if mixed else None,
                    auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen3DecoderLayer}))
            actor.actor_module=wrap(actor.actor_module)
            actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
            baseline=make_checkpoint_actor()
            baseline.actor_module=wrap(template)
            baseline.actor_optimizer=torch.optim.AdamW(baseline.actor_module.parameters(),lr=.001)
            optimizer_update(actor,data);optimizer_update(baseline,data)
            state=copy.deepcopy(actor.actor_optimizer.state_dict())
            results=[];gradients=[]
            originals=[(m,m.__dict__.get('_gradient_checkpointing_func')) for m in actor.actor_module.modules()]
            for cap in (0,1000000):
                cache._cache_budget=lambda _:cap
                results.append(audit_module.FullParameterTokenAudit(actor,dict(graph_group_size=1)).measure(data,1,0))
                gradients.append(study.gradient(actor,data,torch.eye(16)[:,:3],capture_hidden=True))
            assert results[1]['stage_timing']['activation_checkpoint_cache']['reused_matmuls']>0
            for key in ('energy','diagnostic_kl','row','pos'):np.testing.assert_array_equal(results[0][key],results[1][key])
            for a,b in zip(gradients[0][0],gradients[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            np.testing.assert_array_equal(gradients[0][1],gradients[1][1])
            for k in gradients[0][2]:np.testing.assert_array_equal(gradients[0][2][k],gradients[1][2][k])
            for m,original in originals:assert m.__dict__.get('_gradient_checkpointing_func') is original
            for i,values in state['state'].items():
                for k,v in values.items():torch.testing.assert_close(v,actor.actor_optimizer.state_dict()['state'][i][k],rtol=0,atol=0)
            optimizer_update(actor,data);optimizer_update(baseline,data)
            for a,b in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            print(f'PASS: single-rank nested CPU FSDP, BF16={mixed}; identical token and batch statistics and AdamW update',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
