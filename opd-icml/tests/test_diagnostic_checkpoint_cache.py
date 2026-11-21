"""Selective recomputation preserves every gradient and actual optimizer update."""
import copy
import importlib
from functools import partial
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from test_parameter_tokens import audit_module, make_actor, make_data, opd, optimizer_update

cache = importlib.import_module(audit_module.__package__ + '.opd_diagnostic_checkpoint')
study = importlib.import_module(audit_module.__package__ + '.opd_gradient_study')


class MultiplyCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('budget', [0, 512, 1000000])
def test_bounded_cache_reduces_actual_matmuls_with_identical_gradients(monkeypatch, dtype, budget):
    torch.manual_seed(47)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 16))
    model.gradient_checkpointing = True
    original = partial(checkpoint, use_reentrant=False)
    model._gradient_checkpointing_func = original
    x = torch.randn(8, 16, requires_grad=True)
    values = []
    for cap in (0, budget):
        monkeypatch.setattr(cache, '_cache_budget', lambda _: cap)
        model.zero_grad(set_to_none=True); x.grad = None
        counter = MultiplyCounter()
        with counter, cache.diagnostic_checkpoint_cache(model) as stats:
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype==torch.bfloat16):
                y = model._gradient_checkpointing_func(model, x)
                loss = y.square().sum()
            loss.backward()
        values.append((y.detach(), x.grad.clone(), [p.grad.clone() for p in model.parameters()], counter.count))
        assert model._gradient_checkpointing_func is original
        assert stats['cached_bytes'] <= cap
    for a,b in zip(values[0][:2], values[1][:2]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(values[0][2],values[1][2]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert values[0][3]-values[1][3] == stats['reused_matmuls']
    if budget == 1000000:
        assert stats['reused_matmuls'] > 0


def make_checkpoint_actor():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    config = Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                         num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                         vocab_size=32,max_position_embeddings=64,attention_dropout=0.)
    config._attn_implementation='eager'
    actor=make_actor()
    actor.actor_module=Qwen3ForCausalLM(config)
    actor.actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
    return actor


@pytest.mark.parametrize('group_size',[1,4])
def test_token_statistics_and_next_adamw_step_unchanged(monkeypatch, group_size):
    torch.manual_seed(21)
    actor,data=make_checkpoint_actor(),make_data()
    optimizer_update(actor,data)
    baseline=copy.deepcopy(actor)
    rng=torch.random.get_rng_state().clone()
    snapshots=[]
    norm=audit_module.GradientEnergy.local_squared_norm
    def record(self,device):
        snapshots.append([p.grad.clone() for p in self.layout.params])
        return norm(self,device)
    monkeypatch.setattr(audit_module.GradientEnergy,'local_squared_norm',record)
    results=[]
    for budget in (0,1000000):
        monkeypatch.setattr(cache,'_cache_budget',lambda _:budget)
        results.append(audit_module.FullParameterTokenAudit(actor,dict(mode='exact_all',graph_group_size=group_size)).measure(data,1,0))
    for key in ('energy','diagnostic_kl','row','pos'):
        np.testing.assert_array_equal(results[0][key],results[1][key])
    for a,b in zip(snapshots[:5],snapshots[5:]):
        for ga,gb in zip(a,b):torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    assert results[1]['backwards']==5
    assert (results[1]['stage_timing']['activation_checkpoint_cache']['reused_matmuls']>0)==(group_size==1)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert actor.actor_module.is_gradient_checkpointing
    assert all(p.grad is None for p in actor.actor_module.parameters())
    optimizer_update(actor,data);optimizer_update(baseline,data)
    for a,b in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_batch_gradient_hidden_moments_and_losses_unchanged(monkeypatch):
    torch.manual_seed(22)
    actor,data=make_checkpoint_actor(),make_data()
    results=[]
    for budget in (0,1000000):
        monkeypatch.setattr(cache,'_cache_budget',lambda _:budget)
        results.append(study.gradient(actor,data,torch.eye(16)[:,:3],capture_hidden=True))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    np.testing.assert_array_equal(results[0][1],results[1][1])
    for key in results[0][2]:np.testing.assert_array_equal(results[0][2][key],results[1][2][key])


def test_exception_restores_checkpoint_function(monkeypatch):
    model=torch.nn.Linear(4,4)
    model.gradient_checkpointing=True
    original=partial(checkpoint,use_reentrant=False)
    model._gradient_checkpointing_func=original
    monkeypatch.setattr(cache,'_cache_budget',lambda _:1024)
    with pytest.raises(RuntimeError,match='injected'):
        with cache.diagnostic_checkpoint_cache(model):raise RuntimeError('injected')
    assert model._gradient_checkpointing_func is original


def test_retained_graph_and_distributed_paths_keep_original(monkeypatch):
    model=torch.nn.Linear(4,4)
    model.gradient_checkpointing=True
    original=partial(checkpoint,use_reentrant=False)
    model._gradient_checkpointing_func=original
    with cache.diagnostic_checkpoint_cache(model,single_backward=False) as stats:
        assert model._gradient_checkpointing_func is original and stats['budget_bytes']==0
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:4)
    assert cache._cache_budget(model)==0


@pytest.mark.parametrize('free_gib,has_grad,expected_gib', [(30,False,4),(10,False,1),(5,False,0),(10,True,2)])
def test_gpu_budget_reserves_missing_gradients_and_workspace(monkeypatch,free_gib,has_grad,expected_gib):
    from types import SimpleNamespace
    gib=1024**3
    parameter=SimpleNamespace(device=torch.device('cuda:0'),requires_grad=True,
                              grad=object() if has_grad else None,
                              numel=lambda:gib,element_size=lambda:4)
    model=SimpleNamespace(parameters=lambda:[parameter],modules=lambda:[])
    monkeypatch.setenv('OPD_DIAGNOSTIC_ACTIVATION_CACHE_GIB','4')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _: (free_gib*gib,80*gib))
    assert cache._cache_budget(model)==expected_gib*gib
