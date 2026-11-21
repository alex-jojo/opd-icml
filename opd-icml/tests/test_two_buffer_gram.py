"""Exact pair coverage with bounded live buffers and fewer oracle calls."""
import copy
import gc
import importlib
import weakref

import numpy as np
import pytest
import torch

from test_parameter_tokens import audit_module, make_actor, make_data, optimizer_update

study = importlib.import_module(audit_module.__package__ + '.opd_gradient_study')


@pytest.mark.parametrize('count,slots,calls,limit', [(2,0,2,2), (3,0,4,2),
                                                  (4,0,7,2), (4,2,5,3), (4,3,4,4)])
def test_every_entry_identical_and_live_vector_bound(monkeypatch,count,slots,calls,limit):
    torch.manual_seed(811)
    actor = make_actor(tied=True)
    layout = study.ParameterLayout(actor.actor_module)
    # Multiple shapes and groups; values aren't simple integers or orthogonal.
    source = [[torch.randn(p.numel()) for p in layout.params] for _ in range(count)]
    expected = {}
    for i in range(count):
        for j in range(i,count):
            for group,value in layout.dot(source[i],source[j]).items():
                matrix=expected.setdefault(group,np.zeros((count,count),np.float64))
                matrix[i,j]=matrix[j,i]=value
    seen=[]; references=[]; peak=0
    def oracle(i):
        nonlocal peak
        gc.collect()
        # Release obsolete vectors before allocating their replacement.
        assert sum(ref() is not None for ref in references) < limit
        result=[x.clone() for x in source[i]]
        references.append(weakref.ref(result[0]))
        peak=max(peak,sum(ref() is not None for ref in references))
        seen.append(i)
        return result
    monkeypatch.setattr(study,'_gradient_cache_slots',lambda *_:slots)
    actual,n=study.exact_gram(oracle,layout,count)
    assert n==calls==len(seen) and peak==limit
    for group in expected:np.testing.assert_array_equal(actual[group],expected[group])
    assert all(ref() is None for ref in references)


def test_real_gradients_against_original_replays_and_optimizer(monkeypatch):
    torch.manual_seed(812)
    actor,data=make_actor(tied=True),make_data()
    optimizer_update(actor,data)
    baseline=copy.deepcopy(actor)
    layout=study.ParameterLayout(actor.actor_module)
    basis=torch.eye(4)[:,:2]
    batches=[copy.deepcopy(data) for _ in range(4)]
    for i,batch in enumerate(batches):
        batch.batch['opd_teacher_top16_logits']=torch.roll(batch.batch['opd_teacher_top16_logits'],i,-1)
    calls=0
    def oracle(i):
        nonlocal calls
        calls+=1
        return study.gradient(actor,batches[i],basis)[0]
    expected={}
    # Original uncached ordering and arithmetic, ten forward/backward batches.
    for i in range(4):
        a=oracle(i)
        for j in range(i,4):
            b=a if i==j else oracle(j)
            for group,value in layout.dot(a,b).items():
                matrix=expected.setdefault(group,np.zeros((4,4),np.float64))
                matrix[i,j]=matrix[j,i]=value
    assert calls==10
    calls=0
    monkeypatch.setattr(study,'_gradient_cache_slots',lambda *_:0)
    actual,n=study.exact_gram(oracle,layout,4)
    assert calls==n==7
    for group in expected:np.testing.assert_array_equal(actual[group],expected[group])
    assert all(p.grad is None for p in actor.actor_module.parameters())
    optimizer_update(actor,data);optimizer_update(baseline,data)
    for actual,wanted in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0)


@pytest.mark.parametrize('normalization',['token','response'])
def test_skip_unused_loss_readouts_preserves_gradient_and_hidden(normalization):
    torch.manual_seed(814)
    actor,data=make_actor(tied=True),make_data()
    basis=torch.eye(4)[:,:2]
    kwargs=dict(basis=basis,normalization=normalization,capture_hidden=True)
    a,per_a,mom_a=study.gradient(actor,data,collect_losses=True,**kwargs)
    b,per_b,mom_b=study.gradient(actor,data,collect_losses=False,**kwargs)
    assert per_b is None
    for actual,wanted in zip(a,b):torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    for key in mom_a:np.testing.assert_array_equal(mom_a[key],mom_b[key])
    # Compare to the OLD no-grad per-microbatch CPU readout. Grad-enabled vs
    # no-grad LayerNorm can select different CPU kernels even before this change.
    old=[]
    with torch.no_grad(), study.observe_only(actor):
        for part in data.split(1):
            x=part.batch
            _,z,_,_=study.forward(actor,x,1.,x['opd_top16_ids'],diagnostics=False)
            mask=x['response_mask'];kl=study.reverse_kl(z,x['opd_teacher_top16_logits'])
            old.extend(((kl*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist())
    np.testing.assert_array_equal(study.losses(actor,data),np.asarray(old))
