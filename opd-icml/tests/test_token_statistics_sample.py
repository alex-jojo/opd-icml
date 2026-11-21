"""Bounded token diagnostics must not subsample RQ2 or alter optimization."""
import copy
import importlib
import json
import numpy as np
import pytest
import torch
from test_parameter_tokens import make_actor, make_data, objective, study_module, audit_module
from test_parameter_sampling import cache_logits, sampling
from test_rq2_motion import configure

sample = importlib.import_module(audit_module.__package__+'.opd_token_sample')


@pytest.mark.parametrize('population', [129, 200, 1000])
def test_nested_sampling_preserves_parameter_tokens_and_weights(population):
    kl = np.arange(population, dtype=float)
    base = sampling.stratified_plan(kl, 64, 42, 5, 0)
    rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    plan = sample.extend_plan(kl, base, 128, 5, 0)
    assert len(plan['index']) == 128
    assert set(base['index']) <= set(plan['index'])
    assert len(set(plan['index'])) == 128
    assert plan['weight'].sum() == pytest.approx(population)
    for label, group in enumerate(plan['strata']):
        assert plan['weight'][plan['stratum']==label].sum() == pytest.approx(group['population_count'])
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
    assert torch.equal(torch_rng, torch.get_rng_state())
    np.testing.assert_array_equal(plan['index'], sample.extend_plan(kl, base, 128, 5, 0)['index'])


def test_thin_weighted_spectra_match_direct_hidden_gram():
    rng = np.random.default_rng(27)
    g = rng.normal(size=(17, 150))
    u = np.linalg.qr(rng.normal(size=(150, 8)))[0]
    w = rng.uniform(3, 10, len(g))
    stats, spectra = sample.sampled_scopes(g, u, w, 500)
    for name, x in [('full',g),('u',g@u),('outside',g-(g@u)@u.T)]:
        expected = np.maximum(np.linalg.eigvalsh(x.T@(w[:,None]*x)/500)[::-1], 0)
        if name == 'outside':expected=expected[:142]
        np.testing.assert_allclose(spectra[name+'_eigenvalues'], expected, rtol=1e-9, atol=1e-12)
        assert stats[name]['sample_rank_limit'] == min(17, x.shape[1] if name!='outside' else 142)


def test_weighted_concentration_census_matches_existing():
    old = importlib.import_module(audit_module.__package__+'.opd_focus_stats')
    x = np.arange(100, dtype=float)
    actual = sample.concentration(x, np.ones(100))
    expected = old.concentration(x)
    for key in ('total','gini','effective_fraction','entropy_effective_fraction','relative_hhi'):
        assert actual[key] == pytest.approx(expected[key])
    assert actual['top_shares'] == pytest.approx(expected['top_shares'])


def run_backward(actor, data, observer=None, inner=0):
    total = int(data.batch['response_mask'].sum())
    for row, part in enumerate(data.split(1)):
        if observer is None:
            (objective(actor,part)*part.batch['response_mask']).sum().div(total).backward()
        else:
            with observer.watch_gradient_scopes(part.batch,inner,{}), observer.watch_training(part.batch,inner,row):
                (objective(actor,part)*part.batch['response_mask']).sum().div(total).backward()


def test_sampling_retains_full_moment_parameter_sample_and_adamw(tmp_path):
    torch.manual_seed(91)
    actor = make_actor()
    baseline = copy.deepcopy(actor)
    data = make_data().select_idxs([0,1]*40)  # 200 valid tokens, many empty sampled microbatches.
    configure(actor,tmp_path,rounds=106)
    actor.config.policy_loss['replication17']['diagnostic_interval']=5
    actor.config.policy_loss['gradient_study']['parameter_tokens']=dict(enabled=True,mode='kl_stratified',sample_size=64,seed=42)
    observer=study_module.StudyReplication(actor)
    cache_logits(actor,data)
    observer.begin(data,None,1)
    observer.before_inner(0)
    parameter=copy.deepcopy(observer.parameter_measurement)
    plan=copy.deepcopy(observer.token_sample.plan)
    assert len(parameter['energy'])==64 and len(plan['index'])==128
    run_backward(actor,data,observer)
    rows=[]
    norm=study_module._final_norm_module(baseline.actor_module)
    def hook(_m,_a,h):
        def backward(g):
            i=len(rows)
            mask=data.batch['response_mask'][i].bool()
            rows.append(g[0,-4:-1][mask].detach().float().numpy()*200)
        h.register_hook(backward)
    handle=norm.register_forward_hook(hook)
    run_backward(baseline,data)
    handle.remove()
    g=np.concatenate(rows)
    original=[p.grad.clone() for p in actor.actor_module.parameters()]
    observer.before_update(0)
    for p,q,old in zip(actor.actor_module.parameters(),baseline.actor_module.parameters(),original):
        assert torch.equal(p.grad,q.grad) and torch.equal(p.grad,old)
    np.testing.assert_allclose(observer.pending_gradient['second'],g.T@g/200,rtol=2e-5,atol=1e-8)
    assert observer.pending_gradient['count']==200
    record=json.loads((tmp_path/'training_records/rq1_step_0001.json').read_text())
    from test_opd_statistics_reuse import assert_no_removed_scores
    detailed=json.loads((tmp_path/'replication/token_audit/step_001_inner_1.json').read_text())
    assert_no_removed_scores(detailed)
    assert sum('_top' in key for key in detailed['groups'])==12
    with np.load(tmp_path/'replication/token_audit/step_001_inner_1.npz') as saved:
        assert_no_removed_scores({key: None for key in saved.files})
    assert record['token_statistics_measurement']['sample_count']==128
    assert record['parameter_gradient_measurement']['token_count']==64
    assert record['parameter_gradient_measurement']['population_token_count']==200
    np.testing.assert_array_equal(record['samples']['parameter_gradient_energy'],parameter['energy'])
    with np.load(tmp_path/'training_records/rq1_hidden_sample_0001.npz') as arrays:
        assert len(arrays['kl'])==128
        np.testing.assert_allclose(arrays['raw_gradient_energy'],np.square(g[plan['index']]).sum(1),rtol=2e-6)
        high=sample.top_mask(np.concatenate([objective(baseline,p).detach().numpy()[p.batch['response_mask'].bool().numpy()] for p in data.split(1)]),.1)
        expected=(plan['weight']*np.square(g[plan['index']]).sum(1))
        assert record['groups']['kl_top10']['quantities']['raw_gradient_energy']['share']==pytest.approx(expected[high[plan['index']]].sum()/expected.sum(),rel=1e-5)
    actor.actor_optimizer.step();baseline.actor_optimizer.step()
    actor.actor_optimizer.zero_grad(set_to_none=True);baseline.actor_optimizer.zero_grad(set_to_none=True)
    for p,q in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):assert torch.equal(p,q)
    observer.after_inner(1)
    # Light step keeps 128 hidden samples, no new parameter diagnostic.
    observer.begin(data,None,2);cache_logits(actor,data);observer.before_inner(0)
    assert observer.parameter_measurement is None
    run_backward(actor,data,observer);observer.before_update(0)
    record=json.loads((tmp_path/'training_records/rq1_step_0002.json').read_text())
    assert record['parameter_gradient_measurement']=={'measured':False}
    assert record['samples']=={}
    assert record['token_statistics_measurement']['sample_count']==128
    actor.actor_optimizer.zero_grad(set_to_none=True)
    restored=study_module.StudyReplication(actor)
    restored.begin(data,None,3);cache_logits(actor,data);restored.before_inner(0)
    assert restored.thresholds==observer.thresholds
    assert len(restored.token_sample.plan['index'])==128
    run_backward(actor,data,restored);restored.before_update(0)
    assert json.loads((tmp_path/'training_records/rq1_step_0003.json').read_text())['thresholds']==record['thresholds']


def test_reuse_inner_update_uses_new_sample_and_current_population_kl(tmp_path):
    torch.manual_seed(42)
    actor=make_actor();data=make_data().select_idxs([0,1]*40)
    configure(actor,tmp_path,rounds=53,reuse=2)
    actor.config.policy_loss['replication17']['diagnostic_interval']=5
    actor.config.policy_loss['gradient_study']['parameter_tokens']=dict(enabled=True,mode='kl_stratified',sample_size=64,seed=42)
    observer=study_module.StudyReplication(actor)
    cache_logits(actor,data);observer.begin(data,None,1);observer.before_inner(0)
    plan=observer.token_sample.plan['index'].copy()
    run_backward(actor,data,observer);observer.before_update(0)
    actor.actor_optimizer.step();actor.actor_optimizer.zero_grad(set_to_none=True)
    # Reuse keeps the cached targets/logits; the head hook reads current KL.
    observer.before_inner(1)
    assert observer.parameter_measurement is None
    assert not np.array_equal(plan,observer.token_sample.plan['index'])
    run_backward(actor,data,observer,inner=1);observer.before_update(1)
    record=json.loads((tmp_path/'training_records/rq1_step_0002.json').read_text())
    all_kl=sample.output_scores(opd_logits(actor,data),data.batch['opd_teacher_top16_logits'],kl_only=True)['kl'].numpy()[data.batch['response_mask'].bool().numpy()]
    assert record['groups']['kl_above_step1_p90']['fraction']==pytest.approx(np.mean(all_kl>=observer.thresholds['kl_p90']))


def opd_logits(actor,data):
    from test_parameter_tokens import opd
    with torch.no_grad():return opd.forward(actor,data.batch,1.,data.batch['opd_top16_ids'],diagnostics=False)[1]
