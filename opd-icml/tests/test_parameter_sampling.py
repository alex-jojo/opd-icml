"""Stratified weights, exact sampled Jacobians and unchanged training updates."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_parameter_tokens import (ROOT, audit_module, study_module, opd, make_actor,
                                   make_data, objective, reference_energy, optimizer_update)

sampling = importlib.import_module(audit_module.__package__ + '.opd_parameter_sampling')


def cache_logits(actor, data):
    with torch.no_grad():
        _, logits, _, _ = opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False)
    data.batch['opd_student_top16_logits'] = logits.detach()


def test_64_192_allocation_and_deterministic_sampling_from_whole_batch():
    kl = np.arange(392216, dtype=float)
    plan = sampling.stratified_plan(kl, 256, 42, 1, 0)
    assert [s['sample_count'] for s in plan['strata']] == [64, 192]
    assert len(set(plan['index'])) == 256
    assert np.all(np.diff(plan['index']) > 0)
    assert np.all(kl[plan['index'][plan['stratum'] == 0]] >= 352994)
    assert plan['weight'].sum() == pytest.approx(len(kl))
    again = sampling.stratified_plan(kl, 256, 42, 1, 0)
    np.testing.assert_array_equal(plan['index'], again['index'])
    assert not np.array_equal(plan['index'], sampling.stratified_plan(kl, 256, 42, 5, 0)['index'])


def test_weighted_population_share_corrects_deliberate_high_kl_oversampling():
    kl = np.arange(100, dtype=float)
    plan = sampling.stratified_plan(kl, 16)
    high = plan['stratum'] == 0
    energy = np.where(high, 9., 1.)
    assert energy[high].sum() / energy.sum() == pytest.approx(.75)
    report = sampling.weighted_energy(energy, high, 10, plan)
    assert report['sum'] == pytest.approx(90.)
    assert report['share'] == pytest.approx(.5)
    assert report['mean'] == pytest.approx(9.)
    assert report['share_ci95'] == pytest.approx([.5, .5])
    missing = sampling.weighted_energy(energy, np.zeros(16, bool), 1, plan)
    assert missing['share'] is None and missing['share_ci95'] is None


@pytest.mark.parametrize('count,budget', [(1,256), (5,256), (19,18), (100,2)])
def test_small_populations_and_tiny_strata_have_no_duplicate_or_zero_probability_tokens(count,budget):
    plan = sampling.stratified_plan(np.zeros(count), budget)
    assert len(plan['index']) == min(count,budget)
    assert len(set(plan['index'])) == len(plan['index'])
    assert plan['weight'].sum() == pytest.approx(count)
    if count <= budget:
        np.testing.assert_array_equal(plan['index'], np.arange(count))
        np.testing.assert_array_equal(plan['weight'], np.ones(count))


def test_sampled_gradients_match_full_jacobian_and_leave_training_identical():
    torch.manual_seed(71)
    actor, data = make_actor(tied=True), make_data()
    optimizer_update(actor, data)
    cache_logits(actor, data)
    expected = reference_energy(actor, data)
    baseline = copy.deepcopy(actor)
    rng = torch.random.get_rng_state().clone()
    result = audit_module.FullParameterTokenAudit(actor, dict(mode='kl_stratified', sample_size=4, seed=42)).measure(data,1,0)
    np.testing.assert_allclose(result['energy'], expected[result['sampling']['index']], rtol=2e-5, atol=1e-7)
    assert result['backwards'] == 4
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in actor.actor_module.parameters())
    optimizer_update(actor, data)
    optimizer_update(baseline, data)
    for actual, original in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_sampled_study_preserves_full_hidden_population_and_saves_identical_paired_ids(tmp_path, monkeypatch):
    audit = importlib.import_module(audit_module.__package__ + '.opd_token_audit')
    original_prepare = audit.prepare_groups
    preparations = []
    def prepare(*args):
        prepared = original_prepare(*args)
        preparations.append(prepared)
        return prepared
    monkeypatch.setattr(audit, 'prepare_groups', prepare)
    monkeypatch.setattr(study_module, 'prepare_groups', prepare)
    torch.manual_seed(42)
    actor, data = make_actor(tied=True), make_data()
    actor.actor_module.config.opd_response_eos_token_ids=[15,6]
    cache_logits(actor, data)
    cfg = actor.config.policy_loss
    cfg['replication17'].update(mode='full', output_dir=str(tmp_path/'replication'), rank=2,
        freeze_step=10, interval=5, rounds=6, reuse=1, tokenizer_path=None)
    cfg['hidden_projection'] = dict(asset_path=str(tmp_path/'basis.npz'))
    cfg['gradient_study'] = dict(enabled=True, parameter_tokens=dict(enabled=True,mode='kl_stratified',sample_size=4,seed=42))
    replication = study_module.StudyReplication(actor)
    replication.begin(data,None,1)
    replication.before_inner(0)
    measured = replication.parameter_measurement
    cursor = 0
    for part in data.split(1):
        with replication.watch_gradient_scopes(part.batch,0,{}), replication.watch_training(part.batch,0,cursor):
            (objective(actor,part)*part.batch['response_mask']).sum().div(5).backward()
        cursor += len(part)
    gradients = [p.grad.clone() for p in actor.actor_module.parameters()]
    replication.before_update(0)
    assert len(preparations) == 1  # full and sampled reports share the same groups
    for actual, original in zip(actor.actor_module.parameters(), gradients):
        assert torch.equal(actual.grad, original)
    directory = tmp_path/'replication/token_audit'
    report = json.loads((directory/'step_001_inner_1.json').read_text())
    metadata = report['parameter_gradient_measurement']
    assert metadata['sampled'] and not metadata['exact']
    assert metadata['individual_token_gradients_exact']
    assert metadata['token_count'] == 4 and metadata['population_token_count'] == 5
    assert metadata['forward_calls'] == len(set(measured['row']))
    assert metadata['graph_group_size'] == 4
    with np.load(directory/'step_001_inner_1.npz') as full, np.load(directory/'parameter_sample_001_inner_1.npz') as paired:
        assert len(full['row']) == 5 and len(paired['row']) == 4
        np.testing.assert_array_equal(full['eos'],np.isin(full['token'],[15,6]))
        assert 'parameter_gradient_energy' not in full
        lookup = {(int(r),int(p)):i for i,(r,p) in enumerate(zip(full['row'],full['pos']))}
        for r,p,e in zip(paired['row'],paired['pos'],paired['raw_gradient_energy']):
            assert e == full['raw_gradient_energy'][lookup[(int(r),int(p))]]
        np.testing.assert_allclose(paired['parameter_gradient_energy'],measured['energy'])
        total = float(paired['sampling_weight'] @ paired['parameter_gradient_energy'])
        assert report['parameter_gradient_frobenius'] == pytest.approx(np.sqrt(total))
        assert report['parameter_gradient_rms'] == pytest.approx(np.sqrt(total/5))
        assert report['gradient_frobenius_full'] == pytest.approx(np.sqrt(full['raw_gradient_energy'].sum()))
    assert report['paired_parameter_hidden_sample']['sample_count'] == 4
    records = tmp_path/'training_records'
    records.mkdir()
    (records/'post_update_0001.json').write_text(json.dumps(dict(step=1,metrics={'actor/opd_top16_kl':.1})))
    spec = importlib.util.spec_from_file_location('sampled_signal_export',ROOT/'analysis/extract_signal.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    row = dict(zip(exporter.FIELDS,next(exporter.read_run(tmp_path,'tiny'))))
    assert row['parameter_token_mode'] == 'kl_stratified'
    assert row['parameter_token_sampled'] is True
    assert row['parameter_token_sample_count'] == 4 and row['parameter_token_population_count'] == 5


def test_default_deepseek_code_and_cli_sampling_compose_without_changing_training_budget():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    command = [sys.executable,str(ROOT/'train.py'),'--scale','deepseek-coder-1.3b','--task','code',
        '--experiment','paired-full','--hardware','a100x1','--parameter-token-gradients','sampled',
        '--parameter-token-samples','256','--dry-run']
    report = json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout)
    assert report['parameter_token_gradients'] == dict(enabled=True,mode='kl_stratified',sample_size=256,seed=42)
    assert report['rollout_batch'] == 128 and report['rounds_cap'] == 106
    assert report['gpu_count'] == 1
    import shlex
    overrides = shlex.split(report['command'])[shlex.split(report['command']).index('--config-name')+2:]
    overrides = [o for o in overrides if o.startswith('++')]
    search = 'hydra.searchpath='+json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()],ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs/deepseek-coder-1.3b'),version_base=None):
        cfg=compose(config_name='code-paired-full',overrides=[search]+overrides)
    assert OmegaConf.to_container(cfg.actor_rollout_ref.actor.policy_loss.gradient_study.parameter_tokens) == report['parameter_token_gradients']
    bad = subprocess.run(command[:-1]+['--parameter-token-samples','1','--dry-run'],capture_output=True,text=True)
    assert bad.returncode != 0
