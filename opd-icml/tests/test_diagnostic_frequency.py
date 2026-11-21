"""Heavy cadence changes preserve sampled energies and actual AdamW updates."""
import copy
import importlib.util
import json
import numpy as np
import pytest
import torch

from test_parameter_tokens import ROOT, make_actor, make_data, objective, optimizer_update, reference_energy, study_module
from test_parameter_sampling import cache_logits
from test_rq2_motion import configure


@pytest.mark.parametrize('reuse', [1, 2])
def test_five_update_diagnostics_keep_light_records_and_training(tmp_path, reuse):
    torch.manual_seed(71)
    actor, data = make_actor(), make_data()
    baseline = copy.deepcopy(actor)
    configure(actor, tmp_path, rounds=106//reuse, reuse=reuse)
    actor.config.policy_loss['replication17']['diagnostic_interval'] = 5
    actor.config.policy_loss['gradient_study']['parameter_tokens'] = dict(
        enabled=True, mode='kl_stratified', sample_size=4, seed=42)
    observer = study_module.StudyReplication(actor)
    due = {1, *range(5, 107, 5), 106}
    measured = []
    for step in range(1, 106//reuse+1):
        if step == (16 if reuse == 1 else 9):
            observer = study_module.StudyReplication(actor)  # restart observers after U10 exists
        observer.begin(data, None, step)
        for inner in range(reuse):
            update = (step-1)*reuse + inner+1
            cache_logits(actor, data)
            observer.before_inner(inner)
            token_measurement = observer.parameter_measurement
            assert (token_measurement is not None) == (update in due)
            if token_measurement is not None:
                measured.append(update)
                expected = reference_energy(actor, data)
                np.testing.assert_allclose(token_measurement['energy'],
                    expected[token_measurement['sampling']['index']], rtol=2e-5, atol=1e-7)
            for row, part in enumerate(data.split(1)):
                with observer.watch_gradient_scopes(part.batch, inner, {}), observer.watch_training(part.batch, inner, row):
                    (objective(actor, part)*part.batch['response_mask']).sum().div(5).backward()
            gradients = [p.grad.clone() for p in actor.actor_module.parameters()]
            observer.before_update(inner)
            for p, g in zip(actor.actor_module.parameters(), gradients):
                assert torch.equal(p.grad, g)
            # Same microbatch accumulation as the observed training path.
            for part in data.split(1):
                (objective(baseline, part)*part.batch['response_mask']).sum().div(5).backward()
            actor.actor_optimizer.step()
            actor.actor_optimizer.zero_grad(set_to_none=True)
            baseline.actor_optimizer.step()
            baseline.actor_optimizer.zero_grad(set_to_none=True)
            observer.after_inner(inner+1)
            for p, q in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
                assert torch.equal(p, q)
            record = json.loads((tmp_path/'training_records'/f'rq1_step_{update:04d}.json').read_text())
            assert record['step'] == update
            assert 'raw_gradient_energy' in record['groups']['kl_top10']['quantities']
            if update not in due:
                assert record['parameter_gradient_measurement'] == {'measured': False}
                assert record['samples'] == {} and record['parameter_gradient_rms'] is None
                assert 'parameter_gradient_energy' not in record['groups']['kl_top10']['quantities']
                assert 'token_audit/parameter_token_extra_backwards' not in observer.metrics
            else:
                assert record['parameter_gradient_measurement']['token_count'] == 4
            for kind in ('step', 'gradient'):
                assert (tmp_path/'training_records/rq2'/f'{kind}_{update:03d}.npz').exists() == (update in due)
            if update == 15:
                with np.load(tmp_path/'training_records/rq2/step_015.npz') as change:
                    assert not change['previous_step_available']
                    assert change['previous_observation_checkpoint'] == 10
                    assert change['previous_observation_update_gap'] == 5
                    assert change['previous_observation_energy'] >= 0
            (tmp_path/'training_records'/f'post_update_{update:04d}.json').write_text(
                json.dumps(dict(step=update, metrics={'actor/opd_top16_kl': .1})))
    assert measured == sorted(due) and len(measured) == 23
    spec = importlib.util.spec_from_file_location('frequency_signal_export', ROOT/'analysis/extract_signal.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    rows = [dict(zip(exporter.FIELDS, row)) for row in exporter.read_run(tmp_path, 'tiny')]
    assert len(rows) == 106
    for row in rows:
        assert row['above_initial_kl_threshold_pct'] != ''
        assert (row['parameter_token_sample_count'] != '') == (row['step'] in due)
