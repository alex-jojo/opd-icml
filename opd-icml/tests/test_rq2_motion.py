import json
import sys
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from test_parameter_tokens import ROOT, make_actor, make_data, objective, optimizer_update, study_module

sys.path.insert(0, str(ROOT / 'scripts'))
from package_results import package_results


def configure(actor, run, rounds=106, reuse=1, adaptive=False):
    policy = actor.config.policy_loss
    policy['replication17'].update(mode='full', output_dir=str(run/'replication'), rank=2,
        freeze_step=10, interval=5, rounds=rounds, reuse=reuse, tokenizer_path=None,
        adaptive_stability={'enabled': adaptive})
    policy['hidden_projection'] = dict(asset_path=str(run/'basis.npz'))
    policy['gradient_study'] = dict(enabled=True)


def test_full_motion_all_106_updates_and_archived_baseline(tmp_path):
    torch.manual_seed(33)
    run = tmp_path/'training'
    actor, data = make_actor(), make_data()
    configure(actor, run)
    observer = study_module.StudyReplication(actor)
    directory = run/'training_records/rq2'
    early = None
    for update in range(1, 107):
        if update == 14:
            observer = study_module.StudyReplication(actor)  # resume after the reference step
        observer.begin(data if update == 1 else data.select_idxs([1, 0]), None, update)
        if update == 1:
            with np.load(directory/'probe_initial.npz') as initial:
                h0 = initial['initial_hidden']
                np.testing.assert_array_equal(initial['input_ids'], data.batch['input_ids'])
                np.testing.assert_array_equal(initial['row'], [0, 0, 1, 1, 1])
                np.testing.assert_array_equal(initial['response_position'], [0, 2, 0, 1, 2])
        optimizer_update(actor, data)
        observer.after_inner(1)
        with np.load(directory/f'step_{update:03d}.npz') as record:
            h = study_module.panel_hidden(actor, data).numpy()
            delta = h/np.linalg.norm(h, axis=1, keepdims=True) - h0/np.linalg.norm(h0, axis=1, keepdims=True)
            second = delta.T @ delta / len(h)
            spectrum = np.linalg.eigvalsh(second)[::-1].clip(0)
            np.testing.assert_allclose(record['eigenvalues'], spectrum, rtol=2e-3, atol=1e-8)
            np.testing.assert_allclose(record['cumulative_energy_coverage'],
                                       spectrum.cumsum()/spectrum.sum(), atol=3e-5)
            assert record['raw_displacement_rms'] == pytest.approx(np.sqrt(((h-h0)**2).sum(1).mean()))
            assert record['unit_displacement_rms'] == pytest.approx(np.sqrt(np.trace(second)), rel=5e-5)
            basis = record['basis']
            assert int(record['step']) == update
            assert bool(record['fixed_u10_available']) == (update >= 10)
            if update == 10:
                early = basis.copy()
            if update < 10:
                assert 'fixed_u10_coverage' not in record and 'rq2/fixed_u10_coverage' not in observer.metrics
            else:
                expected = np.trace(early.T @ second @ early) / np.trace(second)
                assert record['fixed_u10_coverage'] == pytest.approx(expected, abs=3e-5)
                assert record['fixed_u10_overlap'] == pytest.approx(((early.T @ basis)**2).sum()/2, abs=3e-5)
                assert observer.metrics['rq2/fixed_u10_coverage'] == float(record['fixed_u10_coverage'])
    assert len(list((run/'replication/motion').glob('step_*.npz'))) == 22
    with ZipFile(package_results(run)) as archive:
        prefix = 'training/training_records/rq2/'
        assert prefix+'probe_initial.npz' in archive.namelist()
        assert sorted(n for n in archive.namelist() if n.startswith(prefix+'step_')) == [
            prefix+f'step_{i:03d}.npz' for i in range(1, 107)]


def test_inner_updates_zero_motion_and_sparse_stability(tmp_path):
    torch.manual_seed(41)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path, rounds=8, reuse=2, adaptive=True)
    obs = study_module.StudyReplication(actor)
    directory = tmp_path/'training_records/rq2'
    for step in range(1, 9):
        obs.begin(data, None, step)
        for inner in (1, 2):
            update = 2*(step-1)+inner
            if update > 1:
                optimizer_update(actor, data)
            obs.after_inner(inner)
    with np.load(directory/'step_001.npz') as first:
        assert first['unit_displacement_rms'] == 0
        assert first['basis'].shape == (4, 0) and first['cumulative_energy_coverage'].size == 0
    assert len(list(directory.glob('step_*.npz'))) == 16
    decisions = sorted((tmp_path/'replication/stability').glob('step_*.json'))
    assert [json.loads(p.read_text())['step'] for p in decisions] == [5, 10, 15]
    assert json.loads(decisions[-1].read_text())['steps'] == [5, 10, 15]
    with np.load(directory/'step_010.npz') as early, np.load(directory/'step_016.npz') as late:
        assert late['fixed_u10_overlap'] == pytest.approx(((early['basis'].T @ late['basis'])**2).sum()/2)


def test_legacy_step10_basis_is_used_on_resume(tmp_path):
    torch.manual_seed(12)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path)
    obs = study_module.StudyReplication(actor)
    obs.begin(data, None, 1)
    legacy = tmp_path/'replication/motion'
    legacy.mkdir(parents=True)
    early = torch.eye(4)[:, [1, 3]].numpy()
    np.savez(legacy/'step_010.npz', basis=early)
    obs = study_module.StudyReplication(actor)
    obs.begin(data, None, 11)
    optimizer_update(actor, data)
    obs.after_inner(1)
    with np.load(tmp_path/'training_records/rq2/step_011.npz') as record:
        assert record['fixed_u10_overlap'] == pytest.approx(((early.T @ record['basis'])**2).sum()/2)


def test_training_gradients_all_106_updates_use_the_same_u10(tmp_path):
    torch.manual_seed(25)
    run = tmp_path/'training'
    actor, data = make_actor(), make_data()
    configure(actor, run, rounds=53, reuse=2, adaptive=True)
    observer = study_module.StudyReplication(actor)
    directory = run/'training_records/rq2'
    mask = data.batch['response_mask'].bool()
    fixed = None
    for step in range(1, 54):
        if step == 9:  # resume after adaptive stability installed a newer basis at 15
            observer = study_module.StudyReplication(actor)
        observer.begin(data, None, step)
        for inner in range(2):
            update = (step-1)*2 + inner+1
            hidden = []
            with actor.actor_module.model.norm.register_forward_hook(lambda m, a, h: hidden.append(h)):
                loss = objective(actor, data)[mask].sum()
            (grad,) = torch.autograd.grad(loss, hidden[0])
            g = grad[:, -mask.shape[1]-1:-1][mask].double().numpy()
            expected = np.linalg.svd(g, compute_uv=False)**2 / len(g)
            observer.before_inner(inner)
            for row, part in enumerate(data.split(1)):
                with observer.watch_gradient_scopes(part.batch, inner, {}), observer.watch_training(part.batch, inner, row):
                    (objective(actor, part)*part.batch['response_mask']).sum().div(len(g)).backward()
            observer.before_update(inner)
            path = directory/f'gradient_{update:03d}.npz'
            assert not path.exists()
            actor.actor_optimizer.step()
            actor.actor_optimizer.zero_grad(set_to_none=True)
            observer.after_inner(inner+1)
            with np.load(path) as saved:
                assert saved['optimizer_update'] == update and saved['checkpoint_before'] == update-1
                assert saved['valid_tokens'] == len(g)
                np.testing.assert_allclose(saved['eigenvalues'], expected, rtol=1e-4, atol=1e-8)
                assert saved['own_top128_coverage'] == pytest.approx(1.)
                assert bool(saved['fixed_u10_available']) == (update > 10)
                if update <= 10:
                    assert 'fixed_u10_coverage' not in saved
                else:
                    fraction = np.square(g @ fixed).sum() / np.square(g).sum()
                    assert saved['fixed_u10_coverage'] == pytest.approx(fraction, abs=4e-6)
                    assert observer.metrics['rq2/gradient_fixed_u10_coverage'] == saved['fixed_u10_coverage']
            if update == 10:
                with np.load(directory/'step_010.npz') as early:
                    fixed = early['basis'].copy()
    with ZipFile(package_results(run)) as archive:
        prefix = 'training/training_records/rq2/gradient_'
        assert sorted(n for n in archive.namelist() if n.startswith(prefix)) == [
            prefix+f'{i:03d}.npz' for i in range(1, 107)]


def test_gradient_top128_truncation_and_zero_energy(tmp_path):
    recorder = study_module.MotionRecorder(tmp_path, tmp_path/'legacy', rank=128)
    eigen = np.arange(1, 131, dtype=np.float32)
    second = np.diag(eigen)
    # U10 spans the lowest 128 directions of this step's gradient.
    recorder.reference = torch.eye(len(eigen))[:, :128]
    recorder.record_gradient(11, second, 130)
    with np.load(tmp_path/'gradient_011.npz') as saved:
        np.testing.assert_array_equal(saved['eigenvalues'], eigen[::-1])
        assert saved['own_top128_coverage'] == pytest.approx(eigen[-128:].sum()/eigen.sum())
        assert saved['fixed_u10_coverage'] == pytest.approx(eigen[:128].sum()/eigen.sum())
    recorder.record_gradient(12, np.zeros_like(second), 130)
    with np.load(tmp_path/'gradient_012.npz') as saved:
        assert saved['fixed_u10_available'] and not np.any(saved['eigenvalues'])
        assert 'fixed_u10_coverage' not in saved and 'own_top128_coverage' not in saved
