import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_parameter_tokens import ROOT, make_actor, make_data, optimizer_update, study_module
from test_rq2_motion import configure
from test_model_cleanup import cleanup, math_results, write

storage = importlib.import_module('parameter_tokens_cpu.opd_diagnostic_storage')
grad = importlib.import_module('parameter_tokens_cpu.opd_gradient_study')


def test_lossless_archive_dense_fallback_and_legacy(tmp_path):
    rng = np.random.default_rng(71)
    g = rng.normal(size=(90, 17)).astype(np.float32)
    matrix = g.T @ g
    path = tmp_path/'moment.npz'
    storage.write_moment(path, dict(second=matrix, mean=g.mean(0), count=len(g)))
    with np.load(path) as packed:
        assert 'second' not in packed
        assert packed['second_upper'].nbytes == 17*18//2*4
    decoded = storage.load_moment(path)
    np.testing.assert_array_equal(decoded['second'], matrix)
    np.testing.assert_array_equal(decoded['mean'], g.mean(0))
    matrix[0, 1] += 1
    storage.write_moment(path, dict(second=matrix))
    with np.load(path) as dense:
        assert 'second' in dense
    np.testing.assert_array_equal(storage.load_moment(path)['second'], matrix)
    np.savez(path, second=matrix)
    np.testing.assert_array_equal(storage.load_moment(path)['second'], matrix)


def test_incremental_geometry_matches_direct_states_and_resume(tmp_path):
    torch.manual_seed(72)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path)
    observer = study_module.StudyReplication(actor)
    reference_hidden = basis = previous = None
    for update in range(1, 15):
        if update == 13:
            observer = study_module.StudyReplication(actor)
        observer.begin(data, None, update)
        if update == 1:
            previous = torch.nn.functional.normalize(study_module.panel_hidden(actor, data), dim=-1)
        optimizer_update(actor, data)
        observer.after_inner(1)
        current = torch.nn.functional.normalize(study_module.panel_hidden(actor, data), dim=-1)
        with np.load(tmp_path/f'training_records/rq2/step_{update:03d}.npz') as saved:
            if update == 10:
                reference_hidden = current.clone()
                basis = torch.from_numpy(saved['basis'].copy())
            assert bool(saved['previous_step_available']) == (update != 13)
            if update != 13:
                energy = float((current-previous).double().square().sum())
                assert saved['previous_step_energy'] == pytest.approx(energy, rel=1e-5, abs=1e-12)
            if update > 10:
                delta = current-reference_hidden
                energy = float(delta.double().square().sum())
                coverage = float((delta@basis).double().square().sum())/energy
                assert saved['since_step10_energy'] == pytest.approx(energy, rel=1e-5)
                assert saved['since_step10_fixed_u10_coverage'] == pytest.approx(coverage, abs=2e-4)
        previous = current
    assert not list((tmp_path/'replication/gradient_moments').glob('token_sample*'))


def test_parameter_update_summary_uses_true_coordinate_counts():
    actor = make_actor(tied=True)
    layout = grad.ParameterLayout(actor.actor_module)
    before = layout.capture(gradient=False)
    with torch.no_grad():
        for p in layout.params:
            p.view(-1)[0] += .25
            if p.numel() > 1:
                p.view(-1)[1] -= .5
    delta = layout.delta(before)
    norms = layout.dot(delta, delta)
    report = layout.update_summary(before, delta, norms)
    vector = torch.cat(delta)
    all_stats = report['all']
    assert all_stats['count'] == vector.numel()
    assert all_stats['changed'] == int(torch.count_nonzero(vector))
    assert all_stats['update_norm'] == pytest.approx(float(vector.double().norm()))
    assert all_stats['positive'] == len(layout.params)
    assert all_stats['negative'] == len(layout.params)
    assert 'shared_embedding_head' in report
    assert 'head' not in report and 'embedding' not in report


def test_external_owned_cleanup_preserves_statistics_and_foreign_files(tmp_path):
    run, external = tmp_path/'run', tmp_path/'weights'/'run'
    run.mkdir()
    plan = dict(output=str(run), model_artifacts=str(external))
    cleanup.register_external_models(plan)
    checkpoint = write(external, 'checkpoints/global_step_106/actor/model.pt')
    merged = write(external, 'merged/model.safetensors')
    statistics = write(external, 'validation128/old.json')
    math_results(run)
    cleanup.cleanup_completed_models(run, 'math', external)
    assert not checkpoint.exists() and not merged.exists()
    assert statistics.exists() and (run/'eval/final_mean8.jsonl').exists()
    assert json.loads((run/'model_cleanup.json').read_text(encoding='utf-8'))['status'] == 'complete'
    other = tmp_path/'foreign'
    weights = write(other, 'checkpoints/model.pt')
    with pytest.raises(ValueError, match='ownership'):
        cleanup.cleanup_completed_models(run, 'math', other)
    assert weights.exists()
    with pytest.raises(ValueError, match='unowned'):
        cleanup.register_external_models(dict(output=str(run), model_artifacts=str(other)))


def test_external_cleanup_waits_for_evaluation(tmp_path):
    run, external = tmp_path/'run', tmp_path/'weights'
    run.mkdir()
    cleanup.register_external_models(dict(output=str(run), model_artifacts=str(external)))
    weights = write(external, 'checkpoints/model.pt')
    with pytest.raises(ValueError, match='missing evaluation'):
        cleanup.cleanup_completed_models(run, 'math', external)
    assert weights.exists()


def test_validation_remains_in_portable_run_with_external_checkpoints(tmp_path, monkeypatch):
    import importlib.util
    from package_results import package_results
    from zipfile import ZipFile
    spec = importlib.util.spec_from_file_location('storage_fixed_validation', ROOT/'runtime/verl/trainer/ppo/opd_fixed_validation.py')
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    run = tmp_path/'run'
    class Config(dict):
        __getattr__ = dict.__getitem__
    config = SimpleNamespace(trainer=Config(default_local_dir=str(tmp_path/'external/checkpoints'),
                                            rollout_data_dir=str(run/'training/training_records')))
    monkeypatch.setenv('OPD_RUN', str(run/'training'))
    output = fixed.validation_directory(config)
    assert output == run/'training/validation128'
    write(output, 'checkpoint_010.json', b'{"kl":0.1}')
    with ZipFile(package_results(run)) as archive:
        assert 'run/training/validation128/checkpoint_010.json' in archive.namelist()
        assert 'READ_MOMENTS.py' in archive.namelist()
    monkeypatch.delenv('OPD_RUN')
    assert fixed.validation_directory(config) == output
