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


def test_tiny_increment_projects_the_change_before_roundoff(tmp_path):
    torch.manual_seed(22)
    base = torch.randn(8, 64)
    current = base.clone()
    current[:, 3] += 2e-6
    basis = torch.linalg.qr(torch.randn(64, 8)).Q
    recorder = study_module.MotionRecorder(tmp_path, tmp_path/'legacy', 8)
    recorder.incremental_change(10, base, base, basis)
    result = recorder.incremental_change(11, base, current, basis)
    change = torch.nn.functional.normalize(current, dim=-1)-torch.nn.functional.normalize(base, dim=-1)
    expected = float((change.double()@basis.double()).square().sum()/change.double().square().sum())
    assert result['since_step10_fixed_u10_coverage'] == pytest.approx(expected, abs=2e-7)


def test_resume_before_step10_ignores_future_basis_asset(tmp_path):
    import copy
    torch.manual_seed(83)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path)
    observer = study_module.StudyReplication(actor)
    for step in range(1, 12):
        observer.begin(data, None, step)
        optimizer_update(actor, data)
        observer.after_inner(1)
        if step == 5:
            checkpoint = copy.deepcopy(actor.actor_module.state_dict())
            optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
            expected_basis = observer.u.clone()
    actor.actor_module.load_state_dict(checkpoint)
    actor.actor_optimizer.load_state_dict(optimizer)
    resumed = study_module.StudyReplication(actor)
    resumed.begin(data, None, 6)
    torch.testing.assert_close(resumed.u, expected_basis, atol=0, rtol=0)
    # A restart before any PCA observation keeps the initial basis as well.
    early = study_module.StudyReplication(actor)
    early.begin(data, None, 3)
    torch.testing.assert_close(early.u, torch.eye(4)[:, :2], atol=0, rtol=0)


def test_external_early_evaluation_checks_real_target_without_shortcut(tmp_path):
    run, external = tmp_path/'run', tmp_path/'weights'
    run.mkdir()
    cleanup.register_external_models(dict(output=str(run), model_artifacts=str(external)))
    weights = write(external, 'checkpoints/early_stop/actor/model.pt')
    math_results(run)
    write(run, 'rq4_training_time.json', b'{"shared_early_stop":true}')
    write(external, 'checkpoints/early_stop/selection.json', b'{"step":40,"uses_full_endpoint":false}')
    write(run, 'training/training_records/rq4/early_stop_summary.json',
          b'{"source_strategy":"full","rollout_rounds":30,"uses_full_endpoint":false}')
    with pytest.raises(ValueError, match='does not match'):
        cleanup.cleanup_completed_models(run, 'math', external)
    assert weights.exists()


def test_failed_atomic_moment_preserves_last_complete_file(tmp_path, monkeypatch):
    path = tmp_path/'moment.npz'
    matrix = np.eye(4, dtype=np.float32)
    storage.write_moment(path, dict(second=matrix))
    previous = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('simulated full filesystem')
    monkeypatch.setattr(np.lib.format, 'write_array', fail)
    with pytest.raises(OSError, match='simulated'):
        storage.write_moment(path, dict(second=matrix*2))
    assert path.read_bytes() == previous and not path.with_suffix('.tmp').exists()


def test_resume_requires_all_four_shards_optimizer_and_data(tmp_path):
    from resume_state import validate_resume_checkpoint
    root = tmp_path/'checkpoints'
    with pytest.raises(ValueError, match='refusing to restart'):
        validate_resume_checkpoint(root, 4, 2)
    write(root, 'latest_checkpointed_iteration.txt', b'20')
    checkpoint = root/'global_step_20'
    write(checkpoint, 'actor/fsdp_config.json', b'{"world_size":4}')
    write(checkpoint, 'opd_optimizer_updates.json',
          b'{"rollout_round":20,"optimizer_updates_done":40,"budget_unit":"rollout_rounds"}')
    write(checkpoint, 'data.pt')
    for rank in range(4):
        for kind in ('model', 'optim', 'extra_state'):
            write(checkpoint, f'actor/{kind}_world_size_4_rank_{rank}.pt')
    assert validate_resume_checkpoint(root, 4, 2) == checkpoint
    with pytest.raises(ValueError, match='same GPU count'):
        validate_resume_checkpoint(root, 1, 2)
    with pytest.raises(ValueError, match='counters'):
        validate_resume_checkpoint(root, 4, 1)
    missing = checkpoint/'actor/optim_world_size_4_rank_3.pt'
    missing.unlink()
    with pytest.raises(ValueError, match='optim_world_size_4_rank_3'):
        validate_resume_checkpoint(root, 4, 2)
    write(checkpoint, 'actor/optim_world_size_4_rank_3.pt')
    write(checkpoint, 'final_model_ready.json', b'{"resumable":false}')
    with pytest.raises(ValueError, match='model-only'):
        validate_resume_checkpoint(root, 4, 2)


def test_old_external_run_can_export_without_claiming_or_deleting_weights(tmp_path):
    run, external = tmp_path/'run', tmp_path/'old_weights'
    weights = write(external, 'checkpoints/old.pt')
    math_results(run)
    plan = dict(output=str(run), model_artifacts=str(external), task='math')
    cleanup.register_external_models(plan, allow_unowned_resume=True)
    assert not (external/'.opd-model-owner.json').exists()
    cleanup.finish_experiment(SimpleNamespace(stage='eval', keep_model_files=False), plan)
    assert weights.exists()
    report = json.loads((run/'model_cleanup.json').read_text(encoding='utf-8'))
    assert report['status'] == 'skipped' and report['removed_directories'] == []


def test_fused_dots_preserve_legacy_chunk_order(monkeypatch):
    torch.manual_seed(92)
    layout = grad.ParameterLayout.__new__(grad.ParameterLayout)
    layout.replicated_adapter = None
    layout.slices = [(0, 3, 524311, ['all', 'layer_0']), (1, 1, 39, ['all', 'head'])]
    vectors = [[torch.randn(524320), torch.randn(43)] for _ in range(3)]
    pairs = [(vectors[0], vectors[0]), (vectors[1], vectors[0]), (vectors[2], vectors[0])]
    expected = []
    for a, b in pairs:
        groups = {}
        for i, start, n, labels in layout.slices:
            total = 0.
            for offset in range(start, start+n, 262144):
                end = min(offset+262144, start+n)
                total += float(torch.dot(a[i][offset:end].double(), b[i][offset:end].double()))
            for group in labels:
                groups[group] = groups.get(group, 0.) + total
        expected.append(groups)
    calls = []
    monkeypatch.setattr(grad.dp, 'gather', lambda value: calls.append(value) or [value])
    assert layout.dot_many(pairs) == expected
    assert len(calls) == 1
