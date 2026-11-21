"""Cost reductions retain parameter coverage, numerical accuracy and CPU results."""
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_parameter_tokens import audit_module, study_module, make_actor, make_data
from test_rq3_records import batch128


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_energy_chunks_match_small_chunks_without_mutating_gradient(dtype):
    torch.manual_seed(992)
    model = torch.nn.Linear(1048581, 1, bias=False, dtype=dtype)
    model.weight.grad = torch.randn_like(model.weight) * torch.logspace(-10, 10, model.weight.numel(), dtype=dtype)
    original = model.weight.grad.clone()
    slices = [(0, 3, 524299, ['all']), (0, 524310, 524260, ['all'])]
    model.weight._fqns = ['first', 'second']
    model.weight._shard_param_infos = [SimpleNamespace(in_shard=True, offset_in_shard=start,
                                                      numel_in_shard=count) for _,start,count,_ in slices]
    small = audit_module.GradientEnergy(model, chunk_elements=262144)
    large = audit_module.GradientEnergy(model)
    expected = small.squared_norm()
    actual = large.squared_norm()
    assert actual == pytest.approx(expected, rel=2e-14)
    assert torch.equal(model.weight.grad, original)
    assert large.chunk_elements == 4194304


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_energy_chunk_size(value):
    with pytest.raises(ValueError, match='positive integer'):
        audit_module.GradientEnergy(make_actor().actor_module, value)


def test_full_gram_cache_uses_one_gather_and_preserves_all_groups(monkeypatch):
    actor = make_actor()
    layout = study_module.GradientStudy.__init__.__globals__['ParameterLayout'](actor.actor_module)
    implementation = study_module.GradientStudy.__init__.__globals__
    torch.manual_seed(991)
    vectors = [[torch.randn(p.numel()) for p in layout.params] for _ in range(4)]
    original = layout.dot_many
    calls = []
    def dot_many(pairs):
        calls.append(len(pairs))
        return original(pairs)
    monkeypatch.setattr(layout, 'dot_many', dot_many)
    monkeypatch.setitem(implementation, '_gradient_cache_slots', lambda *_: 0)
    expected, old_count = implementation['exact_gram'](lambda i: vectors[i], layout, 4)
    assert old_count == 10 and calls == [1]*10
    calls.clear()
    monkeypatch.setitem(implementation, '_gradient_cache_slots', lambda *_: 3)
    actual, count = implementation['exact_gram'](lambda i: vectors[i], layout, 4)
    assert count == 4 and calls == [10]
    for group in expected:
        np.testing.assert_array_equal(actual[group], expected[group])


def test_cpu_geometry_overlap_preserves_results_and_joins_worker(tmp_path, monkeypatch):
    torch.manual_seed(993)
    actor = make_actor()
    actor.config.ppo_micro_batch_size_per_gpu = 32
    study = study_module.GradientStudy(actor, dict(batches=4, overlap_cpu_geometry=False), tmp_path)
    data, basis = batch128(), torch.eye(4)[:, :2]
    expected = study.cross_batch(data, basis, 5)
    implementation = study.cross_batch.__func__.__globals__
    original = implementation['scope_geometry']
    calls = []
    def record(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setitem(implementation, 'scope_geometry', record)
    study.cfg['overlap_cpu_geometry'] = True
    actual = study.cross_batch(data, basis, 5)
    assert actual == expected
    assert len(calls) == 4 and all(t != threading.get_ident() for t in calls)
    assert not any(t.name.startswith('opd-cpu-geometry') for t in threading.enumerate())
    def fail(*_args, **_kwargs):
        raise ValueError('geometry failure is not dropped')
    monkeypatch.setitem(implementation, 'scope_geometry', fail)
    with pytest.raises(Exception, match='geometry failure is not dropped'):
        study.cross_batch(data, basis, 5)
    assert not any(t.name.startswith('opd-cpu-geometry') for t in threading.enumerate())


def test_stage_times_are_exported_without_changing_audit_results():
    actor, data = make_actor(), make_data()
    result = audit_module.FullParameterTokenAudit(actor, dict(mode='exact_all')).measure(data, 1, 0)
    assert result['norm_chunk_elements'] == 4194304
    timing = result['stage_timing']
    assert timing['clock'] == 'wall'
    assert set(timing['stage_seconds']) == {'forward', 'backward', 'parameter_energy'}
    assert all(value >= 0 for value in timing['stage_seconds'].values())
