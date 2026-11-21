"""Partial exact-gradient caching preserves statistics, state and memory bounds."""
import copy
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_parameter_tokens import audit_module, make_actor, make_data, optimizer_update

study = importlib.import_module(audit_module.__package__ + '.opd_gradient_study')


@pytest.mark.parametrize('slots,expected_calls', [(0, 7), (1, 7), (2, 5), (3, 4)])
def test_partial_cache_preserves_real_gradients_and_next_update(monkeypatch, slots, expected_calls):
    torch.manual_seed(314)
    actor, data = make_actor(tied=True), make_data()
    optimizer_update(actor, data)
    baseline = copy.deepcopy(actor)
    state = copy.deepcopy(actor.actor_optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    batches = []
    for i in range(4):
        batch = copy.deepcopy(data)
        batch.batch['opd_teacher_top16_logits'] = torch.roll(batch.batch['opd_teacher_top16_logits'], i, -1)
        batches.append(batch)
    layout = study.ParameterLayout(actor.actor_module)
    basis = torch.eye(4)[:, :2]
    requested = []
    def oracle(i):
        requested.append(i)
        return study.gradient(actor, batches[i], basis)[0]
    monkeypatch.setattr(study, '_gradient_cache_slots', lambda *_: 0)
    expected, count = study.exact_gram(oracle, layout, 4)
    assert count == 7
    requested.clear()
    monkeypatch.setattr(study, '_gradient_cache_slots', lambda *_: slots)
    actual, count = study.exact_gram(oracle, layout, 4)
    assert count == len(requested) == expected_calls
    assert list(dict.fromkeys(requested)) == [0, 1, 2, 3]
    assert actual.keys() == expected.keys()
    for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in actor.actor_module.parameters())
    for index, values in state['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, actor.actor_optimizer.state_dict()['state'][index][key], rtol=0, atol=0)
    optimizer_update(actor, data)
    optimizer_update(baseline, data)
    for actual, expected in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def memory_context(monkeypatch, parameters, free_gib, cap='64', world=1):
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=int(free_gib * 1024**3))))
    monkeypatch.setenv('OPD_GRADIENT_CACHE_MAX_GIB', cap)
    monkeypatch.setattr(study.Path, 'read_text', lambda *_: (_ for _ in ()).throw(OSError('no cgroup')))
    monkeypatch.setattr(study.dp, 'size', lambda: world)
    monkeypatch.setattr(study.dp, 'gather', lambda value: [value] * world)
    return SimpleNamespace(params=[SimpleNamespace(numel=lambda: parameters)])


@pytest.mark.parametrize('parameters,free_gib,cap,world,slots', [
    (4_000_000_000, 512, '64', 1, 3),
    (4_000_000_000, 512, '128', 1, 3),
    (4_000_000_000, 128, '64', 1, 0),
    (1_700_000_000, 160, '64', 1, 3),
    (1_000_000_000, 512, '64', 4, 3),
    (10, 512, '0', 1, 0),
    (10, 512, 'bad', 1, 0),
    (10, 512, 'nan', 1, 0),
])
def test_capacity_uses_available_memory_and_cap(monkeypatch, parameters, free_gib, cap, world, slots):
    layout = memory_context(monkeypatch, parameters, free_gib, cap, world)
    assert study._gradient_cache_slots(layout, 4) == slots


def test_rank_agreement_and_container_limit(monkeypatch):
    layout = memory_context(monkeypatch, 4_000_000_000, 512)
    monkeypatch.setattr(study.dp, 'gather', lambda local: [local, 1])
    assert study._gradient_cache_slots(layout, 4) == 1
    monkeypatch.setattr(study.dp, 'gather', lambda local: [local, 0])
    assert study._gradient_cache_slots(layout, 4) == 0
    monkeypatch.setattr(study.dp, 'gather', lambda local: [local])
    def read(path):
        if str(path).endswith('memory.max'): return str(128 * 1024**3)
        if str(path).endswith('memory.current'): return '0'
        raise OSError('missing')
    monkeypatch.setattr(study.Path, 'read_text', read)
    assert study._gradient_cache_slots(layout, 4) == 0


def test_unavailable_memory_information_disables_cache(monkeypatch):
    layout = memory_context(monkeypatch, 10, 512)
    monkeypatch.setitem(sys.modules, 'psutil', None)
    assert study._gradient_cache_slots(layout, 4) == 0
