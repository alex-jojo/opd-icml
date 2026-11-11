"""Grouped diagnostic graphs preserve individual gradients and training state."""
import copy
import importlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_parameter_token_head import wide_actor
from test_parameter_tokens import ROOT, audit_module, opd, optimizer_update


@pytest.mark.parametrize('mode', ['exact_all', 'kl_stratified'])
@pytest.mark.parametrize('group_size', [2, 4])
@pytest.mark.parametrize('prefix', [False, True])
def test_grouped_graphs_match_every_token_gradient_and_preserve_training(monkeypatch, mode, group_size, prefix):
    torch.manual_seed(96)
    actor, data = wide_actor(tied=True)
    optimizer_update(actor, data)
    baseline = copy.deepcopy(actor)
    with torch.no_grad():
        _, cached, _, _ = opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False)
    data.batch['opd_student_top16_logits'] = cached
    options = dict(mode=mode, sample_size=4, causal_prefix=prefix)
    rng = torch.random.get_rng_state().clone()
    snapshots = []
    original_norm = audit_module.GradientEnergy.squared_norm

    def norm(layout):
        snapshots.append([p.grad.clone() for p in layout.layout.params])
        return original_norm(layout)

    monkeypatch.setattr(audit_module.GradientEnergy, 'squared_norm', norm)
    sequential = audit_module.FullParameterTokenAudit(actor, dict(options, graph_group_size=1)).measure(data, 1, 0)
    reference_gradients = snapshots[:]
    snapshots.clear()
    calls, projections = [], []
    hook = actor.actor_module.register_forward_hook(lambda *args: calls.append(1))
    fast_head = importlib.import_module(audit_module.__package__ + '.opd_fast_head')
    original_linear = fast_head.F.linear

    def linear(hidden, weight, bias=None):
        if weight.shape[0] > 4:
            projections.append((tuple(hidden.shape), tuple(weight.shape)))
        return original_linear(hidden, weight, bias)

    monkeypatch.setattr(fast_head.F, 'linear', linear)
    try:
        grouped = audit_module.FullParameterTokenAudit(actor, dict(options, graph_group_size=group_size)).measure(data, 1, 0)
    finally:
        hook.remove()
    for actual, expected in zip(snapshots, reference_gradients):
        for gradient, reference in zip(actual, expected):
            torch.testing.assert_close(gradient, reference, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(grouped['energy'], sequential['energy'], rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(grouped['diagnostic_kl'], sequential['diagnostic_kl'], rtol=3e-5, atol=1e-7)
    np.testing.assert_array_equal(grouped['row'], sequential['row'])
    np.testing.assert_array_equal(grouped['pos'], sequential['pos'])
    expected_calls = sum((sum(grouped['row'] == r)+group_size-1)//group_size for r in np.unique(grouped['row']))
    assert grouped['forward_calls'] == len(calls) == expected_calls < sequential['forward_calls']
    assert projections == [((1, 1, 4), (16, 4))] * grouped['backwards']
    assert grouped['forward_input_tokens'] < sequential['forward_input_tokens']
    assert grouped['backwards'] == sequential['backwards'] == len(snapshots)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in actor.actor_module.parameters())
    assert actor.actor_module.training
    optimizer_update(actor, data)
    optimizer_update(baseline, data)
    for actual, expected in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('size', [0, -1, 1.5, True])
def test_invalid_graph_group_size_is_rejected(size):
    actor, _ = wide_actor()
    with pytest.raises(ValueError, match='positive integer'):
        audit_module.FullParameterTokenAudit(actor, dict(graph_group_size=size))


def test_activation_offload_falls_back_to_single_token():
    actor, data = wide_actor()
    actor.actor_module.model._verl_activation_offloading = True
    result = audit_module.FullParameterTokenAudit(actor, {}).measure(data, 1, 0)
    assert result['graph_group_size'] == 1
    assert result['forward_calls'] == result['backwards'] == 5
    assert 'activation offloading' in result['graph_reuse_fallback']


def test_group_backward_failure_still_clears_gradients_and_restores_modes(monkeypatch):
    actor, data = wide_actor()
    rng = torch.random.get_rng_state().clone()
    calls = []
    original_norm = audit_module.GradientEnergy.squared_norm

    def fail_on_second(layout):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('injected diagnostic failure')
        return original_norm(layout)

    monkeypatch.setattr(audit_module.GradientEnergy, 'squared_norm', fail_on_second)
    with pytest.raises(RuntimeError, match='injected'):
        audit_module.FullParameterTokenAudit(actor, {}).measure(data, 1, 0)
    assert all(p.grad is None for p in actor.actor_module.parameters())
    assert actor.actor_module.training
    assert torch.equal(rng, torch.random.get_rng_state())
    assert 'forward' not in actor.actor_module.lm_head.__dict__
    optimizer_update(actor, data)


def test_group_size_cli_override_preserves_sampling_and_training_protocol():
    command = [sys.executable, '-B', str(ROOT/'train.py'), '--scale', 'deepseek-coder-1.3b',
               '--task', 'code', '--experiment', 'paired-full', '--dry-run']
    base = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    selected = json.loads(subprocess.run(command+['--parameter-token-group-size', '1'],
                                        check=True, capture_output=True, text=True).stdout)
    assert selected['parameter_token_gradients'] == dict(base['parameter_token_gradients'], graph_group_size=1)
    assert 'parameter_tokens.graph_group_size=1' in selected['command']
    for key in ('rollout_batch', 'rounds_cap', 'updates_per_round', 'learning_rate'):
        assert selected[key] == base[key]
    bad = subprocess.run(command+['--parameter-token-group-size', '0'], capture_output=True, text=True)
    assert bad.returncode != 0 and 'must be positive' in bad.stderr
