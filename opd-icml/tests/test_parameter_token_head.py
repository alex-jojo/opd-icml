"""Single-token output work and full gradient equivalence to the dense head."""
import importlib

import numpy as np
import pytest
import torch

from test_parameter_tokens import audit_module, make_actor, make_data, opd, reference_energy


def wide_actor(tied=False, compact=True):
    actor = make_actor(tied, compact)
    model = actor.actor_module
    model.model.embed_tokens = torch.nn.Embedding(64, 4)
    model.lm_head = torch.nn.Linear(4, 64, bias=True)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.config.model_type = 'llama'
    actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    data = make_data()
    # Disjoint, differently ordered supports expose unwanted cross-position work.
    data.batch['opd_top16_ids'] = torch.arange(48).reshape(1, 3, 16).flip(-1).repeat(2, 1, 1)
    return actor, data


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('prefix', [False, True])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_single_position_matches_dense_logits_and_every_parameter_gradient(tied, compact, prefix, position):
    torch.manual_seed(17)
    actor, data = wide_actor(tied, compact=False)
    inputs = data.select_idxs([0]).batch
    teacher = inputs['opd_teacher_top16_logits'][:, position:position+1]
    _, dense, _, _ = opd.forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
    expected = dense[:, position:position+1]
    parameters = list(actor.actor_module.parameters())
    expected_gradients = torch.autograd.grad(opd.reverse_kl(expected, teacher).sum(), parameters)

    actor.config.policy_loss['replication17']['compact_head'] = compact
    trimmed = audit_module.causal_token_prefix(inputs, position) if prefix else inputs
    ids, selected, sampled, mass = opd.forward(actor, trimmed, 1., trimmed['opd_top16_ids'],
                                              diagnostics=False, response_position=position)
    torch.testing.assert_close(selected, expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(ids, inputs['opd_top16_ids'][:, position:position+1])
    assert selected.shape == (1, 1, 16) and sampled.shape == mass.shape == (1, 1)
    actual_gradients = torch.autograd.grad(opd.reverse_kl(selected, teacher).sum(), parameters)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-7)
    # The prompt still contributes through the backbone, including tied weights.
    assert actual_gradients[0][1].norm() > 0
    assert 'forward' not in actor.actor_module.lm_head.__dict__
    assert not hasattr(actor.actor_module.lm_head, '_opd_compact_active')


@pytest.mark.parametrize('prefix', [False, True])
@pytest.mark.parametrize('mode', ['exact_all', 'kl_stratified'])
def test_audit_projects_and_compares_only_one_positions_support(monkeypatch, prefix, mode):
    torch.manual_seed(23)
    actor, data = wide_actor(tied=True, compact=False)
    expected = reference_energy(actor, data)
    with torch.no_grad():
        _, cached, _, _ = opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False)
    data.batch['opd_student_top16_logits'] = cached
    actor.config.policy_loss['replication17']['compact_head'] = True
    fast_head = importlib.import_module(audit_module.__package__ + '.opd_fast_head')
    original_linear, original_kl = fast_head.F.linear, opd.reverse_kl
    projections, comparisons = [], []

    def linear(hidden, weight, bias=None):
        if weight.shape[0] > 4:
            projections.append((tuple(hidden.shape), tuple(weight.shape)))
        return original_linear(hidden, weight, bias)

    def reverse_kl(student, teacher):
        comparisons.append((tuple(student.shape), tuple(teacher.shape)))
        return original_kl(student, teacher)

    monkeypatch.setattr(fast_head.F, 'linear', linear)
    monkeypatch.setattr(opd, 'reverse_kl', reverse_kl)
    result = audit_module.FullParameterTokenAudit(actor, dict(mode=mode, causal_prefix=prefix,
                                                             sample_size=4)).measure(data, 1, 0)
    indices = result['sampling']['index'] if mode == 'kl_stratified' else np.arange(len(expected))
    np.testing.assert_allclose(result['energy'], expected[indices], rtol=2e-5, atol=1e-7)
    assert projections == [((1, 1, 4), (16, 4))] * len(indices)
    assert comparisons == [((1, 1, 16), (1, 1, 16))] * len(indices)
    assert all(p.grad is None for p in actor.actor_module.parameters())


@pytest.mark.parametrize('position', [-1, 3, 0.5])
def test_invalid_target_position_is_rejected_before_forward(position):
    actor, data = wide_actor()
    with pytest.raises(ValueError, match='outside the response'):
        opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False,
                    response_position=position)
    assert 'forward' not in actor.actor_module.lm_head.__dict__
