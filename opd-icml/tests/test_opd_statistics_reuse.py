"""Statistics reuse preserves stable ties, report values and FP64 entropy."""
import copy
import importlib

import numpy as np
import pytest
import torch

from test_parameter_tokens import audit_module, make_actor, make_data, opd, study_module


def test_group_grams_visit_each_row_once_and_preserve_spectra(monkeypatch):
    scopes = importlib.import_module(audit_module.__package__ + '.opd_group_scopes')
    rng = np.random.default_rng(24)
    gradients = rng.standard_normal((150, 7)).astype(np.float32)
    kl = rng.integers(0, 18, len(gradients))
    basis = torch.eye(7)[:, :2]
    visits = np.zeros(len(kl), dtype=int)
    original = scopes.group_gram

    def observed(parts, mask, *args):
        visits[mask] += 1
        return original(parts, mask, *args)

    monkeypatch.setattr(scopes, 'group_gram', observed)
    _, arrays = scopes.compute(np.split(gradients, [59, 110]), kl, basis, 'cpu')
    np.testing.assert_array_equal(visits, np.ones_like(visits))
    groups = scopes.partitions(kl)
    k = min(map(len, groups.values()))
    for name, rows in groups.items():
        picks = rows[np.linspace(0, len(rows)-1, k).round().astype(int)]
        for kind, selected in [('all', rows), ('matched', picks)]:
            g = gradients[selected].astype(np.float64)
            for scope, x in [('full', g), ('u', g[:, :2]), ('outside', g[:, 2:])]:
                expected = np.linalg.svd(x, compute_uv=False)**2 / len(g)
                np.testing.assert_allclose(arrays[f'{name}_{kind}_{scope}_eigenvalues'], expected, rtol=2e-5)

audit = importlib.import_module(audit_module.__package__ + '.opd_token_audit')
sampling = importlib.import_module(audit_module.__package__ + '.opd_parameter_sampling')
SCORES = ('kl', 'tv', 'entropy', 'tip', 'logratio_variance', 'ier', 'alignment', 'priority')
THRESHOLDS = dict(kl_p90=.3, kl_median=.2, entropy_median=1.)


def fields(count, valid_ier=True):
    rng = np.random.default_rng(81)
    result = {key: rng.integers(0, 4, count).astype(float)/10 for key in SCORES}
    result.update(row=np.arange(count)//7, pos=np.arange(count)%7,
                  boundary=np.arange(count)%7 == 0,
                  relative_position=(np.arange(count)%7)/6,
                  ier_valid=(np.arange(count)%3 != 0) if valid_ier else np.zeros(count, bool),
                  raw_gradient_energy=rng.uniform(1., 3., count),
                  projected_gradient_energy=rng.uniform(.1, .9, count))
    return result


@pytest.mark.parametrize('count', [0, 1, 2, 5, 11, 100])
@pytest.mark.parametrize('valid_ier', [False, True])
def test_top_percent_masks_keep_stable_ties_and_sort_each_valid_score_once(monkeypatch, count, valid_ier):
    f = fields(count, valid_ier)
    calls = []
    original_sort = np.argsort

    def argsort(*args, **kwargs):
        calls.append(kwargs.get('kind'))
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(np, 'argsort', argsort)
    masks = audit.groups(f, THRESHOLDS)
    expected_sorts = 7+int(f['ier_valid'].any()) if count else 0
    assert calls == ['stable']*expected_sorts
    for score in SCORES:
        # Python's stable sort is independent of NumPy's ranking implementation.
        eligible = [i for i in range(count) if score != 'ier' or f['ier_valid'][i]]
        ranking = sorted(eligible, key=lambda i: -f[score][i])
        for fraction in (.01, .05, .1, .2):
            chosen = set(ranking[:max(1, int(np.ceil(len(ranking)*fraction)))])
            expected = np.array([i in chosen for i in range(count)], dtype=bool)
            np.testing.assert_array_equal(masks[f'{score}_top{int(fraction*100)}'], expected)


def test_kl_partitions_keep_ascending_tie_order_separate_from_top_percent_groups():
    f = fields(11)
    f['kl'][:] = .2
    grouping = audit.prepare_groups(f, THRESHOLDS)
    np.testing.assert_array_equal(np.flatnonzero(grouping['masks']['kl_top10']), [0, 1])
    np.testing.assert_array_equal(grouping['kl_parts']['low'], [0, 1])
    np.testing.assert_array_equal(grouping['kl_parts']['middle'], np.arange(2, 9))
    np.testing.assert_array_equal(grouping['kl_parts']['high'], [9, 10])


@pytest.mark.parametrize('count,budget', [(1, 256), (5, 256), (100, 16)])
def test_shared_grouping_preserves_full_and_sampled_report_and_array_values(monkeypatch, count, budget):
    f = fields(count)
    z = np.random.default_rng(82).normal(size=(count, 3)).astype(np.float32)
    plan = sampling.stratified_plan(f['kl'], budget)
    chosen = plan['index']
    measurement = dict(row=f['row'][chosen], pos=f['pos'][chosen], sampling=plan,
                       energy=np.linspace(.1, 1., len(chosen)), diagnostic_kl=f['kl'][chosen],
                       backwards=len(chosen), seconds=1.)
    expected_report, expected_arrays = audit.training_summary(f, z, THRESHOLDS)
    expected_metadata, expected_paired = sampling.add_sampled_report(
        expected_report, f, measurement, THRESHOLDS)
    before = copy.deepcopy(f)
    original_prepare = audit.prepare_groups
    preparations = []

    def prepare(*args):
        result = original_prepare(*args)
        preparations.append(result)
        return result

    monkeypatch.setattr(audit, 'prepare_groups', prepare)
    grouping = audit.prepare_groups(f, THRESHOLDS)
    report, arrays = audit.training_summary(f, z, THRESHOLDS, grouping=grouping)
    metadata, paired = sampling.add_sampled_report(report, f, measurement, THRESHOLDS, grouping=grouping)
    assert len(preparations) == 1
    np.testing.assert_equal(report, expected_report)
    np.testing.assert_equal(arrays, expected_arrays)
    np.testing.assert_equal(metadata, expected_metadata)
    np.testing.assert_equal(paired, expected_paired)
    np.testing.assert_equal(f, before)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_training_hook_reuses_teacher_log_probability_with_identical_entropy(monkeypatch, dtype):
    torch.manual_seed(83)
    actor, data = make_actor(), make_data()
    data.batch['opd_teacher_top16_logits'] = (data.batch['opd_teacher_top16_logits']*10).to(dtype)
    inputs = data.batch
    mask = inputs['response_mask'].bool()
    with torch.no_grad():
        _, logits, _, _ = opd.forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
    teacher = inputs['opd_teacher_top16_logits'][mask]
    expected = audit.output_scores(logits[mask], teacher)
    log_q = teacher.detach().double().log_softmax(-1)
    expected_entropy = -(log_q.exp()*log_q).sum(-1)
    assert 'teacher_entropy' not in expected  # existing callers keep their schema
    original_log_softmax = torch.Tensor.log_softmax
    calls = []

    def log_softmax(tensor, *args, **kwargs):
        calls.append(tensor.dtype)
        return original_log_softmax(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'log_softmax', log_softmax)
    runtime = object.__new__(study_module.StudyReplication)
    runtime.actor, runtime.measure, runtime.parts = actor, True, []
    with runtime.watch_training(inputs, 0, 0), torch.no_grad():
        opd.forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
    assert calls == [torch.float64, torch.float64]  # student and teacher, once each
    part = runtime.parts[0]
    for key, value in expected.items():
        np.testing.assert_array_equal(part[key], value.numpy())
    np.testing.assert_array_equal(part['teacher_entropy'], expected_entropy.numpy())
