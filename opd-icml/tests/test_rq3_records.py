"""Check the portable RQ3 records against independent per-response and per-subset gradients."""
import hashlib
import importlib.util
import json
import pickle
import sys
from importlib import import_module
from itertools import combinations
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from test_parameter_tokens import ROOT, make_actor, make_data, objective, study_module
from test_rq2_motion import configure

sys.path.insert(0, str(ROOT/'scripts'))
from package_results import package_results


def batch128():
    data = make_data().select_idxs([0, 1]*64)
    tokens = torch.randint(16, (128, 6))
    data.batch.update(input_ids=tokens, responses=tokens[:, -3:], opd_teacher_top16_logits=torch.randn(128, 3, 16))
    return data


def response_kl(actor, batch):
    with torch.no_grad():
        kl = objective(actor, batch)
        mask = batch.batch['response_mask']
        return ((kl*mask).sum(1)/mask.sum(1)).numpy()


def subset_reference(actor, batch, rows):
    data = batch.select_idxs(rows)
    mask = data.batch['response_mask'].bool()
    hidden = []
    with actor.actor_module.model.norm.register_forward_hook(lambda m, a, h: hidden.append(h)):
        loss = objective(actor, data)[mask].mean()
    dh, *param = torch.autograd.grad(loss, (hidden[0], *actor.actor_module.parameters()))
    g = dh[:, -mask.shape[1]-1:-1][mask].double()*mask.sum()
    axes = torch.linalg.svd(g-g.mean(0), full_matrices=False).Vh[:4]
    return torch.cat([p.flatten() for p in param]).double(), g.mean(0), axes


@pytest.mark.parametrize('interval', [1, 5])
@pytest.mark.parametrize('reuse', [1, 2], ids=['full', 'reuse'])
def test_compact_rq3_on_106_update_schedule(tmp_path, monkeypatch, reuse, interval):
    torch.manual_seed(27)
    run = tmp_path/'training'
    cfg = json.loads((ROOT/'configs/1.7b/paired-full.yaml').read_text())['actor_rollout_ref']['actor']['policy_loss']['gradient_study']
    actor = make_actor()
    # More than four hidden dimensions distinguish centered top-4 overlap from full-space overlap.
    model = actor.actor_module
    model.model.embed_tokens = torch.nn.Embedding(16, 8)
    model.model.layers[0] = torch.nn.Linear(8, 8)
    model.model.norm = torch.nn.LayerNorm(8)
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    actor.config.ppo_micro_batch_size_per_gpu = 32
    configure(actor, run, rounds=106//reuse, reuse=reuse)
    actor.config.policy_loss.update(gradient_study={**cfg, 'subset_alignment_interval': interval, 'parameter_tokens': {}}, jump_reuse_updates=reuse)
    replication = study_module.StudyReplication(actor)
    study = replication.study
    subset_checks = []
    cross_batch = study.cross_batch
    def measured(train, basis, checkpoint, *, detailed=False):
        subset_checks.append((checkpoint, detailed))
        return cross_batch(train, basis, checkpoint, detailed=detailed)
    monkeypatch.setattr(study, 'cross_batch', measured)
    train = batch128()
    directory = run/'training_records/rq3'
    spec = importlib.util.spec_from_file_location('rq3_fixed_validation', ROOT/'runtime/verl/trainer/ppo/opd_fixed_validation.py')
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    source = list(range(2000, 2128))
    trainer = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        actor=actor.config)), train_dataset=list(range(256)),
        opd_validation_source_indices=source, opd_validation_output=run/'validation128')
    trainer.opd_validation_output.mkdir(parents=True)
    readouts = []

    def readout(trainer, checkpoint):
        readouts.append(checkpoint)
        return batch128()

    monkeypatch.setattr(fixed, 'readout', readout)
    monkeypatch.setattr(fixed, 'generate', lambda trainer, rows: batch128().select_idxs(list(range(len(rows)))))
    for update in range(1, 107):
        checkpoint = update-1
        trainer.global_steps, inner = divmod(checkpoint, reuse)
        trainer.global_steps += 1
        payload_bytes = None
        if inner == 0:
            fixed.attach(trainer, train)
            payload_bytes = train.meta_info.pop('gradient_study_payload', None)
            replication.begin(train, None, trainer.global_steps)
        if payload_bytes:
            payload = pickle.loads(payload_bytes)
            validation = payload['validation']
            before = response_kl(actor, validation)
            replication.start_study(payload)
        actor.actor_optimizer.zero_grad(set_to_none=True)
        replication.before_inner(inner)
        assert all(p.grad is None for p in model.parameters())
        alignment_path = directory/f'subset_alignment_{update:03d}.json'
        assert not alignment_path.exists()
        partition = np.random.default_rng(cfg['seed']+checkpoint).permutation(128).reshape(4, 32).tolist()
        check_values = checkpoint in (0, 1, 10, 30, 60, 100, 105)
        if check_values:
            references = [subset_reference(actor, train, rows) for rows in partition]
        objective(actor, train)[train.batch['response_mask'].bool()].mean().backward()
        if payload_bytes:
            study.before_update()
        actor.actor_optimizer.step()
        calls_before_readout = study.calls
        replication.after_inner(inner+1)
        assert study.calls == calls_before_readout
        expected_alignment = update == 1 or update == 106 or update % interval == 0 or checkpoint in cfg['geometry_steps']
        assert alignment_path.exists() == expected_alignment
        cross = json.loads(alignment_path.read_text()) if expected_alignment else None
        if cross is not None:
            assert (cross['checkpoint_before'], cross['optimizer_update']) == (checkpoint, update)
            assert (cross['rollout_round'], cross['inner']) == (trainer.global_steps, inner+1)
            assert cross['subset_train_rows'] == partition and len(cross['pairs']) == 6
        if check_values and cross is not None:
            for pair, (a, b) in zip(cross['pairs'], combinations(range(4), 2), strict=True):
                assert (pair['subset_a'], pair['subset_b']) == (a, b)
                pa, ma, va = references[a]
                pb, mb, vb = references[b]
                assert pair['parameter_gradient_cosine'] == pytest.approx(torch.cosine_similarity(pa, pb, dim=0).item(), abs=2e-6)
                assert pair['mean_hidden_gradient_cosine'] == pytest.approx(torch.cosine_similarity(ma, mb, dim=0).item(), abs=2e-6)
                assert pair['centered_covariance_top4_overlap'] == pytest.approx((va@vb.T).square().sum().item()/4, abs=2e-5)
            for key, value in cross['means'].items():
                assert cross['valid_pairs'][key] == 6 and value == pytest.approx(np.mean([p[key] for p in cross['pairs']]))
        if not payload_bytes:
            continue
        after = response_kl(actor, validation)
        record = json.loads((directory/f'checkpoint_{checkpoint:03d}.json').read_text())
        assert (record['checkpoint_before'], record['optimizer_update']) == (checkpoint, update)
        rows = record['responses']
        assert [r['source_index'] for r in rows] == source
        assert [r['response_row'] for r in rows] == list(range(128))
        for i, row in enumerate(rows):
            ids = validation.batch['responses'][i][validation.batch['response_mask'][i].bool()]
            assert row['response_tokens'] == len(ids)
            assert row['response_sha256'] == hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()
        np.testing.assert_allclose([r['kl_before'] for r in rows], before, atol=1e-6)
        np.testing.assert_allclose([r['kl_after'] for r in rows], after, atol=1e-6)
        np.testing.assert_allclose([r['gain'] for r in rows], before-after, atol=1e-6)
        gains = np.array([r['kl_before']-r['kl_after'] for r in rows])
        assert record['gain'] == pytest.approx(gains.mean())
        assert record['positive_fraction'] == (gains > record['gain_epsilon']).mean()
        assert ('subset_alignment' in record) == (checkpoint in [10, 30, 60, 100])
        if 'subset_alignment' in record:
            assert all(cross[key] == value for key, value in record['subset_alignment'].items())
    fixed.terminal(trainer)
    assert readouts == [*range(0, 101, 10), 106]
    assert subset_checks == [(i, i in cfg['geometry_steps']) for i in range(106)
                             if i == 0 or i == 105 or (i+1) % interval == 0 or i in cfg['geometry_steps']]
    detailed_files = list((run/'replication/gradient_study').glob('checkpoint_*/train_batch_*_hidden_moments.npz'))
    assert len(detailed_files) == len(cfg['geometry_steps'])*cfg['batches']
    with ZipFile(package_results(run)) as archive:
        prefix = 'training/training_records/rq3/'
        assert sorted(n for n in archive.namelist() if n.startswith(prefix)) == [
            f'{prefix}checkpoint_{i:03d}.json' for i in range(0, 101, 10)] + [
            f'{prefix}subset_alignment_{i+1:03d}.json' for i, _ in subset_checks]


def test_undefined_pairs_are_preserved_and_excluded_from_means():
    parameter = {'cosine': [[1, -0.5, None], [-0.5, 1, None], [None, None, None]]}
    hidden = {'cosine': [[None]*3 for _ in range(3)], 'overlap': {}}
    math = import_module(study_module.__package__ + '.opd_gradient_study_math')
    result = math.subset_alignment([[0], [1], [2]], parameter, hidden)
    assert len(result['pairs']) == 3
    assert result['means'] == dict(parameter_gradient_cosine=-0.5, mean_hidden_gradient_cosine=None,
                                   centered_covariance_top4_overlap=None)
    assert result['valid_pairs'] == dict(parameter_gradient_cosine=1, mean_hidden_gradient_cosine=0,
                                         centered_covariance_top4_overlap=0)
