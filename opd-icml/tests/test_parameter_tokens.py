"""Exact token Jacobian checks on a causal network, without CUDA or Ray."""
import copy
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('parameter_tokens_cpu')
package.__path__ = [str(ROOT / 'runtime/verl/workers/actor')]
sys.modules[package.__name__] = package
audit_module = importlib.import_module(package.__name__ + '.opd_parameter_tokens')
opd = importlib.import_module(package.__name__ + '.opd_top16')
study_module = importlib.import_module(package.__name__ + '.opd_study_replication')


class Batch:
    def __init__(self, batch):
        self.batch = batch
        self.meta_info = dict(temperature=1.)

    def __len__(self):
        return len(self.batch['responses'])

    def select_idxs(self, indices):
        return Batch({k: v[indices] for k, v in self.batch.items()})

    def split(self, size):
        return [self.select_idxs(list(range(i, min(i+size, len(self))))) for i in range(0, len(self), size)]


@dataclass
class Output:
    logits: torch.Tensor


class CausalModel(torch.nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(16, 4)
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 16, bias=False)
        self.config = types.SimpleNamespace(tie_word_embeddings=tied, eos_token_id=15)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict, logits_to_keep):
        h = self.model.embed_tokens(input_ids) * attention_mask[..., None]
        # Each loss depends on earlier positions as well as its own position.
        h = h.cumsum(1) / torch.arange(1, h.shape[1]+1, device=h.device)[None, :, None]
        h = torch.tanh(self.model.layers[0](h))
        h = self.model.norm(h)
        return Output(logits=self.lm_head(h[:, -logits_to_keep:]))


def make_actor(tied=False, compact=True):
    model = CausalModel(tied)
    return types.SimpleNamespace(actor_module=model, actor_optimizer=torch.optim.AdamW(model.parameters(), lr=.001),
        use_remove_padding=False, use_ulysses_sp=False,
        config=types.SimpleNamespace(ppo_micro_batch_size_per_gpu=1,
            policy_loss={'replication17': {'compact_head': compact}}))


def make_data():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    return Batch(dict(input_ids=tokens, attention_mask=torch.ones_like(tokens),
        position_ids=torch.arange(6)[None].repeat(2, 1), responses=tokens[:, -3:],
        response_mask=torch.tensor([[1, 0, 1], [1, 1, 1]]),
        opd_top16_ids=torch.arange(16)[None, None].repeat(2, 3, 1),
        opd_teacher_top16_logits=torch.randn(2, 3, 16)))


def objective(actor, data):
    _, logits, _, _ = opd.forward(actor, data.batch, 1., data.batch['opd_top16_ids'], diagnostics=False)
    return opd.reverse_kl(logits, data.batch['opd_teacher_top16_logits'])


def reference_energy(actor, data):
    values = objective(actor, data)[data.batch['response_mask'].bool()]
    params = list(actor.actor_module.parameters())
    result = []
    for value in values:
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        result.append(sum(float(g.double().square().sum()) for g in gradients if g is not None))
    return np.asarray(result)


def optimizer_update(actor, data):
    actor.actor_optimizer.zero_grad(set_to_none=True)
    values = objective(actor, data)
    values[data.batch['response_mask'].bool()].mean().backward()
    actor.actor_optimizer.step()
    actor.actor_optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize('tied,compact', [(False, False), (True, True)])
def test_exact_all_matches_full_token_jacobian_and_preserves_training(tied, compact):
    torch.manual_seed(19)
    actor, data = make_actor(tied, compact), make_data()
    optimizer_update(actor, data)  # Include populated Adam moments in invariance check.
    expected = reference_energy(actor, data)
    baseline = copy.deepcopy(actor)
    weights = copy.deepcopy(actor.actor_module.state_dict())
    optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    result = audit_module.FullParameterTokenAudit(actor, {'mode': 'exact_all'}).measure(data, 1, 0)
    np.testing.assert_allclose(result['energy'], expected, rtol=2e-5, atol=1e-7)
    assert result['backwards'] == 5
    np.testing.assert_array_equal(result['pos'], [0, 2, 0, 1, 2])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert actor.actor_module.training
    assert all(p.grad is None for p in actor.actor_module.parameters())
    for key, value in weights.items():
        assert torch.equal(value, actor.actor_module.state_dict()[key])
    for index, values in optimizer['state'].items():
        for key, value in values.items():
            assert torch.equal(value, actor.actor_optimizer.state_dict()['state'][index][key])
    optimizer_update(actor, data)
    optimizer_update(baseline, data)
    for actual, ordinary in zip(actor.actor_module.parameters(), baseline.actor_module.parameters()):
        torch.testing.assert_close(actual, ordinary, rtol=0, atol=0)


def test_parameter_energy_is_not_head_only():
    torch.manual_seed(4)
    actor, data = make_actor(), make_data()
    objective(actor, data)[0, 0].backward()
    head = float(actor.actor_module.lm_head.weight.grad.double().square().sum())
    full = audit_module.GradientEnergy(actor.actor_module).squared_norm()
    assert full > head * 1.01
    # Earlier context embeddings receive gradients from the selected token.
    assert actor.actor_module.model.embed_tokens.weight.grad[1].norm() > 0


def test_rejects_dropout_and_does_not_silently_estimate():
    actor = make_actor()
    with pytest.raises(ValueError, match='no sampling'):
        audit_module.FullParameterTokenAudit(actor, {'mode': 'sample'})
    actor.actor_module.model.layers.append(torch.nn.Dropout(.1))
    with pytest.raises(ValueError, match='zero dropout'):
        audit_module.FullParameterTokenAudit(actor, {'mode': 'exact_all'})


def test_same_population_groups_and_outputs(tmp_path):
    torch.manual_seed(42)
    actor, data = make_actor(tied=True), make_data()
    cfg = actor.config.policy_loss
    cfg['replication17'].update(mode='full', output_dir=str(tmp_path / 'replication'), rank=2,
        freeze_step=10, interval=5, rounds=6, reuse=1, tokenizer_path=None)
    cfg['hidden_projection'] = dict(asset_path=str(tmp_path / 'basis.npz'))
    cfg['gradient_study'] = dict(enabled=True, parameter_tokens=dict(enabled=True, mode='exact_all'))
    replication = study_module.StudyReplication(actor)
    replication.begin(data, None, 1)
    replication.before_inner(0)
    cursor = 0
    for part in data.split(1):
        with replication.watch_gradient_scopes(part.batch, 0, {}), replication.watch_training(part.batch, 0, cursor):
            values = objective(actor, part)
            (values * part.batch['response_mask']).sum().div(5).backward()
        cursor += len(part)
    before = [p.grad.clone() for p in actor.actor_module.parameters()]
    replication.before_update(0)
    for p, original in zip(actor.actor_module.parameters(), before):
        assert torch.equal(p.grad, original)
    report = json.loads((tmp_path / 'replication/token_audit/step_001_inner_1.json').read_text())
    with np.load(tmp_path / 'replication/token_audit/step_001_inner_1.npz') as arrays:
        energy = arrays['parameter_gradient_energy'].copy()
        kl = arrays['kl'].copy()
        assert len(energy) == len(arrays['raw_gradient_energy']) == 5
    expected = reference_energy(actor, data)
    np.testing.assert_allclose(energy, expected, rtol=2e-5, atol=1e-7)
    high = np.argsort(-kl, kind='stable')[:1]
    share = report['groups']['kl_top10']['quantities']['parameter_gradient_energy']['share']
    assert share == pytest.approx(energy[high].sum()/energy.sum())
    assert report['parameter_gradient_rms'] == pytest.approx(np.sqrt(energy.mean()))
    assert report['parameter_gradient_frobenius'] == pytest.approx(np.sqrt(energy.sum()))
    assert report['parameter_gradient_measurement']['coverage_fraction'] == 1.
    assert not report['parameter_gradient_measurement']['sampled']
    for label in ('low', 'middle', 'high'):
        assert 'parameter_gradient_rms' in report['kl_high_middle_low'][label]
    replication.begin(data, None, 2)
    assert not replication.measure
    actor.actor_optimizer.zero_grad(set_to_none=True)
    replication.before_inner(0)
    assert replication.parameter_measurement is None


def test_join_reorders_ids_and_rejects_missing_population():
    fields = dict(row=np.array([1, 0]), pos=np.array([2, 1]), kl=np.array([.3, .2]))
    measurement = dict(row=np.array([0, 1]), pos=np.array([1, 2]), energy=np.array([4., 9.]),
                       diagnostic_kl=np.array([.2, .3]), seconds=1., backwards=2)
    audit_module.attach_parameter_energy(fields, measurement)
    np.testing.assert_array_equal(fields['parameter_gradient_energy'], [9., 4.])
    measurement['pos'][1] = 3
    with pytest.raises(ValueError, match='SAME valid tokens'):
        audit_module.attach_parameter_energy(fields, measurement)


def test_export_new_and_legacy_logs(tmp_path):
    spec = importlib.util.spec_from_file_location('signal_export_cpu', ROOT / 'analysis/extract_signal.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    records = tmp_path / 'training_records'
    records.mkdir()
    for step in (1, 2):
        (records / f'post_update_{step:04}.json').write_text(json.dumps(
            dict(step=step, metrics={'actor/opd_top16_kl': .1, 'actor/grad_norm': 2.})))
    audits = tmp_path / 'replication/token_audit'
    audits.mkdir(parents=True)
    (audits / 'step_001_inner_1.json').write_text(json.dumps(dict(
        groups={'kl_top10': {'quantities': {'raw_gradient_energy': {'share': .8},
                  'parameter_gradient_energy': {'share': .7}}},
                'kl_above_step1_p90': {'fraction': .1, 'quantities': {'parameter_gradient_energy': {'share': .7}}}},
        parameter_gradient_rms=3., parameter_gradient_measurement={'extra_backwards': 5, 'seconds': 1.})))
    rows = [dict(zip(exporter.FIELDS, row)) for row in exporter.read_run(tmp_path, 'tiny')]
    assert rows[0]['top_kl_decile_parameter_energy_pct'] == 70.
    assert rows[0]['parameter_gradient_rms'] == 3.
    assert rows[0]['parameter_grad_norm'] == 2.
    assert rows[1]['top_kl_decile_parameter_energy_pct'] == ''


def test_paired_profiles_enable_exact_for_math_and_code():
    spec = importlib.util.spec_from_file_location('parameter_launcher_cpu', ROOT / 'train.py')
    launcher = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(launcher)
    for scale in ('1.7b', '4b', '8b', 'gemma3-12b'):
        for task in ('math', 'code'):
            _, cfg = launcher.training_config(ROOT / 'configs' / scale, 'paired-full', task)
            options = cfg['actor_rollout_ref']['actor']['policy_loss']['gradient_study']['parameter_tokens']
            assert options == dict(enabled=True, mode='exact_all')
