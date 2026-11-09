"""CPU checks for pinned model profiles, text conversion and OPD gradients."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Gemma3Config, Gemma3ForConditionalGeneration,
    Gemma3TextConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
    SiglipVisionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_assets


def tokenizer_fixture(path):
    vocab = {f'token{i}': i for i in range(128)}
    for index, token in [(0, '<pad>'), (1, '<eos>'), (2, '<bos>'), (3, '<unk>'),
                         (105, '<start_of_turn>'), (106, '<end_of_turn>')]:
        vocab.pop(f'token{index}')
        vocab[token] = index
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='<unk>')),
                                       bos_token='<bos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tokenizer.save_pretrained(path)


@pytest.fixture
def gemma_checkpoint(tmp_path):
    torch.manual_seed(42)
    text = Gemma3TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           vocab_size=128, sliding_window=16, max_position_embeddings=64,
                           query_pre_attn_scalar=8)
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8)
    config = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         mm_tokens_per_image=4, boi_token_index=100, eoi_token_index=101, image_token_index=102)
    model = Gemma3ForConditionalGeneration(config).eval()
    source = tmp_path / 'original'
    model.save_pretrained(source, max_shard_size='10KB')
    tokenizer_fixture(source)
    return source, model


def test_gemma_export_preserves_logits_weights_and_protocol(gemma_checkpoint, tmp_path):
    source, original = gemma_checkpoint
    target = model_assets.export_gemma_text(source, tmp_path / 'text')
    converted, info = AutoModelForCausalLM.from_pretrained(target, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    tokens = torch.tensor([[2, 10, 11, 12, 13]])
    with torch.no_grad():
        expected = original(tokens, use_cache=False).logits
        actual = converted.eval()(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, value in original.state_dict().items():
        new = model_assets.text_key(key)
        if new is not None:
            torch.testing.assert_close(converted.state_dict()[new], value, atol=0, rtol=0)
    assert converted.config.model_type == 'gemma3_text'
    assert converted.config.eos_token_id == 106
    assert not any('vision' in key or 'projector' in key for key in converted.state_dict())
    tokenizer = AutoTokenizer.from_pretrained(target)
    assert tokenizer.eos_token_id == 106
    assert tokenizer.get_vocab() == AutoTokenizer.from_pretrained(source).get_vocab()
    assert tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hello'}], tokenize=False,
                                         add_generation_prompt=True) == '<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n'
    assert model_assets.export_gemma_text(source, target) == target


def test_legacy_gemma_key_mapping():
    assert model_assets.text_key('language_model.model.layers.0.weight') == 'model.layers.0.weight'
    assert model_assets.text_key('language_model.lm_head.weight') == 'lm_head.weight'
    assert model_assets.text_key('model.language_model.norm.weight') == 'model.norm.weight'
    assert model_assets.text_key('model.vision_tower.weight') is None


def test_bfloat16_export_preserves_tensor_dtype(gemma_checkpoint, tmp_path):
    from safetensors import safe_open
    source, model = gemma_checkpoint
    bf16_source = tmp_path / 'bf16-source'
    model.to(torch.bfloat16).save_pretrained(bf16_source, max_shard_size='10KB')
    tokenizer_fixture(bf16_source)
    target = model_assets.export_gemma_text(bf16_source, tmp_path / 'bf16-text')
    expected = {model_assets.text_key(key): value for key, value in model.state_dict().items()
                if model_assets.text_key(key) is not None}
    for shard in target.glob('*.safetensors'):
        with safe_open(str(shard), framework='pt') as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                assert tensor.dtype == torch.bfloat16
                torch.testing.assert_close(tensor, expected[key], atol=0, rtol=0)


def test_local_gemma_pair_is_prepared_without_download(gemma_checkpoint, tmp_path, monkeypatch):
    import huggingface_hub
    source, _ = gemma_checkpoint
    def no_download(*args, **kwargs):
        raise AssertionError('Local checkpoints must not download weights')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', no_download)
    paths = model_assets.prepare_models('gemma3-12b', source, source, model_dir=tmp_path / 'prepared')
    assert paths['student'] != paths['teacher']
    assert all(AutoModelForCausalLM.from_pretrained(path).config.model_type == 'gemma3_text' for path in paths.values())


def test_cross_size_qwen_teacher():
    opd = load_opd()
    models = [Qwen3ForCausalLM(Qwen3Config(hidden_size=width, intermediate_size=width * 2, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, head_dim=width // 4, vocab_size=128)).eval()
              for width in [32, 64]]
    actors = [types.SimpleNamespace(actor_module=model, use_remove_padding=False, use_ulysses_sp=False,
              config=types.SimpleNamespace(policy_loss={})) for model in models]
    tokens = torch.tensor([[2, 10, 11, 12, 13]])
    inputs = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), position_ids=torch.arange(5)[None],
                  responses=tokens[:, -3:])
    ids, student, _, _ = opd.forward(actors[0], inputs, 1.)
    _, teacher, _, _ = opd.forward(actors[1], inputs, 1., ids, teacher=True)
    assert teacher.shape == student.shape == (1, 3, 16)
    assert torch.isfinite(opd.reverse_kl(student, teacher)).all()


def test_export_rejects_existing_unrelated_directory(gemma_checkpoint, tmp_path):
    source, _ = gemma_checkpoint
    destination = tmp_path / 'occupied'
    destination.mkdir()
    (destination / 'keep.txt').write_text('existing user data')
    with pytest.raises(ValueError, match='empty'):
        model_assets.export_gemma_text(source, destination)
    assert (destination / 'keep.txt').read_text() == 'existing user data'


@pytest.mark.parametrize('scale', ['1.7b', '4b', '8b', 'gemma3-12b'])
def test_launchers_dry_run_without_data_download_or_cuda(scale):
    for experiment in ['paired-full', 'original-full', 'early-stop', 'reuse']:
        result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--scale', scale,
                                 '--experiment', experiment, '--dry-run'], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        assert summary['rounds_cap'] == (53 if experiment == 'reuse' else 106)
        assert summary['updates_per_round'] == (2 if experiment == 'reuse' else 1)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_models.py'), '--scale', scale,
                             '--dry-run'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['models'] == model_assets.read_models(scale)


def test_new_profile_protocols():
    for scale in ['8b', 'gemma3-12b']:
        for name in ['paired-full', 'original-full', 'early-stop', 'reuse']:
            cfg = json.loads((ROOT / 'configs' / scale / (name + '.yaml')).read_text())
            policy = cfg['actor_rollout_ref']['actor']['policy_loss']
            assert policy['loss_mode'] == 'opd_top16'
            assert cfg['trainer']['n_gpus_per_node'] == policy['replication17']['distributed_world_size'] == 4
            assert cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size'] == 4
            if name != 'original-full':
                assert policy['gradient_study']['enabled']
                assert policy['fixed_heldout_probe_size'] == 0
            if name == 'reuse':
                assert policy['replication17']['mode'] == 'full'
                assert policy['replication17']['rounds'] * policy['jump_reuse_updates'] == 106


def load_opd():
    # Core forward/gradient functions can be tested without Ray, CUDA or DataProto.
    actor_path = ROOT / 'runtime/verl/workers/actor'
    package = types.ModuleType('opd_cpu_test')
    package.__path__ = [str(actor_path)]
    sys.modules[package.__name__] = package
    name = package.__name__ + '.opd_top16'
    spec = importlib.util.spec_from_file_location(name, actor_path / 'opd_top16.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('family', ['qwen3', 'gemma3'])
def test_opd_compact_logits_gradients_and_adam_update(family, gemma_checkpoint, tmp_path):
    opd = load_opd()
    if family == 'gemma3':
        source, _ = gemma_checkpoint
        target = model_assets.export_gemma_text(source, tmp_path / 'opd-text')
        model = AutoModelForCausalLM.from_pretrained(target, attn_implementation='eager')
    else:
        model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    actor = types.SimpleNamespace(actor_module=model, use_remove_padding=False, use_ulysses_sp=False,
                                 config=types.SimpleNamespace(policy_loss={'replication17': {'compact_head': True}}))
    tokens = torch.tensor([[2, 10, 11, 12, 13]])
    inputs = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                  position_ids=torch.arange(5)[None], responses=tokens[:, -3:])
    ids, dense, _, _ = opd.forward(actor, inputs, 1.)
    _, compact, _, _ = opd.forward(actor, inputs, 1., ids, diagnostics=False)
    torch.testing.assert_close(compact, dense, rtol=1e-5, atol=1e-6)
    teacher = torch.randn_like(dense)
    opd.reverse_kl(dense, teacher).mean().backward()
    expected = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    _, compact, _, _ = opd.forward(actor, inputs, 1., ids, diagnostics=False)
    loss = opd.reverse_kl(compact, teacher).mean()
    loss.backward()
    for key, parameter in model.named_parameters():
        if key in expected:
            torch.testing.assert_close(parameter.grad, expected[key], rtol=5e-4, atol=1e-6)
    before = model.get_input_embeddings().weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    assert torch.isfinite(loss) and not torch.equal(before, model.get_input_embeddings().weight)
