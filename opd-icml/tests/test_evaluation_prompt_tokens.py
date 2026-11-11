"""Real Gemma tokenization at both evaluation entry points, without a GPU."""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, GemmaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import code_benchmarks as cb
import code_benchmark_worker as worker
import evaluate as math_eval
import evaluate_code as code_eval
from evaluation_prompts import CHAT_PROMPT_ENCODING
from model_assets import GEMMA_CHAT_TEMPLATE


@pytest.fixture
def gemma(tmp_path, monkeypatch):
    path = tmp_path/'gemma'
    vocab = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<unk>': 3,
             '<start_of_turn>': 4, '<end_of_turn>': 5}
    tokenizer = GemmaTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='<unk>')),
        add_bos_token=True, eos_token='<end_of_turn>', chat_template=GEMMA_CHAT_TEMPLATE,
        additional_special_tokens=['<start_of_turn>', '<end_of_turn>'])
    tokenizer.save_pretrained(path)
    config = dict(model_type='gemma3_text', bos_token_id=2, eos_token_id=5, pad_token_id=0)
    (path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    (path/'generation_config.json').write_text(json.dumps(config), encoding='utf-8')
    loaded = AutoTokenizer.from_pretrained(path, local_files_only=True)
    captured = []
    backend = types.ModuleType('vllm')

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs['model'] == kwargs['tokenizer'] == str(path)
            assert json.loads((path/'generation_config.json').read_text())['eos_token_id'] == loaded.eos_token_id == 5

        def generate(self, prompts, sampling_params, **kwargs):
            assert all(isinstance(p, dict) and set(p) == {'prompt_token_ids'} for p in prompts)
            assert all(p['prompt_token_ids'].count(loaded.bos_token_id) == 1 for p in prompts)
            assert not getattr(sampling_params, 'ignore_eos', False)
            captured.append((prompts, sampling_params))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=r'\boxed{2}',
                token_ids=[3, 5], finish_reason='stop', stop_reason=5) for _ in range(sampling_params.n)])
                for _ in prompts]

    backend.LLM = LLM
    backend.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', backend)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    return path, loaded, captured


def arguments(path):
    return types.SimpleNamespace(model_path=str(path), max_prompt_tokens=1024, max_tokens=100,
        max_num_seqs=16, temperature=1., top_p=.95, seed=42, n=8, samples=4,
        prompt_style='chat', benchmark='humanevalplus', batch_size=1)


@pytest.mark.parametrize('entry', ['math', 'code'])
def test_chat_entry_points_submit_training_ids_with_one_bos_and_intact_turns(gemma, tmp_path, entry):
    path, tokenizer, captured = gemma
    args = arguments(path)
    if entry == 'math':
        inputs = [dict(problem='1 + 1?', answer=2),
                  dict(problem='2 + 2?\n'+math_eval.BOXED_INSTRUCTION, answer=4)]
        rows = math_eval.generate(args, inputs)
        contents = [item['problem'] if item['problem'].endswith(math_eval.BOXED_INSTRUCTION)
                    else item['problem']+'\n'+math_eval.BOXED_INSTRUCTION for item in inputs]
    else:
        inputs = [dict(benchmark='humanevalplus', task_id=f'test/{i}',
                       prompt=f'def answer_{i}():\n', entry_point=f'answer_{i}') for i in range(2)]
        worker.generate(args, inputs, tmp_path)
        rows = list(cb.read_jsonl(tmp_path/'generations.jsonl'))
        cb.validate_generations(inputs, rows, args.samples)
        contents = [cb.prompt_for(q) for q in inputs]
    sent = [prompt for batch, _ in captured for prompt in batch]
    for row, prompt, content in zip(rows, sent, contents):
        expected = tokenizer.apply_chat_template([dict(role='user', content=content)],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt['prompt_token_ids'] == expected
        assert row['rendered_prompt'].startswith('<bos><start_of_turn>user\n')
        assert row['rendered_prompt'].endswith('<end_of_turn>\n<start_of_turn>model\n')
        assert tokenizer.convert_tokens_to_ids('<end_of_turn>') in expected
        # This fixture exposes the bug if rendered strings are encoded by default.
        assert tokenizer.encode(row['rendered_prompt']) == [tokenizer.bos_token_id]+expected
        assert len(row['responses']) == (args.n if entry == 'math' else args.samples)
        if entry == 'math':
            assert row['generation_execution']['prompt_encoding'] == CHAT_PROMPT_ENCODING
            assert row['stop_reasons'] == [5]*args.n
        else:
            assert row['prompt_encoding'] == CHAT_PROMPT_ENCODING
            assert row['prompt_tokens'] == len(expected)
    for _, params in captured:
        assert (params.temperature, params.top_p, params.seed, params.max_tokens) == (1., .95, 42, 100)
        assert not hasattr(params, 'stop') and not hasattr(params, 'stop_token_ids')


@pytest.mark.parametrize('entry', ['math', 'code'])
def test_prompt_length_limit_applies_to_the_exact_submitted_ids(gemma, tmp_path, entry):
    path, _, captured = gemma
    args = arguments(path)
    args.max_prompt_tokens = 1
    with pytest.raises(ValueError, match='refusing'):
        if entry == 'math':
            math_eval.generate(args, [dict(problem='1 + 1?', answer=2)])
        else:
            worker.generate(args, [dict(benchmark='humanevalplus', task_id='test/0',
                prompt='def answer():\n', entry_point='answer')], tmp_path)
    assert captured == []


@pytest.mark.parametrize('cache_name', ['generations.jsonl', 'generation.partial.jsonl'])
def test_code_worker_refuses_legacy_chat_cache_without_modifying_it(gemma, tmp_path, cache_name):
    path, _, captured = gemma
    q = dict(benchmark='humanevalplus', task_id='test/0', prompt='def answer():\n', entry_point='answer')
    old = dict(task_id=q['task_id'], responses=['pass']*4,
        prompt_sha256=hashlib.sha256(cb.prompt_for(q).encode()).hexdigest())
    cache = tmp_path/cache_name
    cb.save_jsonl(cache, [old])
    before = cache.read_bytes()
    with pytest.raises(ValueError, match='different prompt encoding'):
        worker.generate(arguments(path), [q], tmp_path)
    assert cache.read_bytes() == before and captured == []


@pytest.mark.parametrize('cache_name', ['generations.jsonl', 'generation.partial.jsonl', 'generation.seed.jsonl'])
@pytest.mark.parametrize('parallel', ['tensor', 'data'])
def test_code_launcher_rejects_legacy_serial_and_parallel_cache_before_generation(gemma, tmp_path, monkeypatch, cache_name, parallel):
    path, _, captured = gemma
    q = dict(benchmark='humanevalplus', task_id='test/0', prompt='def answer():\n', entry_point='answer')
    data, output = tmp_path/'data', tmp_path/'evaluation'
    questions = data/'humanevalplus/questions.jsonl'
    cb.save_jsonl(questions, [q])
    cb.save_json(questions.with_name('manifest.json'), dict(protocol=cb.PROTOCOL,
        benchmark='humanevalplus', files={'questions.jsonl': cb.digest(questions)}))
    cache = output/'humanevalplus'/cache_name
    cb.save_jsonl(cache, [dict(task_id=q['task_id'], responses=['pass']*4,
        prompt_sha256=hashlib.sha256(cb.prompt_for(q).encode()).hexdigest())])
    before = cache.read_bytes()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setattr(sys, 'argv', ['evaluate_code.py', '--model-path', str(path),
        '--benchmark', 'humanevalplus', '--data-dir', str(data), '--output-dir', str(output),
        '--parallel-mode', parallel, '--generate-only'])
    with pytest.raises(ValueError, match='different prompt encoding'):
        code_eval.main()
    assert cache.read_bytes() == before and captured == []
    assert not (output/'run.json').exists()


def test_math_cache_encoding_change_requires_new_generation_but_allows_regrading(gemma, tmp_path, monkeypatch):
    path, _, captured = gemma
    source, dest = tmp_path/'math.jsonl', tmp_path/'answers.jsonl'
    math_eval.save_rows(source, [dict(problem='1 + 1?', answer=2)])
    argv = ['evaluate.py', '--input_file', str(source), '--model_path', str(path),
            '--output_file', str(dest), '--n', '2', '--score_workers', '1']
    monkeypatch.setattr(math_eval, 'grade', lambda rows, *args: [dict(row, acc_list=[True]*2) for row in rows])
    monkeypatch.setattr(sys, 'argv', argv)
    math_eval.main()
    metadata = dest.with_suffix('.generations.meta.json')
    saved = json.loads(metadata.read_text(encoding='utf-8'))
    assert saved['identity'].pop('prompt_encoding') == CHAT_PROMPT_ENCODING
    math_eval.save_json(metadata, saved)
    cache = dest.with_suffix('.generations.jsonl')
    before = cache.read_bytes()
    with pytest.raises(ValueError, match='prompt encoding identity changed'):
        math_eval.main()
    assert len(captured) == 1 and cache.read_bytes() == before
    monkeypatch.setattr(sys, 'argv', argv+['--regrade_only'])
    math_eval.main()
    assert len(captured) == 1 and cache.read_bytes() == before
