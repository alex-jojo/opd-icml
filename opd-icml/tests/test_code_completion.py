"""Base-model continuation prompts, tokenization, extraction and run identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import code_benchmarks as cb
import code_benchmark_worker as worker


def question(benchmark='humanevalplus'):
    return dict(benchmark=benchmark, task_id='test/0', prompt='def add(a,b):\n    """Add."""\n', entry_point='add')


@pytest.mark.parametrize('benchmark', ['humanevalplus', 'mbppplus'])
def test_native_code_prefix_and_indentation(benchmark):
    q = question(benchmark)
    if benchmark == 'mbppplus':
        q['prompt'] = '"""Return the sum of a and b.\nassert add(1,2)==3\n"""\n'
    prefix = cb.prompt_for(q, 'completion')
    assert prefix == q['prompt'] and 'Implement the following' not in prefix
    response = '    return a+b\n' if benchmark == 'humanevalplus' else 'def add(a,b):\n    return a+b\n'
    solution = cb.extract_solution(response, q, 'completion')
    assert solution == prefix + response
    scope = {}
    exec(solution, scope)  # This test fixture is owned code, not dataset/model output.
    assert scope['add'](2, 3) == 5


def test_lcb_base_code_block_is_completed_and_closed():
    q = dict(benchmark='livecodebench', task_id='test/0', prompt='Add two integers.', starter_code='')
    assert cb.prompt_for(q, 'completion').endswith('```python\n')
    assert 'standard input' in cb.prompt_for(q, 'completion')
    assert cb.extract_solution('print(3)\n```\nMore text', q, 'completion') == 'print(3)'


def test_completion_prompt_identity_and_old_chat_cache():
    q = question()
    row = dict(task_id=q['task_id'], prompt_style='completion',
               prompt_sha256=hashlib.sha256(cb.prompt_for(q, 'completion').encode()).hexdigest(), responses=['    return a+b']*4)
    cb.validate_generations([q], [row], 4)
    with pytest.raises(ValueError, match='identity'):
        cb.validate_generations([q], [dict(row, prompt_style='chat')], 4)
    row.pop('prompt_style')
    row['prompt_sha256'] = hashlib.sha256(cb.prompt_for(q).encode()).hexdigest()
    cb.validate_generations([q], [row], 4)


def test_base_generation_never_calls_chat_and_preserves_bos(tmp_path, monkeypatch):
    fake_torch, fake_hf, fake_vllm = [types.ModuleType(n) for n in ('torch', 'transformers', 'vllm')]
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 1)
    calls = []
    class Tokenizer:
        chat_template = None
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError('Base model has no chat template')
        def encode(self, text, add_special_tokens):
            assert add_special_tokens
            return [32013, 1, 2, 3]
    fake_hf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: Tokenizer())
    class Model:
        def __init__(self, **kwargs):
            assert kwargs['tensor_parallel_size'] == 1
        def generate(self, prompts, sampling_params, use_tqdm=True):
            assert use_tqdm is False
            calls.append(prompts)
            assert prompts == [{'prompt_token_ids': [32013, 1, 2, 3]}]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='    return a+b\n',
                token_ids=[4, 5], finish_reason='stop') for _ in range(sampling_params.n)])]
    fake_vllm.LLM = Model
    fake_vllm.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for module in (fake_torch, fake_hf, fake_vllm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    args = types.SimpleNamespace(model_path='unused', prompt_style='completion', samples=4, max_prompt_tokens=100,
        max_tokens=100, max_num_seqs=16, batch_size=4, temperature=1., top_p=1., seed=42, benchmark='humanevalplus')
    worker.generate(args, [question()], tmp_path)
    rows = list(cb.read_jsonl(tmp_path / 'generations.jsonl'))
    cb.validate_generations([question()], rows, 4)
    assert rows[0]['rendered_prompt'] == question()['prompt']
    assert rows[0]['prompt_style'] == 'completion' and rows[0]['prompt_tokens'] == 4
    assert len(calls) == 1
    args.prompt_style = 'chat'
    other = tmp_path / 'chat'
    other.mkdir()
    with pytest.raises(ValueError, match='--prompt-style completion'):
        worker.generate(args, [question()], other)


def test_completion_choice_is_in_run_plan(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_code.py'), '--model-path', 'unused',
        '--output-dir', str(tmp_path / 'run'), '--prompt-style', 'completion', '--dry-run'],
        capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['sampling']['prompt_style'] == 'completion' and plan['metric'] == 'pass@8'


def test_evalplus_receives_reconstructed_full_code(tmp_path, monkeypatch):
    fake = types.ModuleType('evalplus.evaluate')
    def evaluate(dataset, samples, parallel, i_just_wanna_run):
        saved = list(cb.read_jsonl(samples))
        assert all(row['solution'] == question()['prompt']+'    return a+b\n' for row in saved)
        path = Path(samples)
        cb.save_json(path.with_name(path.stem+'_eval_results.json'),
                     {'eval': {'test/0': [dict(base_status='pass', plus_status='pass')]*4}})
    fake.evaluate = evaluate
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(worker.importlib.metadata, 'version', lambda name: '0.3.1')
    row = dict(task_id='test/0', prompt_style='completion', responses=['    return a+b\n']*4)
    args = types.SimpleNamespace(benchmark='humanevalplus', workers=1, samples=4)
    grades, _ = worker.grade_evalplus(args, [question()], [row], tmp_path, tmp_path)
    assert grades == [[True]*4]
