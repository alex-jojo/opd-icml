"""Question progress stays aligned with saved answers/judgments on resume."""
from concurrent.futures import Future
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import code_benchmarks as cb
import code_benchmark_worker as worker
import watch_code_progress as watch


class Bar:
    created = []
    def __init__(self, **kwargs):
        self.options = kwargs
        self.updates = []
        self.created.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n): self.updates.append(n)
    def set_postfix(self, **kwargs): pass


def q(index):
    return dict(task_id=f'HumanEval/{index}', benchmark='humanevalplus', prompt='def add(a,b):\n', entry_point='add')


def row(question):
    return dict(task_id=question['task_id'], prompt_sha256=hashlib.sha256(cb.prompt_for(question).encode()).hexdigest(),
                responses=['def add(a,b): return a+b']*4)


def test_incremental_counter_handles_partial_append_rename_and_truncate(tmp_path):
    path = tmp_path / 'answers.jsonl'
    counter = watch.LineCounter()
    assert counter.read(path) == 0
    path.write_bytes(b'{"a":1}\n{"b":')
    assert counter.read(path) == counter.read(path) == 1
    with path.open('ab') as stream:
        stream.write(b'2}\n')
    assert counter.read(path) == 2
    path.write_bytes(b'')
    assert counter.read(path) == 0
    other = tmp_path / 'replacement'
    other.write_bytes(b'{}\n{}\n{}\n')
    other.replace(path)
    assert counter.read(path) == 3


def test_snapshot_reuses_prefix_and_never_counts_renamed_answers_twice(tmp_path):
    cb.save_json(tmp_path / 'run.json', dict(benchmarks=['livecodebench', 'humanevalplus'],
        data={'livecodebench': {'rows': 3}, 'humanevalplus': {'rows': 2}}))
    lcb = tmp_path / 'livecodebench'
    cb.save_jsonl(lcb / 'generation.partial.jsonl', [{}, {}])
    state = watch.EvaluationProgress(tmp_path)
    assert state.snapshot()['livecodebench'] == dict(total=3, generated=2, judged=0, stage='generating')
    with (lcb / 'generation.partial.jsonl').open('a') as stream:
        stream.write('{}\n')
    (lcb / 'generation.partial.jsonl').replace(lcb / 'generations.jsonl')
    cb.save_jsonl(lcb / 'livecodebench.results.jsonl', [{}])
    assert state.snapshot()['livecodebench'] == dict(total=3, generated=3, judged=1, stage='judging')
    cb.save_jsonl(lcb / 'generation.partial.jsonl', [{}]*3)
    assert state.snapshot()['livecodebench']['generated'] == 3
    cb.save_json(lcb / 'summary.json', dict(problems=3))
    assert state.snapshot()['livecodebench']['stage'] == 'done'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/watch_code_progress.py'),
        '--output-dir', str(tmp_path), '--once'], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['livecodebench']['judged'] == 3


def test_generation_bar_starts_at_saved_questions_and_suppresses_batch_bar(tmp_path, monkeypatch):
    Bar.created = []
    monkeypatch.setattr(worker, 'tqdm', Bar)
    fake_torch, fake_hf, fake_vllm = [types.ModuleType(n) for n in ('torch', 'transformers', 'vllm')]
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 4)
    tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]['content'],
                                     encode=lambda text, **kw: [1, 2])
    fake_hf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: tokenizer)
    calls = []
    class Model:
        def __init__(self, **kw): assert kw['tensor_parallel_size'] == 4
        def generate(self, inputs, sampling_params, use_tqdm):
            assert not use_tqdm
            calls.append(inputs)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='def add(a,b): return a+b',
                token_ids=[3, 4], finish_reason='stop') for _ in range(4)]) for _ in inputs]
    fake_vllm.LLM = Model
    fake_vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    for module in (fake_torch, fake_hf, fake_vllm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    questions = [q(0), q(1)]
    from evaluation_prompts import CHAT_PROMPT_ENCODING
    prefix = dict(row(questions[0]), prompt_encoding=CHAT_PROMPT_ENCODING)
    cb.save_jsonl(tmp_path / 'generation.partial.jsonl', [prefix])
    args = types.SimpleNamespace(model_path='unused', samples=4, max_prompt_tokens=1000, max_tokens=100,
        max_num_seqs=8, batch_size=2, temperature=1., top_p=1., seed=42, benchmark='humanevalplus')
    worker.generate(args, questions, tmp_path)
    assert Bar.created[0].options['initial'] == 1 and Bar.created[0].options['total'] == 2
    assert Bar.created[0].updates == [1] and len(calls[0]) == 1
    saved = list(cb.read_jsonl(tmp_path / 'generations.jsonl'))
    assert saved[0] == prefix
    cb.validate_generations(questions, saved, 4)


def test_lcb_judge_bar_counts_only_new_committed_questions(tmp_path, monkeypatch):
    Bar.created = []
    monkeypatch.setattr(worker, 'tqdm', Bar)
    class Pool:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task, codes, timeout):
            future = Future()
            future.set_result(dict(task_id=task['task_id'], grades=[True]*4))
            return future
    monkeypatch.setattr(worker, 'ProcessPoolExecutor', Pool)
    questions = [dict(q(i), benchmark='livecodebench') for i in range(2)]
    generations = [row(question) for question in questions]
    cb.save_jsonl(tmp_path / 'livecodebench.results.jsonl', [dict(task_id=questions[0]['task_id'], grades=[True]*4)])
    with gzip.open(tmp_path / 'tests.jsonl.gz', 'wt') as stream:
        for question in questions:
            stream.write(json.dumps(dict(task_id=question['task_id']))+'\n')
    args = types.SimpleNamespace(workers=1, timeout=6, samples=4, benchmark='livecodebench')
    grades, _ = worker.grade_lcb(args, questions, generations, tmp_path, tmp_path)
    assert grades == [[True]*4]*2
    assert Bar.created[0].options['initial'] == 1 and Bar.created[0].options['total'] == 2
    assert Bar.created[0].updates == [1]
