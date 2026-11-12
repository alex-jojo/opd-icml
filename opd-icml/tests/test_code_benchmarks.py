"""CPU protocol checks for pass@8 code benchmarking; no dataset code executes."""
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
from math import comb
from pathlib import Path
import pickle
import subprocess
import sys
import types
import zlib

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'third_party/livecodebench')]
import code_benchmarks as cb
import prepare_code_benchmarks as prepare
import evaluate_code as evaluate
import code_benchmark_worker as worker


def question(benchmark='humanevalplus', task_id='HumanEval/0'):
    return dict(benchmark=benchmark, task_id=task_id, prompt='def add(a, b):\n    """Add two numbers."""\n', entry_point='add')


def generation(q, samples=8):
    return dict(task_id=q['task_id'], prompt_sha256=hashlib.sha256(cb.prompt_for(q).encode()).hexdigest(),
                responses=['```python\ndef add(a,b):\n    return a+b\n```'] * samples)


def lcb_row(contest='2025-01-04T00:00:00', qid='q1'):
    private = json.dumps([dict(input='2 3', output='5', testtype='stdin')])
    return dict(question_id=qid, question_content='Add two integers.', starter_code='', contest_date=contest,
                difficulty='easy', platform='atcoder', metadata='{}',
                public_test_cases=json.dumps([dict(input='1 2', output='3', testtype='stdin')]),
                private_test_cases=base64.b64encode(zlib.compress(pickle.dumps(private))).decode())


@pytest.mark.parametrize('n', [8, 9, 16, 20])
def test_pass8_is_unbiased_and_matches_official_lcb(n):
    from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k
    for c in range(n + 1):
        grades = [True] * c + [False] * (n - c)
        expected = 1. - (comb(n-c, 8) / comb(n, 8) if n-c >= 8 else 0.)
        actual = cb.summarize_pass8([grades])['pass@8']
        assert actual == pytest.approx(expected)
        assert actual == pytest.approx(float(estimate_pass_at_k(n, [c], 8)[0]))


def test_pass8_is_any_complete_solution_not_average_sample_accuracy():
    summary = cb.summarize_pass8([[False] * 7 + [True], [False] * 8])
    assert summary['pass@8'] == .5 and summary['samples'] == 16
    with pytest.raises(ValueError):
        cb.summarize_pass8([[True] * 7])
    with pytest.raises(ValueError):
        cb.summarize_pass8([])


def test_estimator_matches_pinned_evalplus_when_available():
    from evalplus.eval import estimate_pass_at_k
    for c in range(9):
        expected = float(estimate_pass_at_k(8, [c], 8)[0])
        assert cb.summarize_pass8([[True] * c + [False] * (8-c)])['pass@8'] == pytest.approx(expected)


@pytest.mark.parametrize('samples', [4, 7])
def test_cli_rejects_too_few_answers_for_pass8(tmp_path, samples):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_code.py'),
        '--model-path', 'unused', '--output-dir', str(tmp_path / 'run'),
        '--samples', str(samples), '--dry-run'], capture_output=True, text=True)
    assert result.returncode != 0 and 'pass@8 requires >=8' in result.stderr


def test_old_eval_protocol_cannot_be_reused_as_pass8():
    old = dict(protocol=cb.PROTOCOL, metric='pass@4', sampling={'samples': 8})
    current = dict(protocol=cb.EVALUATION_PROTOCOL, metric='pass@8', sampling={'samples': 8})
    assert not evaluate.same_identity(old, current)


def test_legacy_prepared_data_remains_reusable(tmp_path):
    directory = tmp_path / 'humanevalplus'
    cb.save_jsonl(directory / 'questions.jsonl', [question()])
    spec = cb.lockfile()['datasets']['humanevalplus']
    manifest = dict(protocol=cb.PROTOCOL, benchmark='humanevalplus', source=spec,
                    release=spec['official_release'], start_date=None, end_date=None,
                    rows=1, default_samples=4, metric='pass@4',
                    files={'questions.jsonl': cb.digest(directory / 'questions.jsonl')})
    cb.save_json(directory / 'manifest.json', manifest)
    assert prepare.prepare('humanevalplus', tmp_path, tmp_path / 'unused-cache') == manifest


def test_evalplus_requires_base_and_plus_and_complete_task_coverage():
    row = dict(task_id='HumanEval/0', base_status='pass', plus_status='fail')
    raw = {'eval': {'HumanEval/0': [row] * 4}}
    assert cb.evalplus_grades(raw, ['HumanEval/0'], 4) == [[False] * 4]
    raw['eval']['HumanEval/0'][0] = dict(row, plus_status='pass')
    assert cb.evalplus_grades(raw, ['HumanEval/0'], 4) == [[True, False, False, False]]
    with pytest.raises(ValueError, match='task IDs'):
        cb.evalplus_grades(raw, ['HumanEval/1'], 4)
    with pytest.raises(ValueError, match='incomplete'):
        cb.evalplus_grades(raw, ['HumanEval/0'], 5)


def test_private_decode_accepts_json_string_and_rejects_executable_pickle():
    row = lcb_row()
    sample = cb.lcb_sample(row)
    cases = json.loads(sample['input_output'])
    assert sample['public_tests'] == sample['private_tests'] == 1
    assert cases['inputs'] == ['1 2', '2 3'] and cases['outputs'] == ['3', '5']
    row['private_test_cases'] = json.dumps([dict(input='9 9', output='18', testtype='stdin')])
    assert json.loads(cb.lcb_sample(row)['input_output'])['inputs'][-1] == '9 9'
    payload = pickle.dumps(Path)
    with pytest.raises(ValueError, match='only a JSON string'):
        cb.private_tests(base64.b64encode(zlib.compress(payload)).decode())


def test_dates_are_inclusive_and_lcb_preparation_keeps_hidden_tests(tmp_path):
    source = tmp_path / 'test.jsonl'
    cb.save_jsonl(source, [lcb_row('2024-12-31T23:59:59', 'old'), lcb_row(), lcb_row('2025-01-04T23:59:59', 'late')])
    output = tmp_path / 'prepared'
    questions, _ = prepare.build_livecodebench({}, [source], output, 'release_v6', '2025-01-04', '2025-01-04')
    assert len(questions) == 2 and all('private_test_cases' not in q for q in questions)
    tests = list(cb.read_jsonl(output / 'tests.jsonl.gz'))
    assert all(r['private_tests'] == 1 for r in tests)
    assert [q['task_id'] for q in questions] == [r['task_id'] for r in tests]


def test_evalplus_preparation_uses_official_tests_and_excludes_answers_from_prompts(tmp_path):
    hf = tmp_path / 'hf.parquet'
    pq.write_table(pa.Table.from_pylist([dict(task_id='HumanEval/0', prompt='stub', canonical_solution='SECRET_CANONICAL')]), hf)
    official = tmp_path / 'official.jsonl.gz'
    with gzip.open(official, 'wt', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(task_id='HumanEval/0', prompt=question()['prompt'], entry_point='add',
                                    canonical_solution='SECRET_CANONICAL', base_input=[[1, 2]], plus_input=[[9, 9]])) + '\n')
    output = tmp_path / 'bundle'
    rows, _ = prepare.build_evalplus('humanevalplus', {'rows': 1}, hf, official, output)
    assert 'SECRET_CANONICAL' not in cb.prompt_for(rows[0])
    assert 'canonical_solution' not in rows[0] and 'plus_input' not in rows[0]
    saved = list(cb.read_jsonl(output / 'official.jsonl'))[0]
    assert saved['plus_input'] == [[9, 9]]


def test_manifest_detects_changed_tests(tmp_path):
    directory = tmp_path / 'humanevalplus'
    cb.save_jsonl(directory / 'questions.jsonl', [question()])
    cb.save_json(directory / 'manifest.json', dict(protocol=cb.PROTOCOL, benchmark='humanevalplus',
                  files={'questions.jsonl': cb.digest(directory / 'questions.jsonl')}))
    cb.read_manifest(tmp_path, 'humanevalplus')
    with (directory / 'questions.jsonl').open('a') as stream:
        stream.write('changed')
    with pytest.raises(ValueError, match='changed'):
        cb.read_manifest(tmp_path, 'humanevalplus')


def test_generation_validation_rejects_missing_samples_and_changed_prompt():
    q, row = question(), generation(question())
    cb.validate_generations([q], [row], 8)
    with pytest.raises(ValueError, match='Incomplete'):
        cb.validate_generations([q], [dict(row, responses=row['responses'][:3])], 8)
    with pytest.raises(ValueError, match='identity'):
        cb.validate_generations([dict(q, prompt='different task')], [row], 8)


def test_extracts_final_code_and_humaneval_completion():
    q = question()
    response = 'reasoning\n```python\nwrong\n```\n```python\ndef add(a,b): return a+b\n```'
    assert cb.extract_solution(response, q) == 'def add(a,b): return a+b'
    assert cb.extract_solution('    return a + b\n', q) == q['prompt'] + '    return a + b\n'


def test_both_benchmarks_default_to_eight_samples_without_loading_data_or_models(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_code.py'), '--model-path', 'unused-model',
                             '--output-dir', str(tmp_path / 'run'), '--dry-run'], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['metric'] == 'pass@8' and plan['sampling']['samples'] == 8
    assert plan['benchmarks'] == ['livecodebench', 'humanevalplus']
    assert not (tmp_path / 'run').exists()


def test_pipeline_grade_only_never_loads_model_and_rejects_changed_data(tmp_path, monkeypatch):
    data, output, model = tmp_path / 'data', tmp_path / 'output', tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    q = question()
    directory = data / 'humanevalplus'
    cb.save_jsonl(directory / 'questions.jsonl', [q])
    manifest = dict(protocol=cb.PROTOCOL, benchmark='humanevalplus',
                    files={'questions.jsonl': cb.digest(directory / 'questions.jsonl')})
    cb.save_json(directory / 'manifest.json', manifest)
    calls = []
    def run(command, **kwargs):
        mode = command[command.index('--mode') + 1]
        calls.append((mode, kwargs))
        destination = output / 'humanevalplus'
        if mode == 'generate':
            cb.save_jsonl(destination / 'generations.jsonl', [generation(q)])
        else:
            cb.save_json(destination / 'summary.json', dict(benchmark='humanevalplus', **cb.summarize_pass8([[True] + [False] * 7])))
    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    argv = ['evaluate_code.py', '--benchmark', 'humanevalplus', '--data-dir', str(data), '--output-dir', str(output)]
    monkeypatch.setattr(sys, 'argv', argv + ['--model-path', str(model)])
    evaluate.main()
    assert [mode for mode, _ in calls] == ['generate', 'grade']
    assert calls[-1][1]['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert json.loads((output / 'summary.json').read_text())['benchmarks']['humanevalplus']['pass@8'] == 1.
    from evaluation_prompts import CHAT_PROMPT_ENCODING
    marker = output / 'run.json'
    legacy = json.loads(marker.read_text(encoding='utf-8'))
    assert legacy.pop('prompt_encoding') == CHAT_PROMPT_ENCODING
    cb.save_json(marker, legacy)
    calls.clear()
    with pytest.raises(ValueError, match='prompt encoding identity changed'):
        evaluate.main()
    assert calls == []
    # Saved model identity suffices for regrading, even if its path is unavailable.
    (model / 'config.json').unlink()
    calls.clear()
    monkeypatch.setattr(sys, 'argv', argv + ['--grade-only'])
    evaluate.main()
    assert [mode for mode, _ in calls] == ['grade']
    cb.save_jsonl(directory / 'questions.jsonl', [dict(q, prompt='changed')])
    manifest['files']['questions.jsonl'] = cb.digest(directory / 'questions.jsonl')
    cb.save_json(directory / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='identity changed'):
        evaluate.main()


def test_generation_worker_outputs_four_responses_and_checks_prompt_limit(tmp_path, monkeypatch):
    fake_torch, fake_hf, fake_vllm = [types.ModuleType(name) for name in ['torch', 'transformers', 'vllm']]
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 1)
    tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[0]['content'],
                                      encode=lambda text, **kwargs: list(range(len(text))))
    fake_hf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: tokenizer)
    class LLM:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, sampling_params, use_tqdm=True):
            assert use_tqdm is False
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='def add(a,b): return a+b',
                                token_ids=[1, 2], finish_reason='stop') for _ in range(sampling_params.n)]) for _ in prompts]
    fake_vllm.LLM = LLM
    fake_vllm.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for module in [fake_torch, fake_hf, fake_vllm]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    args = types.SimpleNamespace(model_path='unused', samples=4, max_prompt_tokens=1000, max_tokens=100,
                                 max_num_seqs=16, temperature=1., top_p=1., seed=42, batch_size=1, benchmark='humanevalplus')
    worker.generate(args, [question()], tmp_path)
    rows = list(cb.read_jsonl(tmp_path / 'generations.jsonl'))
    assert len(rows[0]['responses']) == 4
    cb.validate_generations([question()], rows, 4)
    other = tmp_path / 'overlong'
    other.mkdir()
    args.max_prompt_tokens = 1
    with pytest.raises(ValueError, match='refusing to truncate/drop'):
        worker.generate(args, [question()], other)


def test_official_lcb_judge_checks_entire_test_set(monkeypatch):
    import lcb_runner.evaluation.compute_code_generation_metrics as official
    monkeypatch.setattr(official, 'evaluate_generations_by_problem',
                        lambda args: ([[True, True], [True, False], [True], [-2]], [{}, {}, {}, {}]))
    result = worker.lcb_one(dict(task_id='LCB/0', input_output='{}', public_tests=1, private_tests=1), ['a'] * 4, 6)
    assert result['grades'] == [True, False, False, False]


def test_evalplus_adapter_passes_complete_solutions_and_override(monkeypatch, tmp_path):
    fake = types.ModuleType('evalplus.evaluate')
    calls = []
    def official(dataset, samples, parallel, i_just_wanna_run):
        calls.append(dict(dataset=dataset, samples=samples, parallel=parallel, i_just_wanna_run=i_just_wanna_run))
        path = Path(samples)
        cb.save_json(path.with_name(path.stem + '_eval_results.json'),
                     {'eval': {'HumanEval/0': [dict(base_status='pass', plus_status='pass')] * 4}})
    fake.evaluate = official
    monkeypatch.setitem(sys.modules, 'evalplus.evaluate', fake)
    monkeypatch.setattr(worker.importlib.metadata, 'version', lambda name: '0.3.1')
    args = types.SimpleNamespace(benchmark='humanevalplus', workers=2, samples=4)
    grades, _ = worker.grade_evalplus(args, [question()], [generation(question(), samples=4)], tmp_path, tmp_path)
    assert grades == [[True] * 4]
    assert calls[0]['dataset'] == 'humaneval' and calls[0]['parallel'] == 2
    assert calls[0]['i_just_wanna_run'] is True
    assert len(list(cb.read_jsonl(tmp_path / 'evalplus.samples.jsonl'))) == 4


def test_official_vendored_sources_have_not_changed():
    source = ROOT / 'third_party/livecodebench'
    manifest = json.loads((source / 'UPSTREAM.json').read_text())
    for name, expected in manifest['files'].items():
        assert cb.digest(source / name) == expected


def test_official_lcb_runner_with_owned_small_fixtures(monkeypatch):
    # Only these finite, owned fixture programs are executed by this CPU test.
    import lcb_runner.evaluation.testing_util as official
    monkeypatch.setattr(official, 'reliability_guard', lambda: None)
    monkeypatch.setattr(official.signal, 'SIGALRM', 14, raising=False)
    monkeypatch.setattr(official.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(official.signal, 'alarm', lambda *args: None, raising=False)
    sample = {'input_output': json.dumps(dict(inputs=['1 2\n', '3 4\n'], outputs=['3\n', '7\n'], fn_name=None))}
    results, _ = official.run_test(sample, test='a,b=map(int,input().split())\nprint(a+b)', debug=False, timeout=1)
    assert len(results) == 2 and all(bool(value == True) for value in results)



@pytest.mark.parametrize('script,extra', [
    ('evaluate_code.py', ['--output-dir', 'unused', '--model-path', 'unused', '--dry-run']),
    ('prepare_code_benchmarks.py', ['--dry-run']),
    ('code_benchmark_worker.py', ['--mode', 'generate', '--data-dir', 'unused', '--output-dir', 'unused']),
])
def test_mbpp_is_rejected_by_every_cli(script, extra):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                             '--benchmark', 'mbppplus', *extra], capture_output=True, text=True)
    assert result.returncode != 0 and 'invalid choice' in result.stderr


def test_no_mbpp_download_in_config():
    assert set(cb.lockfile()['datasets']) == {'livecodebench', 'humanevalplus'}


def test_removing_mbpp_can_reuse_only_identical_remaining_results():
    current = dict(protocol=cb.EVALUATION_PROTOCOL, benchmarks=list(cb.BENCHMARKS),
                   data={'livecodebench': {'hash': 'a'}, 'humanevalplus': {'hash': 'b'}},
                   model={'path': 'model', 'sha256': {'weights': 'xyz'}},
                   sampling={'samples': 8}, metric='pass@8', prompt_encoding='test')
    old = copy.deepcopy(current)
    old['benchmarks'] = ['livecodebench', 'mbppplus', 'humanevalplus']
    old['data']['mbppplus'] = {'hash': 'discarded'}
    assert evaluate.same_identity(old, current)
    assert old['benchmarks'] == ['livecodebench', 'mbppplus', 'humanevalplus']
    for modified in [
        dict(current, sampling={'samples': 4}),
        dict(current, data={'livecodebench': {'hash': 'changed'}, 'humanevalplus': {'hash': 'b'}}),
        dict(current, model={'path': 'model', 'sha256': {'weights': 'changed'}}),
        dict(current, prompt_encoding='changed'),
    ]:
        assert not evaluate.same_identity(old, modified)
