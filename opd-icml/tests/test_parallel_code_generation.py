"""CPU multi-process tests of the GPU-isolated scheduler and checkpoint merge."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import code_benchmarks as cb
import parallel_code_generation as parallel
import watch_code_progress as watch


def question(index):
    return dict(benchmark='humanevalplus', task_id=f'HumanEval/{index}',
                prompt=f'def answer_{index}():\n    """Return {index}."""\n', entry_point=f'answer_{index}')


def generation(q):
    return dict(task_id=q['task_id'], responses=['    return 1\n']*4, prompt_style='completion',
        prompt_sha256=hashlib.sha256(cb.prompt_for(q, 'completion').encode()).hexdigest())


def test_partition_merge_preserves_old_answers_and_all_questions(tmp_path):
    questions = [question(i) for i in range(11)]
    prefix = [generation(q) for q in questions[:3]]
    cb.save_jsonl(tmp_path / 'generation.partial.jsonl', prefix)
    original_bytes = (tmp_path / 'generation.partial.jsonl').read_bytes()
    plan, saved = parallel.prepare_plan(tmp_path, questions, 4, 4)
    assert saved == prefix and plan['reused_questions'] == 3
    seen = []
    for index in range(4):
        subset = parallel.shard_questions(tmp_path, questions, 4, 4, index)
        seen.extend(q['task_id'] for q in subset)
        cb.save_jsonl(parallel.shard_directory(tmp_path, 4, index)/'generations.jsonl', [generation(q) for q in subset])
    assert sorted(seen) == sorted(q['task_id'] for q in questions[3:])
    merged = parallel.merge_shards(tmp_path, questions, 4, 4)
    assert merged[:3] == prefix and [row['task_id'] for row in merged] == [q['task_id'] for q in questions]
    assert (tmp_path / 'generation.partial.jsonl').read_bytes() == original_bytes
    cb.validate_generations(questions, merged, 4)


def test_plan_rejects_changed_gpu_count_population_and_serial_writer(tmp_path):
    questions = [question(i) for i in range(4)]
    parallel.prepare_plan(tmp_path, questions, 4, 4)
    with pytest.raises(ValueError, match='plan changed'):
        parallel.prepare_plan(tmp_path, questions, 4, 2)
    with pytest.raises(ValueError, match='identity mismatch'):
        parallel.shard_questions(tmp_path, [dict(questions[0], prompt='different'), *questions[1:]], 4, 4, 0)
    cb.save_jsonl(tmp_path / 'generation.partial.jsonl', [generation(questions[0])])
    with pytest.raises(ValueError, match='Serial answers changed'):
        parallel.prepare_plan(tmp_path, questions, 4, 4)


def test_merge_does_not_publish_incomplete_answers(tmp_path):
    questions = [question(i) for i in range(5)]
    parallel.prepare_plan(tmp_path, questions, 4, 4)
    with pytest.raises(ValueError, match='task count mismatch'):
        parallel.merge_shards(tmp_path, questions, 4, 4)
    assert not (tmp_path / 'generations.jsonl').exists()


def test_live_progress_counts_completed_shards_without_double_counting(tmp_path):
    cb.save_json(tmp_path / 'run.json', dict(benchmarks=['humanevalplus'], data={'humanevalplus': {'rows': 6}}))
    folder = tmp_path / 'humanevalplus'
    questions = [question(i) for i in range(6)]
    cb.save_jsonl(folder / 'generation.partial.jsonl', [generation(questions[0])])
    parallel.prepare_plan(folder, questions, 4, 4)
    cb.save_jsonl(parallel.shard_directory(folder, 4, 1)/'generation.partial.jsonl', [generation(questions[1])])
    cb.save_jsonl(parallel.shard_directory(folder, 4, 2)/'generations.jsonl', [generation(questions[2])])
    state = watch.EvaluationProgress(tmp_path)
    assert state.snapshot()['humanevalplus']['generated'] == 3
    cb.save_jsonl(folder / 'generations.jsonl', [generation(q) for q in questions])
    assert state.snapshot()['humanevalplus']['generated'] == 6


def test_real_subprocesses_isolate_devices_resume_and_merge(tmp_path):
    # This fake generator never imports CUDA and only writes owned fixture answers.
    fake = tmp_path / 'owned_fake_generator.py'
    fake.write_text('''import argparse,sys,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import parallel_code_generation as p
from code_benchmarks import read_jsonl,save_jsonl,validate_generations
parser=argparse.ArgumentParser()
parser.add_argument('scripts')
parser.add_argument('--root')
parser.add_argument('--source')
parser.add_argument('--shard-count',type=int)
parser.add_argument('--shard-index',type=int)
a=parser.parse_args()
questions=list(read_jsonl(a.source))
subset=p.shard_questions(a.root,questions,4,a.shard_count,a.shard_index)
folder=p.shard_directory(a.root,a.shard_count,a.shard_index)
path=folder/'generation.partial.jsonl'
rows=list(read_jsonl(path)) if path.exists() else []
for q in subset[len(rows):]:
 import hashlib
 from code_benchmarks import prompt_for
 rows.append(dict(task_id=q['task_id'],responses=['    return 1\\n']*4,prompt_style='completion',
  device_seen=os.environ['CUDA_VISIBLE_DEVICES'],
  prompt_sha256=hashlib.sha256(prompt_for(q,'completion').encode()).hexdigest()))
save_jsonl(path,rows)
validate_generations(subset,rows,4)
path.replace(folder/'generations.jsonl')
''', encoding='utf8')
    questions = [question(i) for i in range(9)]
    source = tmp_path / 'questions.jsonl'
    cb.save_jsonl(source, questions)
    folder = tmp_path / 'benchmark'
    cb.save_jsonl(folder/'generation.partial.jsonl', [generation(questions[0])])
    parallel.prepare_plan(folder, questions, 4, 4)
    # One shard has a saved partial prefix from an interrupted run.
    first = parallel.shard_questions(folder, questions, 4, 4, 1)[0]
    kept = dict(generation(first), device_seen='kept')
    cb.save_jsonl(parallel.shard_directory(folder, 4, 1)/'generation.partial.jsonl', [kept])
    command = [sys.executable, str(fake), str(ROOT/'scripts'), '--root', str(folder), '--source', str(source)]
    parallel.generate_parallel(command, folder, questions, 4, ['0', '1', '2', '3'])
    saved = list(cb.read_jsonl(folder/'generations.jsonl'))
    cb.validate_generations(questions, saved, 4)
    assert saved[0] == generation(questions[0]) and saved[1] == kept
    for index, row in enumerate(saved[2:], 2):
        assert row['device_seen'] == str(index % 4)
    receipt = json.loads((folder/'generation.execution.json').read_text(encoding='utf8'))
    assert receipt['status'] == 'complete' and receipt['tensor_parallel_per_worker'] == 1
    # Complete answers are reused without starting any process.
    parallel.generate_parallel(['nonexistent-command'], folder, questions, 4, ['0', '1', '2', '3'])


def test_failure_preserves_partial_answers(tmp_path):
    questions = [question(i) for i in range(4)]
    failure = tmp_path / 'fail.py'
    failure.write_text('raise SystemExit(3)')
    folder = tmp_path / 'benchmark'
    with pytest.raises(RuntimeError, match='worker failed'):
        parallel.generate_parallel([sys.executable, str(failure)], folder, questions, 4, ['0','1','2','3'])
    assert not (folder/'generations.jsonl').exists()
    assert (folder/'generation.seed.jsonl').exists()
    assert json.loads((folder/'generation.execution.json').read_text(encoding='utf8'))['status'] == 'interrupted_or_failed'


def test_visible_devices_requires_explicit_unique_mask(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='explicit unique'):
        parallel.visible_devices()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    assert parallel.visible_devices() == ['0','1','2','3']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,0')
    with pytest.raises(ValueError):
        parallel.visible_devices()


def test_data_mode_dry_run_preserves_sampling(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_code.py'), '--model-path', 'unused',
        '--output-dir', str(tmp_path/'run'), '--prompt-style', 'completion', '--parallel-mode', 'data',
        '--batch-size', '8', '--max-num-seqs', '16', '--workers', '32', '--dry-run'],
        check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['parallel_mode'] == 'data' and plan['generation_devices'] == ['0','1','2','3']
    assert plan['sampling']['samples'] == 8 and plan['sampling']['max_tokens'] == 8192
