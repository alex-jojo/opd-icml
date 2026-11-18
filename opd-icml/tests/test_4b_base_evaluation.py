from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate_4b_base as base
import disk_space


def test_paper_questions_match_all_five_benchmarks(tmp_path):
    rows = base.paper_questions(ROOT/'data/benchmarks.jsonl')
    assert Counter(row['benchmark'] for row in rows) == {
        'math500': 500, 'hmmt25_feb': 30, 'hmmt26_feb': 33, 'aime26': 30, 'minerva': 272}
    damaged = tmp_path/'damaged.jsonl'
    base.evaluate.save_rows(damaged, rows[1:] + [rows[1]])
    with pytest.raises(ValueError, match='distinct'):
        base.paper_questions(damaged)


def test_base_generates_once_and_grades_same_samples_on_retry(tmp_path, monkeypatch):
    counts = dict.fromkeys(base.QUESTIONS, 1)
    counts['math500'] = 2  # A pooled average would differ from the five-task macro average.
    monkeypatch.setattr(base, 'QUESTIONS', counts)
    source = tmp_path/'questions-source.jsonl'
    inputs = [dict(benchmark=b, benchmark_index=i, problem=f'{b} problem {i}', answer='2')
              for b, size in counts.items() for i in range(size)]
    base.evaluate.save_rows(source, inputs + [dict(benchmark='amc23')])
    output = tmp_path/'output'
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    (snapshot/'config.json').write_text('{}')

    def download(spec, cache_dir):
        assert spec == base.read_models('4b')['student'] and spec['repo'] == 'Qwen/Qwen3-4B-Base'
        return snapshot

    def generate(args, questions):
        assert questions == inputs and args.model_path == str(snapshot)
        assert (args.n, args.seed, args.temperature, args.top_p, args.max_tokens) == (8, 42, 1., 1., 8192)
        answers = {'math500': r'\boxed{2}', 'minerva': 'Final answer: 2'}
        return [dict(q, responses=[answers.get(q['benchmark'], 'no answer')]*4 + ['no answer']*4,
                     response_lengths=[4]*8) for q in questions]

    # Exercise the evaluator's real grading, journals and aggregation with an integer-only verifier.
    strict_scorer = base.evaluate.score_row
    monkeypatch.setattr(base.evaluate, 'score_row', lambda row: strict_scorer(
        row, str.__eq__, base.evaluate.strict_last_box, lambda value: value))
    monkeypatch.setattr(base, 'base_snapshot', download)
    monkeypatch.setattr(base.evaluate, 'generate', generate)
    args = ['--math-benchmarks', str(source), '--output', str(output), '--score-workers', '1']
    base.main(args)
    expected = base.read(output/'summary.json')
    assert expected['questions'] == 6 and expected['samples'] == 48
    assert base.read(output/'responses.meta.json')['model_sha256'] == {'config.json': base.evaluate.digest(snapshot/'config.json')}
    assert expected['optimizer_updates'] == 0 and expected['model'] == base.read_models('4b')['student']
    strict, lenient = expected['strict_last_box'], expected['lenient_answers_v1']
    assert strict['five_benchmark_mean_at_8'] == pytest.approx(.1)
    assert lenient['five_benchmark_mean_at_8'] == pytest.approx(.2)
    with gzip.open(output/expected['responses_file'], 'rt', encoding='utf-8') as stream:
        generations = [json.loads(line) for line in stream]
    scores = output/expected['scores_file']
    entries = base.evaluate.read_rows(scores)
    assert len(entries) == len(generations) == 6
    for item, q in zip(entries, generations):
        assert item['generation_sha256'] == expected['responses_sha256']
        assert q['benchmark_index'] == item['row']['benchmark_index'] and len(q['responses']) == 8
    minerva = next(item['row'] for item in entries if item['row']['benchmark'] == 'minerva')
    assert sum(minerva['strict_last_box_acc_list'])/8 == strict['mean_at_8']['minerva'] == 0.
    assert sum(minerva['acc_list'])/8 == lenient['mean_at_8']['minerva'] == .5
    assert all(b'Final answer: 2' not in f.read_bytes() for f in output.iterdir() if f.suffix != '.gz')
    assert expected['responses_bytes'] == (output/expected['responses_file']).stat().st_size
    assert expected['scores_bytes'] == scores.stat().st_size
    monkeypatch.setattr(base.evaluate, 'generate', lambda *args: pytest.fail('generated answers twice'))
    monkeypatch.setattr(base, 'base_snapshot', lambda *args: pytest.fail('accessed model to regrade'))
    # Recover a published receipt awaiting its final rename and a partial score journal.
    (output/'responses.jsonl.gz').rename(output/'responses.jsonl.gz.tmp')
    scores.write_bytes(b''.join(scores.read_bytes().splitlines(keepends=True)[:2]) + b'{"cut":')
    base.main(args)
    assert base.read(output/'summary.json') == expected
    with (output/'responses.jsonl.gz').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        base.main(args)


def test_no_model_download_or_output_when_disk_is_low(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda path: SimpleNamespace(free=123))
    monkeypatch.setenv('OPD_MIN_FREE_GIB', '1')
    monkeypatch.setattr(base, 'base_snapshot', lambda *args: pytest.fail('downloaded with low disk space'))
    with pytest.raises(OSError, match='reserve'):
        base.main(['--output', str(tmp_path/'output')])
    assert not (tmp_path/'output').exists()


def test_cold_snapshot_checks_missing_bytes_on_cache_volume(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    spec = base.read_models('4b')['student']
    cache = tmp_path/'cache'
    cache.mkdir()
    def offline(*args, **kwargs):
        raise LocalEntryNotFoundError('not cached')
    fake_info = SimpleNamespace(sha=spec['revision'], siblings=[
        SimpleNamespace(rfilename='model.safetensors', size=500),
        SimpleNamespace(rfilename='config.json', size=10)])
    monkeypatch.setattr(hub, 'snapshot_download', offline)
    monkeypatch.setattr(hub.HfApi, 'model_info', lambda *args, **kwargs: fake_info)
    monkeypatch.setattr(hub, 'try_to_load_from_cache', lambda repo, name, **kw: '/cached' if name == 'config.json' else None)
    monkeypatch.setattr(hub, 'hf_hub_download', lambda *args, **kwargs: pytest.fail('downloaded to full cache'))
    def free(path):
        assert path == cache
        return SimpleNamespace(free=disk_space.GIB+499)
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', free)
    monkeypatch.setenv('OPD_MIN_FREE_GIB', '1')
    with pytest.raises(OSError, match='reserve'):
        base.base_snapshot(spec, cache)


def test_snapshot_reuses_complete_cache_and_fetches_only_missing_shard(tmp_path, monkeypatch):
    import huggingface_hub as hub
    spec = base.read_models('4b')['student']
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    names = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
    for name in names:
        (snapshot/name).write_text('{}')
    index = snapshot/'model.safetensors.index.json'
    index.write_text(json.dumps(dict(weight_map={'w': 'weights.safetensors'})))
    names += [index.name, 'weights.safetensors']
    requests = []
    def manifest(*args, **kwargs):
        requests.append('manifest')
        return SimpleNamespace(sha=spec['revision'], siblings=[SimpleNamespace(rfilename=n, size=100) for n in names])
    def download(repo, name, **kwargs):
        requests.append(name)
        (snapshot/name).write_bytes(b'weights')
    monkeypatch.setattr(hub, 'snapshot_download', lambda repo, **kw: str(snapshot))
    monkeypatch.setattr(hub.HfApi, 'model_info', manifest)
    monkeypatch.setattr(hub, 'try_to_load_from_cache', lambda repo, name, **kw: str(snapshot/name) if (snapshot/name).exists() else None)
    monkeypatch.setattr(hub, 'hf_hub_download', download)
    assert base.base_snapshot(spec, tmp_path) == snapshot
    assert base.base_snapshot(spec, tmp_path) == snapshot
    assert requests == ['manifest', 'weights.safetensors']


def test_old_generations_are_verified_and_reused_without_another_text_copy(tmp_path, monkeypatch):
    rows = [dict(benchmark='math500', benchmark_index=0, problem='1+1?', answer='2')]
    original_questions = tmp_path/'questions.jsonl'
    base.evaluate.save_rows(original_questions, rows)
    raw = tmp_path/'mean8.generations.jsonl'
    old_rows = [dict(rows[0], responses=['2']*8, response_lengths=[1]*8)]
    base.evaluate.save_rows(raw, old_rows)
    model = tmp_path/'snapshot'
    model.mkdir()
    config = model/'config.json'
    config.write_text('{}')
    identity = dict(model=base.read_models('4b')['student'], dataset_sha256='portable-hash',
                    sampling={'n': 8}, prompt_encoding=base.evaluate.CHAT_PROMPT_ENCODING)
    previous = dict(schema=1, dataset_sha256=base.evaluate.digest(original_questions), sampling=identity['sampling'],
                    prompt_encoding=identity['prompt_encoding'], model_sha256={'config.json': base.evaluate.digest(config)})
    checksum = base.evaluate.digest(raw)
    base.evaluate.save_json(raw.with_suffix('.meta.json'), dict(identity=previous, sha256=checksum, rows=1))
    monkeypatch.setattr(base, 'base_snapshot', lambda *args: model)
    monkeypatch.setattr(base.evaluate, 'generate', lambda *args: pytest.fail('regenerated old answers'))
    assert base.generation_rows(tmp_path, rows, identity, SimpleNamespace(n=8), None) == (old_rows, raw, checksum)
    assert not (tmp_path/'responses.jsonl.gz').exists()
    config.write_text('changed')
    with pytest.raises(ValueError, match='identity changed'):
        base.generation_rows(tmp_path, rows, identity, SimpleNamespace(n=8), None)


def test_failed_compression_removes_its_partial_file(tmp_path, monkeypatch):
    rows = [dict(benchmark='math500', benchmark_index=0, responses=['abc'] * 8)]
    monkeypatch.setattr(base, 'base_snapshot', lambda *args: tmp_path)
    monkeypatch.setattr(base.evaluate, 'generate', lambda *args: rows)
    def fail_write(*args):
        raise OSError('low space')
    monkeypatch.setattr(base.SpaceCheckedWriter, 'write', fail_write)
    with pytest.raises(OSError, match='low space'):
        base.generation_rows(tmp_path, rows, {'model': {}}, SimpleNamespace(n=8), None)
    assert not list(tmp_path.iterdir())
