"""Shared data identities, prompts and pass@4 checks for code benchmarks."""
import base64
from datetime import date
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import re
import zlib

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ('livecodebench', 'mbppplus', 'humanevalplus')
PROTOCOL = 'official-code-pass4-v1'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def lockfile():
    return json.loads((ROOT / 'configs/code_benchmarks.json').read_text(encoding='utf-8'))


def read_manifest(data_dir, benchmark):
    directory = Path(data_dir) / benchmark
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != PROTOCOL or manifest['benchmark'] != benchmark:
        raise ValueError('Code benchmark protocol mismatch')
    for name, expected in manifest['files'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f'Benchmark file changed: {path}')
    return manifest


def private_tests(value):
    """Decode LCB's JSON or pickled JSON STRING, never arbitrary pickle objects."""
    try:
        tests = json.loads(value)
    except json.JSONDecodeError:
        payload = zlib.decompress(base64.b64decode(value, validate=True))
        allowed = {'PROTO', 'FRAME', 'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8',
                   'MEMOIZE', 'BINPUT', 'LONG_BINPUT', 'STOP'}
        if any(op.name not in allowed for op, _arg, _pos in pickletools.genops(payload)):
            raise ValueError('LCB private-test pickle must contain only a JSON string')
        text = pickle.loads(payload)
        if not isinstance(text, str):
            raise ValueError('LCB private tests did not decode to a JSON string')
        tests = json.loads(text)
    if not isinstance(tests, list):
        raise ValueError('LCB private tests must be a list')
    return tests


def lcb_sample(row):
    public = json.loads(row['public_test_cases'])
    private = private_tests(row['private_test_cases'])
    tests = public + private
    if not tests or any(not {'input', 'output', 'testtype'}.issubset(t) for t in tests):
        raise ValueError('Missing LiveCodeBench tests')
    metadata = json.loads(row['metadata'])
    return dict(input_output=json.dumps(dict(inputs=[t['input'] for t in tests],
                    outputs=[t['output'] for t in tests], fn_name=metadata.get('func_name'))),
                public_tests=len(public), private_tests=len(private))


def date_selected(value, start=None, end=None):
    current = date.fromisoformat(value[:10])
    return (start is None or current >= date.fromisoformat(start)) and (end is None or current <= date.fromisoformat(end))


def prompt_for(question):
    if question['benchmark'] == 'livecodebench':
        prefix = 'Solve the following programming problem in Python. '
        if question.get('starter_code'):
            prefix += 'Use the provided starter-code interface. '
        else:
            prefix += 'Read from standard input and write to standard output. '
        text = prefix + 'Return the complete solution in a final ```python code block.\n\n' + question['prompt']
        if question.get('starter_code'):
            text += '\n\nStarter code:\n```python\n' + question['starter_code'] + '\n```'
        return text
    return ('Implement the following Python programming task. Return the complete Python solution, including '
            'the required function and imports, in a final ```python code block.\n\n' + question['prompt'])


def extract_solution(response, question):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else response.strip()
    if question['benchmark'] == 'humanevalplus':
        entry = question['entry_point']
        if not re.search(r'\b(?:async\s+)?def\s+' + re.escape(entry) + r'\s*\(', code):
            # Preserve indentation in a completion-only reply to a HumanEval stub.
            body = blocks[-1] if blocks else response
            if body.startswith('\n'):
                body = body[1:]
            if body and body[0].isspace():
                return question['prompt'] + body
    return code


def validate_generations(questions, generations, samples):
    if samples < 4 or len(generations) != len(questions):
        raise ValueError('pass@4 requires >=4 samples for every question; task count mismatch')
    for question, generation in zip(questions, generations):
        expected = hashlib.sha256(prompt_for(question).encode()).hexdigest()
        if generation['task_id'] != question['task_id'] or generation['prompt_sha256'] != expected:
            raise ValueError('Generation task order/prompt identity mismatch')
        if len(generation['responses']) != samples or not all(isinstance(x, str) for x in generation['responses']):
            raise ValueError('Incomplete code generations')


def summarize_pass4(graded):
    """The same unbiased estimator as EvalPlus/LCB, fixed k=4."""
    if not graded:
        raise ValueError('Cannot summarize an empty benchmark')
    values = []
    for grades in graded:
        n, c = len(grades), sum(grades)
        if n < 4 or any(type(x) is not bool for x in grades):
            raise ValueError('pass@4 requires at least four boolean grades per task')
        probability = 1.
        if n - c < 4:
            probability = 0.
        else:
            for i in range(4):
                probability *= (n - c - i) / (n - i)
        values.append(1. - probability)
    return dict(metric='pass@4', **{'pass@4': sum(values) / len(values)},
                problems=len(graded), samples=sum(map(len, graded)),
                samples_per_problem=sorted(set(map(len, graded))))


def evalplus_grades(raw, task_ids, samples):
    if set(raw['eval']) != set(task_ids):
        raise ValueError('EvalPlus result task IDs do not match the requested benchmark')
    grades = []
    for task_id in task_ids:
        results = raw['eval'][task_id]
        if len(results) != samples:
            raise ValueError('EvalPlus returned an incomplete sample count')
        if any('base_status' not in r or 'plus_status' not in r for r in results):
            raise ValueError('EvalPlus results must include both base and plus tests')
        grades.append([r['base_status'] == r['plus_status'] == 'pass' for r in results])
    return grades
