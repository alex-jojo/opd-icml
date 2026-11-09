"""Eurus code monitoring: all-tests pass rate, independent of the OPD loss."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import signal


CODE_SOURCES = {'apps', 'codecontests', 'codeforces', 'taco'}


def extract_code(completion):
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', completion, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip(), True
    blocks = re.findall(r'```\s*\n(.*?)```', completion, re.DOTALL)
    if blocks:
        return blocks[-1].strip(), True
    return completion.strip(), False


def test_spec(ground_truth):
    tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(tests, dict):
        raise ValueError('Code ground truth must contain a test-case object')
    inputs, outputs = tests.get('inputs'), tests.get('outputs')
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        raise ValueError('Code tests require nonempty aligned inputs/outputs')
    return tests


def _local_results(code, tests, timeout):
    if not hasattr(signal, 'alarm'):
        raise RuntimeError('The local PRIME code judge requires Linux/Unix; use a Sandbox Fusion URL on Windows')
    from verl.utils.reward_score.prime_code.utils import check_correctness
    results, _metadata = check_correctness(in_outs=tests, generation=code, timeout=timeout, debug=False)
    return results


def compute_score(data_source, solution_str, ground_truth, extra_info=None, *, backend='local',
                  sandbox_url=None, timeout=5, memory_limit_mb=1024):
    if data_source not in CODE_SOURCES:
        raise ValueError(f'Unexpected code data source: {data_source}')
    tests = test_spec(ground_truth)
    code, fenced = extract_code(solution_str)
    count = len(tests['inputs'])
    if backend == 'sandbox':
        if not sandbox_url:
            raise ValueError('Sandbox code scoring requires a Sandbox Fusion /run_code URL')
        from verl.utils.reward_score.sandbox_fusion.utils import check_correctness
        results, metadata = check_correctness(
            sandbox_fusion_url=sandbox_url, concurrent_semaphore=None, memory_limit_mb=memory_limit_mb,
            in_outs=tests, generation=code, timeout=timeout)
        if any(result == -1 for result in results):
            raise RuntimeError('Code judge service failed; refusing to report infrastructure failures as incorrect solutions')
    elif backend == 'local':
        results = _local_results(code, tests, timeout)
    else:
        raise ValueError(f'Unknown code judge backend: {backend}')
    passed = sum(bool(result == True) for result in results)  # noqa: E712 -- supports numpy.bool_.
    success = len(results) == count and passed == count
    runtime_errors = (-2, -3) if backend == 'sandbox' else (-1,)
    compile_errors = (-4,) if backend == 'sandbox' else (-2,)
    return dict(score=float(success), acc=float(success), code_format=float(fenced),
                code_test_count=count, code_tests_passed=passed,
                code_tests_executed=len(results), code_timeout_or_runtime_error=float(any(result in runtime_errors for result in results)),
                code_compile_error=float(any(result in compile_errors for result in results)))


def compute_scores(data_sources, solution_strs, ground_truths, extra_infos=None, *, workers=8, **kwargs):
    workers = int(workers)
    if not (len(data_sources) == len(solution_strs) == len(ground_truths)):
        raise ValueError('Code scorer batch arrays have different lengths')
    if workers < 1:
        raise ValueError('Code scorer workers must be positive')
    extras = extra_infos if extra_infos is not None else [None] * len(data_sources)
    if len(extras) != len(data_sources):
        raise ValueError('Code scorer extra-info batch has a different length')
    rows = list(zip(data_sources, solution_strs, ground_truths, extras))
    def score(row):
        return compute_score(*row, **kwargs)
    if workers == 1:
        return [score(row) for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score, rows))
