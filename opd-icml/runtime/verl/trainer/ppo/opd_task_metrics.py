"""Task labels and compact code-test references for OPD logs."""
import hashlib
import json


def code_ground_truth_reference(ground_truth):
    if isinstance(ground_truth, dict) and 'code_tests_sha256' in ground_truth:
        return ground_truth
    tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    raw = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth, sort_keys=True)
    return dict(code_tests_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                test_count=len(tests['inputs']), fn_name=tests.get('fn_name'),
                location='source training/validation parquet reward_model.ground_truth')


def code_accuracy_metrics(accuracy, extras):
    metrics = {'monitor/train_code_accuracy': float(accuracy)}
    for source, target in [('code_format', 'code_format_rate'),
                           ('code_compile_error', 'code_compile_error_rate'),
                           ('code_timeout_or_runtime_error', 'code_timeout_or_runtime_error_rate')]:
        values = extras.get(source)
        if values is not None and len(values):
            metrics['monitor/' + target] = float(sum(values) / len(values))
    return metrics
