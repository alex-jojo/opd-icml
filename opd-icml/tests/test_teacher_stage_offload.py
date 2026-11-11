"""Check actual worker residency context without importing GPU/Ray workers."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def residency(events, enabled=True, world=4, version=1, direction=False):
    source = ast.parse((ROOT/'runtime/verl/workers/fsdp_workers.py').read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == '_opd_teacher_on_device')
    namespace = dict(contextmanager=contextmanager,
        load_fsdp_model_to_gpu=lambda model: events.append('load'),
        offload_fsdp_model_to_cpu=lambda model: events.append('offload'),
        fsdp_version=lambda model: version)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'worker_context', 'exec'), namespace)
    model = SimpleNamespace(_handle=SimpleNamespace(reshard=lambda flag: events.append('reshard')),
                            reshard=lambda: events.append('reshard'))
    worker = SimpleNamespace(world_size=world, ref_module_fsdp=model,
        config=SimpleNamespace(ref={'opd_stage_offload': enabled},
            actor=SimpleNamespace(policy_loss={'loss_mode': 'opd_top16', 'direction_reward': {'enabled': direction}})))
    return namespace['_opd_teacher_on_device'](worker)


@pytest.mark.parametrize('world,version', [(1, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize('fails', [False, True])
def test_query_stage_loads_once_and_releases_on_success_or_error(world, version, fails):
    events = []
    try:
        with residency(events, world=world, version=version):
            events.append('query')
            if fails:
                raise RuntimeError('query failed')
    except RuntimeError:
        assert fails
    assert events == ['load', 'query'] + (['reshard'] if world > 1 else []) + ['offload']


@pytest.mark.parametrize('enabled,direction', [(False, False), (True, True)])
def test_existing_resident_or_direction_teacher_does_not_move(enabled, direction):
    events = []
    with residency(events, enabled=enabled, direction=direction):
        events.append('query')
    assert events == ['query', 'reshard']


def test_actual_reference_query_uses_residency_context():
    source = ast.parse((ROOT/'runtime/verl/workers/fsdp_workers.py').read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'compute_ref_log_prob')
    context = next(n for n in ast.walk(method) if isinstance(n, ast.With)
                   and '_opd_teacher_on_device' in ast.unparse(n.items[0].context_expr))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'compute_log_prob' for n in ast.walk(context))
