"""Bounded selective activation caching for single-backward diagnostics.

Keep checkpointing enabled. Reuse expensive matrix-multiply outputs during
recomputation, using PyTorch's public selective-checkpoint API. Training and
retained graphs with multiple backwards keep their existing implementation.
"""
from contextlib import contextmanager
from functools import partial
import math
import os
import torch
import torch.utils.checkpoint as checkpoint_api


def _cache_budget(model):
    """Reserve gradient allocation and workspace, then use <=1/4 of the rest."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        return 0
    parameters = list(model.parameters())
    if not parameters or parameters[0].device.type != 'cuda':
        return 0
    if any(type(m).__name__ == 'OptimizedModule' or
           getattr(m, '_verl_activation_offloading', False) for m in model.modules()):
        return 0
    limit = float(os.environ.get('OPD_DIAGNOSTIC_ACTIVATION_CACHE_GIB', '4'))
    if not math.isfinite(limit) or limit < 0:
        raise ValueError('OPD_DIAGNOSTIC_ACTIVATION_CACHE_GIB must be finite and nonnegative')
    if not limit:
        return 0
    # mem_get_info excludes allocator-reserved memory: deliberately conservative.
    free, _ = torch.cuda.mem_get_info(parameters[0].device)
    missing_gradients = sum(p.numel() * max(4, p.element_size())
                            for p in parameters if p.requires_grad and p.grad is None)
    available = max(0, free - missing_gradients - 2 * 1024**3)
    return min(int(limit * 1024**3), available // 4)


def _matrix_bytes(op, args):
    if op == torch.ops.aten.mm.default:
        a, b = args[:2]
    elif op == torch.ops.aten.addmm.default:
        a, b = args[1:3]
    else:
        return 0
    # These ATen operations produce a fresh, dense matrix. Autocast has already
    # selected the operands' dtype before reaching the dispatch policy.
    return a.shape[0] * b.shape[1] * a.element_size()


@contextmanager
def diagnostic_checkpoint_cache(model, *, single_backward=True):
    stats = dict(budget_bytes=0, cached_bytes=0, cached_matmuls=0, reused_matmuls=0)
    originals = []
    if single_backward and hasattr(checkpoint_api, 'create_selective_checkpoint_contexts'):
        for module in model.modules():
            # FSDP proxies attributes of its wrapped module. Patch only the
            # actual owner, without leaving a new shadow attribute on FSDP.
            original = module.__dict__.get('_gradient_checkpointing_func')
            if (getattr(module, 'gradient_checkpointing', False)
                    and isinstance(original, partial) and original.func is checkpoint_api.checkpoint
                    and original.keywords.get('use_reentrant') is False
                    and not original.keywords.get('debug', False)
                    and 'context_fn' not in original.keywords):
                originals.append((module, original))
    if not originals:
        yield stats
        return
    stats['budget_bytes'] = _cache_budget(model)
    if not stats['budget_bytes']:
        yield stats
        return
    policy_type = checkpoint_api.CheckpointPolicy

    def wrap(original):
        def call(function, *args, **kwargs):
            if 'context_fn' in kwargs or stats['cached_bytes'] >= stats['budget_bytes']:
                return original(function, *args, **kwargs)

            def contexts():
                decisions = []
                cursor = 0

                def policy(ctx, op, *op_args, **op_kwargs):
                    nonlocal cursor
                    size = _matrix_bytes(op, op_args)
                    if not size:
                        return policy_type.PREFER_RECOMPUTE
                    if ctx.is_recompute:
                        expected_op, selected = decisions[cursor]
                        cursor += 1
                        if expected_op != op:
                            raise RuntimeError('Diagnostic checkpoint operation order changed')
                        stats['reused_matmuls'] += int(selected)
                    else:
                        selected = stats['cached_bytes'] + size <= stats['budget_bytes']
                        decisions.append((op, selected))
                        if selected:
                            stats['cached_bytes'] += size
                            stats['cached_matmuls'] += 1
                    return policy_type.MUST_SAVE if selected else policy_type.PREFER_RECOMPUTE

                return checkpoint_api.create_selective_checkpoint_contexts(policy)

            return original(function, *args, context_fn=contexts, **kwargs)
        return call

    try:
        for module, original in originals:
            module._gradient_checkpointing_func = wrap(original)
        yield stats
    finally:
        for module, original in originals:
            module._gradient_checkpointing_func = original
