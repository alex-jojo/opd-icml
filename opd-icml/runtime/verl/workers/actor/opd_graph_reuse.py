"""Bounded retained-graph backwards, including the FSDP1 hook lifecycle.

FSDP1 removes AccumulateGrad hooks and clears the unshard flag after each
backward. Re-arm ONLY handles participating in this graph, using the original
AccumulateGrad object (important with mixed precision). Reductions, resharding
and cleanup still run through PyTorch's normal post-backward implementation.

Lifecycle checked against PyTorch 2.6.0 and 2.7.1 _runtime_utils.py:
https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/fsdp/_runtime_utils.py
"""
from functools import partial
import torch


def reuse_fallback_reason(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if any(type(m).__name__ == 'OptimizedModule' for m in model.modules()):
        return 'compiled model graph reuse is not supported'
    wrappers = [m for m in model.modules() if isinstance(m, FSDP)]
    if wrappers:
        # Private FSDP1 lifecycle: audit new releases before enabling them.
        version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if version not in {(2, 6), (2, 7)}:
            return 'FSDP graph reuse requires the audited PyTorch 2.6/2.7 lifecycle'
        if any(m._use_orig_params or m.cpu_offload.offload_params for m in wrappers):
            return 'FSDP graph reuse requires use_orig_params=false and no parameter CPU offload'
        if any(m._handle is not None and not m._handle.flat_param.requires_grad for m in wrappers):
            return 'FSDP graph reuse requires trainable flat parameters'
    # The project's activation offloader consumes saved tensors on first use.
    if any(getattr(m, '_verl_activation_offloading', False) for m in model.modules()):
        return 'activation offloading consumes saved tensors on the first backward'
    return None


class RetainedGraph:
    """Capture after forward, before the first backward; release after the group."""
    def __init__(self, model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        self.handles = []
        for state in model.modules():
            if not isinstance(state, FSDP) or state._handle is None:
                continue
            handle = state._handle
            hook_state = getattr(handle.flat_param, '_post_backward_hook_state', None)
            if hook_state is None and handle._needs_pre_backward_unshard and handle.flat_param.requires_grad:
                raise RuntimeError('Missing FSDP retained-graph hooks; use graph_group_size=1')
            if hook_state is not None:
                if len(hook_state) != 2 or not handle._needs_pre_backward_unshard:
                    raise RuntimeError('Unsupported FSDP retained-graph state; use graph_group_size=1')
                self.handles.append((state, handle, hook_state[0]))

    def prepare_next_backward(self):
        from torch.distributed.fsdp._runtime_utils import _post_backward_hook
        for state, handle, acc_grad in self.handles:
            parameter = handle.flat_param
            if hasattr(parameter, '_post_backward_hook_state') or handle._ran_pre_backward_hook:
                raise RuntimeError('Previous FSDP backward did not finish cleanup')
            # The output's existing pre-backward hook will unshard on demand.
            # No prefetch order remains after finalization; do not fabricate one.
            handle._needs_pre_backward_unshard = True
            hook = acc_grad.register_hook(partial(_post_backward_hook, state, handle))
            parameter._post_backward_hook_state = (acc_grad, hook)
