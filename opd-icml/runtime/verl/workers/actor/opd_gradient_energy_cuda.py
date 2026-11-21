"""FP64 sum of squared dense gradient elements for NVIDIA training workers."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    fused_squared_sum = None
else:
    @triton.jit
    def _squared_sums(values, partials, length, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        grad = tl.load(values + offsets, mask=offsets < length, other=0.).to(tl.float64)
        tl.store(partials + block, tl.sum(grad * grad, axis=0))

    def fused_squared_sum(values):
        # Only one FP64 partial per block reaches global memory.
        block = 4096
        blocks = triton.cdiv(values.numel(), block)
        partials = torch.empty(blocks, dtype=torch.float64, device=values.device)
        with torch.cuda.device(values.device):
            _squared_sums[(blocks,)](values, partials, values.numel(), BLOCK=block)
        return partials.sum()
