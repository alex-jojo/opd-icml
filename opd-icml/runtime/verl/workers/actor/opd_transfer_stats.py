"""Answer-equal transfer statistics. Thresholds fixed before training, not fitted."""
import math
import torch
from .opd_top16 import reverse_kl


def answer_means(values,counts):
    if sum(counts)!=len(values) or any(n<=0 for n in counts):
        raise ValueError('response counts must partition nonempty valid tokens')
    return torch.stack([x.double().mean() for x in values.split(counts)])


def row_losses(logits,teacher,counts):
    return answer_means(reverse_kl(logits,teacher),counts)


def lower_tail(values,fraction=.2):
    return float(values.sort().values[:max(1,math.ceil(len(values)*fraction))].mean())


def statistics(before,after,baseline_after=None,absolute=.0001,relative=.001,need=.01):
    before=torch.as_tensor(before,dtype=torch.float64); after=torch.as_tensor(after,dtype=torch.float64)
    if before.shape!=after.shape or before.ndim!=1 or not len(before) or not torch.isfinite(before+after).all():
        raise ValueError('invalid per-answer transfer values')
    gain=before-after; threshold=absolute+relative*before.abs()
    eligible=before>need
    out=dict(before=before.tolist(),after=after.tolist(),gain=gain.tolist(),
        threshold=threshold.tolist(),eligible=eligible.tolist(),mean_gain=float(gain.mean()),
        coverage=float((gain>threshold).double().mean()),
        negative_transfer=float((gain < -threshold).double().mean()),
        lower_tail_gain=lower_tail(gain),worst_gain=float(gain.min()),
        eligible_count=int(eligible.sum()),
        eligible_coverage=float((gain[eligible]>threshold[eligible]).double().mean()) if eligible.any() else None,
        positive_mass=float(gain.clamp_min(0).mean()),negative_mass=float((-gain).clamp_min(0).mean()))
    if baseline_after is not None:
        base=torch.as_tensor(baseline_after,dtype=torch.float64); extra=base-after
        base_gain=before-base
        out.update(extra_gain=extra.tolist(),mean_extra_gain=float(extra.mean()),
            lower_tail_extra=lower_tail(extra),robust_extra=float(extra.mean())+.5*lower_tail(extra),
            extra_improved_fraction=float((extra>threshold).double().mean()),
            extra_harmed_fraction=float((extra < -threshold).double().mean()),
            coverage_lift=float(((gain>threshold).double()-(base_gain>threshold).double()).mean()),
            negative_transfer_lift=float(((gain < -threshold).double()-(base_gain < -threshold).double()).mean()))
    return out
