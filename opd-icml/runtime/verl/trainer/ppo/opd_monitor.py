"""Read-only OPD train/inference consistency on the actual sampled tokens."""
import torch


@torch.no_grad()
def train_inference_consistency(train_logp, rollout_logp, response_mask):
    if train_logp.shape != rollout_logp.shape or train_logp.shape != response_mask.shape:
        raise ValueError("train/inference log-probability shapes do not match")
    valid = response_mask.bool()
    train = train_logp.detach().float()[valid]
    rollout = rollout_logp.detach().to(train_logp.device).float()[valid]
    finite = torch.isfinite(train) & torch.isfinite(rollout)
    result = {"consistency/valid_tokens": float(valid.sum()),
              "consistency/finite_fraction": float(finite.float().mean()) if finite.numel() else 0.,
              "consistency/available": float(bool(finite.any()))}
    if not finite.any():
        return result  # Missing/unusable observations must never masquerade as zero error.
    delta = train[finite] - rollout[finite]
    bounded = delta.clamp(-20., 20.)
    result.update({
        "consistency/logprob_abs_mean": float(delta.abs().mean()),
        "consistency/logprob_abs_p95": float(torch.quantile(delta.abs(), .95)),
        "consistency/logprob_rmse": float(delta.square().mean().sqrt()),
        "consistency/sampled_kl_rollout_to_train_k1": float(-delta.mean()),
        "consistency/sampled_kl_rollout_to_train_k3": float((bounded.expm1() - bounded).mean()),
        "consistency/k3_clamped_fraction": float((delta != bounded).float().mean()),
    })
    return result
