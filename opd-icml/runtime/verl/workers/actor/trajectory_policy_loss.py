"""Single-rollout signed policy loss; no advantage centering or PPO ratio."""
import torch


def signed_token_policy_loss(log_prob, advantage, response_mask):
    mask=response_mask.bool()
    selected=torch.where(mask,log_prob,torch.zeros_like(log_prob))
    weights=torch.where(mask,advantage.detach(),torch.zeros_like(advantage))
    return -(selected*weights).sum()/mask.sum().clamp_min(1)


def microbatch_token_weight(response_mask, total_tokens, world_size=1):
    # Convert to float so logging never receives a CUDA Tensor as a metric.
    return float((world_size*response_mask.sum().detach()/total_tokens.clamp_min(1)).item())
