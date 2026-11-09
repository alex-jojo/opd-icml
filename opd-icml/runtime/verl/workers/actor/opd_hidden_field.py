"""Teacher-free continuation through a frozen local subspace flow.

The asset contains an early OPD hidden-displacement basis plus empirical
arrival-state/velocity anchors.  A kernel field chooses a different direction
for each current hidden state, and midpoint integration accounts for local
curvature.  No token-time feature, teacher, quality scalar, or checkpoint
interpolation is used online.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_CACHE: dict[str, dict[str, torch.Tensor]] = {}
_DEVICE_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def _load(actor) -> dict[str, torch.Tensor]:
    cfg = actor.config.policy_loss.hidden_field
    path = Path(str(cfg.get("asset_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"quoted hidden_field.asset_path is required: {path}")
    key = str(path.resolve())
    if key not in _CACHE:
        required = (
            "basis",
            "prompt_basis",
            "anchors",
            "velocities",
            "x_mean",
            "x_std",
            "prompt_mean",
            "prompt_std",
            "prompt_weight",
            "prompt_bins",
            "bandwidth",
            "support_radius",
            "default_neighbors",
            "special_token_ids",
        )
        with np.load(path, allow_pickle=False) as payload:
            missing = [name for name in required if name not in payload]
            if missing:
                raise ValueError(f"subspace-flow asset missing {missing}")
            _CACHE[key] = {
                name: torch.from_numpy(np.asarray(payload[name]).copy()) for name in required
            }
    device = next(actor.actor_module.parameters()).device
    device_key = (key, str(device))
    if device_key not in _DEVICE_CACHE:
        result = {}
        for name, value in _CACHE[key].items():
            if name in {"special_token_ids", "default_neighbors", "prompt_bins"}:
                result[name] = value.to(device=device, dtype=torch.long)
            else:
                result[name] = value.to(device=device, dtype=torch.float32)
        _DEVICE_CACHE[device_key] = result
    return _DEVICE_CACHE[device_key]


def _valid_response_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    special_token_ids: torch.Tensor,
) -> torch.Tensor:
    special = special_token_ids.to(device=responses.device, dtype=responses.dtype)
    return response_mask.bool() & ~torch.isin(responses, special)


@torch.no_grad()
def _prompt_descriptor(
    prompt_hidden: torch.Tensor,
    prompt_mask: torch.Tensor,
    basis: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    unit = F.normalize(prompt_hidden.float(), dim=-1)
    coordinates = unit @ basis
    lengths = prompt_mask.sum(-1).clamp_min(1)
    ordinal = (prompt_mask.long().cumsum(-1) - 1).clamp_min(0)
    bin_index = torch.minimum(
        ordinal * bins // lengths[:, None], torch.full_like(ordinal, bins - 1)
    )
    pooled = []
    for index in range(bins):
        selected = prompt_mask & (bin_index == index)
        count = selected.sum(-1, keepdim=True).clamp_min(1)
        pooled.append((coordinates * selected[..., None]).sum(1) / count)
    mean = (coordinates * prompt_mask[..., None]).sum(1) / lengths[:, None]
    variance = (
        (coordinates - mean[:, None, :]).square() * prompt_mask[..., None]
    ).sum(1) / lengths[:, None]
    spread = variance.clamp_min(0).sqrt()
    return torch.cat([*pooled, spread], dim=-1)


@torch.no_grad()
def _kernel_velocity(
    standardized_state: torch.Tensor,
    asset: dict[str, torch.Tensor],
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions, nearest = [], []
    bandwidth = float(asset["bandwidth"].item())
    for start in range(0, len(standardized_state), 1024):
        query = standardized_state[start : start + 1024]
        distance = torch.cdist(query, asset["anchors"]).square()
        values, indices = distance.topk(
            min(neighbors, asset["anchors"].shape[0]), largest=False, dim=-1
        )
        weights = torch.softmax(
            -values / max(2.0 * bandwidth * bandwidth, 1e-8), dim=-1
        )
        predictions.append((weights[..., None] * asset["velocities"][indices]).sum(-2))
        nearest.append(values[:, 0].sqrt())
    return torch.cat(predictions), torch.cat(nearest)


@torch.no_grad()
def _midpoint_direction(
    hidden: torch.Tensor,
    prompt_hidden: torch.Tensor,
    prompt_mask: torch.Tensor,
    asset: dict[str, torch.Tensor],
    neighbors: int,
    integration_step: float,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = hidden.shape
    flat_mask = (
        torch.ones(shape[0] * shape[1], device=hidden.device, dtype=torch.bool)
        if active_mask is None else active_mask.reshape(-1).bool()
    )
    direction_out = torch.zeros(shape, device=hidden.device, dtype=torch.float32)
    nearest_out = torch.zeros(shape[:-1], device=hidden.device, dtype=torch.float32)
    gate_out = torch.zeros_like(nearest_out)
    if not bool(flat_mask.any()):
        return direction_out, nearest_out, gate_out
    unit = F.normalize(hidden.reshape(-1, shape[-1])[flat_mask].float(), dim=-1)
    coordinates = unit @ asset["basis"]
    prompt_by_case = _prompt_descriptor(
        prompt_hidden,
        prompt_mask.bool(),
        asset["prompt_basis"],
        int(asset["prompt_bins"].item()),
    )
    case_index = torch.arange(shape[0], device=hidden.device)[:, None].expand(
        shape[0], shape[1]
    ).reshape(-1)[flat_mask]
    prompt_coordinates = prompt_by_case[case_index]

    def joint_state(response_coordinates: torch.Tensor) -> torch.Tensor:
        response = (response_coordinates - asset["x_mean"]) / asset["x_std"]
        prompt = asset["prompt_weight"] * (
            (prompt_coordinates - asset["prompt_mean"]) / asset["prompt_std"]
        )
        return torch.cat([response, prompt], dim=-1)

    first, nearest = _kernel_velocity(joint_state(coordinates), asset, neighbors)
    midpoint = coordinates + 0.5 * integration_step * first
    second, _ = _kernel_velocity(joint_state(midpoint), asset, neighbors)
    direction = second @ asset["basis"].T
    # Hidden norm is deliberately removed as a degree of freedom.  The field
    # acts in the tangent plane of the unit-hidden sphere.
    direction = direction - unit * (unit * direction).sum(-1, keepdim=True)
    radius = float(asset["support_radius"].item())
    overflow = (nearest / max(radius, 1e-6) - 1.0).clamp_min(0.0)
    gate = torch.exp(-4.0 * overflow.square())
    gate = gate * (nearest <= 2.0 * radius).float()
    direction = direction * gate[:, None]
    direction_out.reshape(-1, shape[-1])[flat_mask] = direction
    nearest_out.reshape(-1)[flat_mask] = nearest
    gate_out.reshape(-1)[flat_mask] = gate
    return direction_out, nearest_out, gate_out


def update(actor, data):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("local-subspace-flow continuation currently requires one GPU")
    cfg = actor.config.policy_loss.hidden_field
    if not bool(cfg.get("enabled", False)):
        raise ValueError("loss_mode=opd_subspace_flow requires hidden_field.enabled=true")
    if actor.config.ppo_epochs != 1 or len(data) != actor.config.ppo_mini_batch_size:
        raise ValueError("subspace-flow continuation requires one optimizer step per rollout batch")

    asset = _load(actor)
    scale = float(cfg.get("scale", 1.0))
    minimum = int(cfg.get("minimum_tokens", 16))
    neighbors = int(cfg.get("neighbors", int(asset["default_neighbors"].item())))
    integration_step = float(cfg.get("integration_step", 1.0))
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("hidden_field.scale must be finite and positive")
    if neighbors < 2:
        raise ValueError("hidden_field.neighbors must be >= 2")
    if not 0 < integration_step <= 2:
        raise ValueError("hidden_field.integration_step must be in (0, 2]")

    complete_mask = _valid_response_mask(
        data.batch["responses"], data.batch["response_mask"], asset["special_token_ids"].cpu()
    )
    lengths = complete_mask.sum(-1)
    eligible_total = ((lengths >= minimum)[:, None] & complete_mask).sum().clamp_min(1)

    actor.actor_module.train()
    actor.actor_optimizer.zero_grad()
    loss_sum = 0.0
    direction_sq = 0.0
    nearest_sum = 0.0
    supported = 0.0
    valid_tokens = 0
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        device = next(actor.actor_module.parameters()).device
        part = part.to(device)
        inputs = dict(part.batch)
        _, _, full_hidden = actor._forward_micro_batch(
            inputs,
            temperature=float(data.meta_info["temperature"]),
            calculate_entropy=False,
            return_last_hidden=True,
            compute_log_probs=False,
        )
        response_length = inputs["responses"].shape[1]
        prompt_width = full_hidden.shape[1] - response_length
        prompt_hidden = full_hidden[:, :prompt_width, :]
        prompt_mask = inputs["attention_mask"][:, :prompt_width].bool()
        response_hidden = full_hidden[:, -response_length:, :]
        mask = _valid_response_mask(
            inputs["responses"], inputs["response_mask"], asset["special_token_ids"]
        )
        eligible = (mask.sum(-1) >= minimum)[:, None] & mask
        with torch.no_grad():
            direction, nearest, gate = _midpoint_direction(
                response_hidden.detach(),
                prompt_hidden.detach(),
                prompt_mask,
                asset,
                neighbors,
                integration_step,
                active_mask=eligible,
            )
        # Because hidden is normalized here, the backward signal is tangential;
        # gradient descent on the negative dot product moves along +direction.
        unit_hidden = F.normalize(response_hidden.float(), dim=-1)
        token_loss = -(unit_hidden * direction).sum(-1)
        loss = scale * (token_loss * eligible).sum() / eligible_total.to(token_loss.device)
        loss.backward()
        loss_sum += float(loss.detach())
        direction_sq += float((direction.square().sum(-1) * eligible).sum())
        nearest_sum += float((nearest * eligible).sum())
        supported += float(((gate > 0) & eligible).sum())
        valid_tokens += int(eligible.sum())
        del full_hidden, prompt_hidden, prompt_mask, response_hidden, direction, nearest, gate, unit_hidden, token_loss

    grad_norm = actor._optimizer_step()
    actor.actor_optimizer.zero_grad(set_to_none=True)
    count = max(valid_tokens, 1)
    return {
        "actor/subspace_flow_loss": [loss_sum],
        "actor/grad_norm": [float(grad_norm)],
        "actor/subspace_flow_direction_rms": [math.sqrt(direction_sq / count)],
        "actor/subspace_flow_nearest_distance_mean": [nearest_sum / count],
        "actor/subspace_flow_support_fraction": [supported / count],
        "actor/subspace_flow_tokens": [float(valid_tokens)],
        "actor/subspace_flow_scale": [scale],
        "monitor/teacher_present": [0.0],
        "monitor/token_time_feature_present": [0.0],
        "monitor/all_prompt_token_hidden_condition_present": [1.0],
        "monitor/linear_checkpoint_extrapolation": [0.0],
        "monitor/local_midpoint_field": [1.0],
    }
