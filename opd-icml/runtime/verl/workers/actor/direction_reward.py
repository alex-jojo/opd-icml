"""Frozen signed high-vs-low representation-direction reward.

The measurement model must be a frozen copy of the calibration backbone.  No
tensor produced here is connected to actor autograd.  The scalar reward is
positive when an observed trajectory moves more like real high-quality
transitions than real low-quality transitions, and negative for the reverse.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-6


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(EPS)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1) / (
        left.norm(dim=-1) * right.norm(dim=-1)
    ).clamp_min(EPS)


class FrozenDirectionReward(nn.Module):
    """Evaluate frozen trajectories against two real transition fields."""

    REQUIRED_KEYS = {
        "format_version",
        "hidden_size",
        "projection_dim",
        "bins",
        "neighbors",
        "hash_columns",
        "hash_signs",
        "ridge_good_mean",
        "ridge_good_scale",
        "ridge_good_coef",
        "ridge_good_intercept",
        "ridge_low_mean",
        "ridge_low_scale",
        "ridge_low_coef",
        "ridge_low_intercept",
        "bank_states",
        "bank_directions",
        "bank_magnitudes",
        "bank_mask",
        "ridge_weight",
        "observed_weight",
        "reward_scale",
        "support_center",
        "support_temperature",
        "special_token_ids",
    }

    def __init__(self, manifest_path: str, *, minimum_tokens: int = 16) -> None:
        super().__init__()
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"direction reward manifest not found: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("method") != "signed_good_vs_low_direction_reward_v1":
            raise ValueError("unsupported direction reward method")
        weights_name = manifest.get("weights_file")
        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("direction reward manifest requires a quoted weights_file")
        # The bundled, trusted calibration asset contains NumPy arrays as
        # well as tensors. PyTorch 2.6 rejects those objects when
        # weights_only=True, so make the trusted full-load decision explicit.
        asset = torch.load(path.parent / weights_name, map_location="cpu", weights_only=False)
        missing = self.REQUIRED_KEYS - set(asset)
        if missing:
            raise KeyError(f"direction reward asset missing keys: {sorted(missing)}")
        if int(asset["format_version"]) != 1:
            raise ValueError("unsupported direction reward asset version")

        self.manifest = manifest
        self.hidden_size = int(asset["hidden_size"])
        self.projection_dim = int(asset["projection_dim"])
        self.bins = int(asset["bins"])
        self.transitions = self.bins - 1
        self.neighbors = int(asset["neighbors"])
        self.minimum_tokens = max(int(minimum_tokens), self.bins)
        self.ridge_weight = float(asset["ridge_weight"])
        self.observed_weight = float(asset["observed_weight"])
        self.reward_scale = float(asset["reward_scale"])
        self.support_center = float(asset["support_center"])
        self.support_temperature = float(asset["support_temperature"])
        if self.ridge_weight < 0 or self.observed_weight < 0:
            raise ValueError("direction branch weights must be non-negative")
        if abs(self.ridge_weight + self.observed_weight - 1.0) > 1e-5:
            raise ValueError("direction branch weights must sum to one")
        if self.reward_scale <= 0 or self.support_temperature <= 0:
            raise ValueError("reward scale and support temperature must be positive")

        int_buffers = {"hash_columns", "special_token_ids"}
        bool_buffers = {"bank_mask"}
        for name in self.REQUIRED_KEYS:
            if name in {
                "format_version",
                "hidden_size",
                "projection_dim",
                "bins",
                "neighbors",
                "ridge_weight",
                "observed_weight",
                "reward_scale",
                "support_center",
                "support_temperature",
            }:
                continue
            value = asset[name]
            dtype = torch.long if name in int_buffers else torch.bool if name in bool_buffers else torch.float32
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype), persistent=False)
        self.register_buffer("stage_codes", self._make_stage_codes(), persistent=False)
        self._validate_shapes()

    def _make_stage_codes(self) -> torch.Tensor:
        rows = []
        for stage in range(self.transitions):
            one_hot = torch.zeros(self.transitions, dtype=torch.float32)
            one_hot[stage] = 1.0
            fraction = stage / max(self.transitions - 1, 1)
            periodic = torch.tensor(
                [
                    fraction,
                    math.sin(2.0 * math.pi * fraction),
                    math.cos(2.0 * math.pi * fraction),
                    math.sin(4.0 * math.pi * fraction),
                    math.cos(4.0 * math.pi * fraction),
                ],
                dtype=torch.float32,
            )
            rows.append(torch.cat((one_hot, periodic)))
        return torch.stack(rows)

    def _validate_shapes(self) -> None:
        input_dim = self.projection_dim + self.transitions + 5
        if self.hash_columns.shape != (self.hidden_size,) or self.hash_signs.shape != (self.hidden_size,):
            raise ValueError("feature-hash arrays do not match hidden_size")
        if int(self.hash_columns.max()) >= self.projection_dim or int(self.hash_columns.min()) < 0:
            raise ValueError("feature-hash column out of range")
        for prefix in ("ridge_good", "ridge_low"):
            if getattr(self, f"{prefix}_mean").shape != (input_dim,):
                raise ValueError(f"{prefix}_mean has invalid shape")
            if getattr(self, f"{prefix}_scale").shape != (input_dim,):
                raise ValueError(f"{prefix}_scale has invalid shape")
            if getattr(self, f"{prefix}_coef").shape != (self.projection_dim, input_dim):
                raise ValueError(f"{prefix}_coef has invalid shape")
            if getattr(self, f"{prefix}_intercept").shape != (self.projection_dim,):
                raise ValueError(f"{prefix}_intercept has invalid shape")
        expected_prefix = (2, self.transitions)
        if tuple(self.bank_states.shape[:2]) != expected_prefix or self.bank_states.shape[-1] != self.projection_dim:
            raise ValueError("bank_states has invalid shape")
        if self.bank_directions.shape != self.bank_states.shape:
            raise ValueError("bank_directions shape mismatch")
        if self.bank_magnitudes.shape != self.bank_states.shape[:-1]:
            raise ValueError("bank_magnitudes shape mismatch")
        if self.bank_mask.shape != self.bank_states.shape[:-1]:
            raise ValueError("bank_mask shape mismatch")
        if not self.bank_mask.any(dim=-1).all():
            raise ValueError("every class/stage bank must contain real transitions")

    def validate_runtime(self, *, hidden_size: int, model_path: str | None) -> None:
        if int(hidden_size) != self.hidden_size:
            raise ValueError(f"direction reward hidden mismatch: asset={self.hidden_size}, model={hidden_size}")
        expected = self.manifest.get("model_id")
        if expected and model_path:
            observed = str(model_path).replace("\\", "/").rstrip("/")
            expected = str(expected).replace("\\", "/").rstrip("/")
            if observed != expected and not observed.endswith("/" + expected.split("/")[-1]):
                raise ValueError(f"direction reward was calibrated for {expected}, not {model_path}")

    def _valid_mask(self, responses: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        valid = response_mask.bool()
        if self.special_token_ids.numel():
            special = (responses[..., None] == self.special_token_ids.to(responses.device)).any(dim=-1)
            valid = valid & ~special
        return valid

    def _hash_project(self, relative: torch.Tensor) -> torch.Tensor:
        source = relative.float() * self.hash_signs.unsqueeze(0)
        projected = source.new_zeros((source.shape[0], self.projection_dim))
        projected.index_add_(1, self.hash_columns, source)
        return projected

    def _states(self, hidden: torch.Tensor, prompt_end: torch.Tensor) -> torch.Tensor:
        relative = _unit(hidden.float() - prompt_end.float().unsqueeze(0))
        projected = self._hash_project(relative)
        edges = torch.linspace(0, len(projected), self.bins + 1, device=projected.device)
        rows = []
        for index in range(self.bins):
            start = min(int(math.floor(float(edges[index].item()))), len(projected) - 1)
            stop = max(start + 1, int(math.ceil(float(edges[index + 1].item()))))
            rows.append(projected[start : min(stop, len(projected))].mean(dim=0))
        return _unit(torch.stack(rows))

    def _ridge(self, states: torch.Tensor, prefix: str) -> torch.Tensor:
        design = torch.cat((states[:-1], self.stage_codes), dim=-1)
        mean = getattr(self, f"{prefix}_mean")
        scale = getattr(self, f"{prefix}_scale").clamp_min(EPS)
        coef = getattr(self, f"{prefix}_coef")
        intercept = getattr(self, f"{prefix}_intercept")
        return ((design - mean) / scale) @ coef.transpose(0, 1) + intercept

    def _observed(self, state: torch.Tensor, stage: int, class_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.bank_mask[class_index, stage]
        bank_state = self.bank_states[class_index, stage, mask]
        direction = self.bank_directions[class_index, stage, mask]
        magnitude = self.bank_magnitudes[class_index, stage, mask]
        similarity = bank_state @ state
        count = min(self.neighbors, int(similarity.numel()))
        local, chosen = torch.topk(similarity, k=count)
        temperature = local.std(unbiased=False).clamp_min(0.05)
        weight = F.softmax((local - local.max()) / temperature, dim=0)
        vector = (direction[chosen] * magnitude[chosen, None] * weight[:, None]).sum(dim=0)
        agreement = (direction[chosen] * weight[:, None]).sum(dim=0).norm().clamp(0.0, 1.0)
        return vector, agreement, local.max()

    @torch.no_grad()
    def forward(
        self,
        *,
        full_hidden: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if full_hidden.ndim != 3 or full_hidden.shape[-1] != self.hidden_size:
            raise ValueError("full_hidden must be [batch, sequence, hidden_size]")
        if responses.shape != response_mask.shape:
            raise ValueError("responses and response_mask must have identical shapes")
        response_length = responses.shape[1]
        if full_hidden.shape[1] < response_length + 1:
            raise ValueError("full_hidden does not contain prompt-end hidden")
        response_hidden = full_hidden[:, -response_length:, :]
        prompt_end = full_hidden[:, -response_length - 1, :]
        valid_mask = self._valid_mask(responses, response_mask)

        rewards = []
        raw_scores = []
        ridge_scores = []
        observed_scores = []
        support_scores = []
        valid_rows = []
        for row in range(full_hidden.shape[0]):
            hidden = response_hidden[row][valid_mask[row]]
            is_valid = len(hidden) >= self.minimum_tokens
            valid_rows.append(float(is_valid))
            if not is_valid:
                zero = full_hidden.new_zeros((), dtype=torch.float32)
                rewards.append(zero)
                raw_scores.append(zero)
                ridge_scores.append(zero)
                observed_scores.append(zero)
                support_scores.append(zero)
                continue

            states = self._states(hidden, prompt_end[row])
            actual = states[1:] - states[:-1]
            ridge_good = self._ridge(states, "ridge_good")
            ridge_low = self._ridge(states, "ridge_low")
            ridge_contrast = 0.5 * (_cosine(actual, ridge_good) - _cosine(actual, ridge_low))

            observed_contrast = []
            observed_confidence = []
            support = []
            for stage in range(self.transitions):
                good_vector, good_agreement, good_support = self._observed(states[stage], stage, 0)
                low_vector, low_agreement, low_support = self._observed(states[stage], stage, 1)
                observed_contrast.append(
                    0.5
                    * (
                        _cosine(actual[stage : stage + 1], good_vector.unsqueeze(0))[0]
                        - _cosine(actual[stage : stage + 1], low_vector.unsqueeze(0))[0]
                    )
                )
                observed_confidence.append((good_agreement * low_agreement).clamp_min(0.0).sqrt())
                support.append(torch.maximum(good_support, low_support))
            observed_contrast_t = torch.stack(observed_contrast)
            observed_confidence_t = torch.stack(observed_confidence)
            support_t = torch.stack(support)
            support_gate = torch.sigmoid(
                (support_t - self.support_center) / max(self.support_temperature, EPS)
            )
            stage_score = (
                self.ridge_weight * ridge_contrast
                + self.observed_weight * observed_contrast_t * observed_confidence_t
            )
            # Support is a single positive trajectory confidence. Applying it
            # after stage aggregation guarantees that it can attenuate but can
            # never flip the good-vs-low reward sign.
            raw = stage_score.mean() * support_gate.mean()
            reward = (raw / max(self.reward_scale, EPS)).clamp(-1.0, 1.0)
            rewards.append(reward)
            raw_scores.append(raw)
            ridge_scores.append(ridge_contrast.mean())
            observed_scores.append((observed_contrast_t * observed_confidence_t).mean())
            support_scores.append(support_gate.mean())

        return {
            "reward": torch.stack(rewards),
            "raw_score": torch.stack(raw_scores),
            "ridge_contrast": torch.stack(ridge_scores),
            "observed_contrast": torch.stack(observed_scores),
            "support": torch.stack(support_scores),
            "valid": torch.tensor(valid_rows, device=full_hidden.device, dtype=torch.float32),
            "valid_tokens": valid_mask.sum(dim=-1).float(),
        }
