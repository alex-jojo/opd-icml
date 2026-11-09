"""Strict boxed-only mathematical correctness for DAPO training monitors."""

from __future__ import annotations

from typing import Optional

from .math_verify import (
    compute_score as compute_strict_boxed_math_score,
    last_boxed_only_string,
    remove_boxed,
)


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict[str, float | str | None]:
    """Return strict final-boxed correctness and format diagnostics.

    The compatibility arguments remain in the signature for existing VERL
    call sites. Training and formal evaluation use the same boxed-only
    correctness contract.
    """

    del strict_box_verify, pause_tokens_index
    boxed = last_boxed_only_string(solution_str)
    boxed_format = boxed is not None
    boxed_correct = bool(
        boxed_format and compute_strict_boxed_math_score(solution_str, ground_truth)
    )
    pred = remove_boxed(boxed) if boxed_format else None
    return {
        "score": float(boxed_correct),
        "acc": float(boxed_correct),
        "boxed_acc": float(boxed_correct),
        "boxed_format": float(boxed_format),
        "pred": pred,
        "extraction_mode": "boxed" if boxed_format else "none",
    }
