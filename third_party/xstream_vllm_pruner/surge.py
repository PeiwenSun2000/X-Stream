"""SURGE patch-level keep mask used to replace vLLM EVS for video tokens.

This is a slimmed inference-time port of ``surge_core.compute_surge`` from
``SURGE/surge/surge_core.py``. Compared to the reference implementation it:

- skips the diagnostics that need ``scipy.signal.find_peaks`` (we only need
  ``keep_mask``);
- enforces an exact ``retain_num`` top-k selection so the returned mask has
  precisely the number of ``True`` entries that vLLM's prompt-stage already
  reserved via ``compute_retained_tokens_count``;
- works on the post-merger ``video_embeds`` tensor of shape ``[T*Hm*Wm, D]``
  that vLLM hands to ``vllm.multimodal.evs.compute_retention_mask``.

The output is a 1-D bool tensor of length ``T * Hm * Wm`` (matching the
contract of ``compute_retention_mask``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SurgeConfig:
    """SURGE configuration for inference-time pruning."""

    enable_drift_correction: bool = True
    enable_variance_norm: bool = True
    ema_var_decay: float = 0.9
    epsilon: float = 1e-8
    keep_first_frame: bool = True


def _build_spatial_coords(
    H_merged: int, W_merged: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build the spatial design matrix ``[1, x, y]`` of shape ``[Hm*Wm, 3]``."""
    ys = torch.arange(H_merged, device=device, dtype=dtype)
    xs = torch.arange(W_merged, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones(H_merged * W_merged, device=device, dtype=dtype)
    return torch.stack([ones, grid_x.flatten(), grid_y.flatten()], dim=1)


def _compute_surprise_scores(
    tokens: torch.Tensor,
    T: int,
    Hm: int,
    Wm: int,
    cfg: SurgeConfig,
) -> torch.Tensor:
    """Compute ``[T, m]`` per-token surprise scores in float32.

    Frame 0 is assigned ``+inf`` so the always-keep-first-frame guarantee can
    be implemented by simply taking the top-k of the resulting tensor.
    """
    device = tokens.device
    compute_dtype = torch.float32
    m = Hm * Wm
    tokens_f32 = tokens.to(compute_dtype)

    if cfg.enable_drift_correction and m > 0:
        X = _build_spatial_coords(Hm, Wm, device, compute_dtype)
        # Solve once per call: XtX_inv_Xt has shape [3, m].
        XtX_inv_Xt = torch.linalg.solve(X.T @ X, X.T)
    else:
        X = None
        XtX_inv_Xt = None

    scores = torch.zeros(T, m, device=device, dtype=compute_dtype)
    running_var: torch.Tensor | None = None

    for t in range(T):
        if t == 0:
            scores[t] = float("inf")
            continue

        tokens_t = tokens_f32[t]
        tokens_tm1 = tokens_f32[t - 1]

        if t == 1:
            error = tokens_t - tokens_tm1
            score = (error * error).sum(dim=1)
            running_var = score.clone()
            if cfg.enable_variance_norm:
                scores[t] = score / (running_var + cfg.epsilon)
            else:
                scores[t] = score
            continue

        tokens_tm2 = tokens_f32[t - 2]
        raw_delta = tokens_tm1 - tokens_tm2

        if cfg.enable_drift_correction and X is not None and XtX_inv_Xt is not None:
            C = XtX_inv_Xt @ raw_delta
            drift = X @ C
            detrended_delta = raw_delta - drift
        else:
            detrended_delta = raw_delta

        pred = tokens_tm1 + detrended_delta
        error = tokens_t - pred
        score = (error * error).sum(dim=1)

        if cfg.enable_variance_norm:
            assert running_var is not None
            running_var = cfg.ema_var_decay * running_var + (1 - cfg.ema_var_decay) * score
            scores[t] = score / (running_var + cfg.epsilon)
        else:
            scores[t] = score

    return scores


def _topk_mask_from_scores(
    scores: torch.Tensor,
    retain_num: int,
    keep_first_frame: bool,
) -> torch.Tensor:
    """Select exactly ``retain_num`` positions with the highest score.

    When ``keep_first_frame`` is True, the entire first frame is forced into
    the kept set first, then the remaining budget is filled by the top-k of
    the rest.
    """
    T, m = scores.shape
    flat = scores.reshape(T * m).clone()
    total = T * m
    retain_num = max(0, min(retain_num, total))

    keep_mask = torch.zeros(total, dtype=torch.bool, device=scores.device)
    if retain_num == 0:
        return keep_mask

    # First-frame forced keep (frame 0 already has +inf scores, so a normal
    # top-k would pick it anyway; this branch handles the corner case where
    # the user explicitly disables that behavior).
    if keep_first_frame and m > 0 and retain_num >= m:
        keep_mask[:m] = True
        flat[:m] = float("-inf")  # avoid re-selecting frame 0 patches
        remaining = retain_num - m
    elif keep_first_frame and m > 0:
        # Budget smaller than one frame: keep the highest-score positions
        # from frame 0 (their scores are inf, ties broken arbitrarily).
        topk_idx = torch.topk(flat[:m], k=retain_num, dim=0).indices
        keep_mask[topk_idx] = True
        return keep_mask
    else:
        remaining = retain_num

    if remaining > 0:
        # Only consider not-yet-selected positions.
        finite = torch.where(
            keep_mask, torch.full_like(flat, float("-inf")), flat
        )
        # Replace any NaNs that may have slipped through (defensive).
        finite = torch.nan_to_num(finite, nan=float("-inf"))
        topk_idx = torch.topk(finite, k=remaining, dim=0).indices
        keep_mask[topk_idx] = True

    return keep_mask


def run_surge(
    video_embeds: torch.Tensor,
    T: int,
    Hm: int,
    Wm: int,
    retain_num: int,
    cfg: SurgeConfig,
) -> torch.Tensor:
    """Compute the SURGE retention mask for a single video.

    Args:
        video_embeds: ``[T*Hm*Wm, D]`` post-merger video embeddings.
        T: temporal grid size (frames).
        Hm: post-merge spatial height.
        Wm: post-merge spatial width.
        retain_num: total tokens to keep (must match
            ``vllm.multimodal.evs.compute_retained_tokens_count``).
        cfg: ``SurgeConfig`` instance.

    Returns:
        Bool tensor of shape ``[T*Hm*Wm]`` with exactly ``retain_num`` True
        entries.
    """
    m = Hm * Wm
    expected = T * m
    assert video_embeds.shape[0] == expected, (
        f"surge: expected {expected} tokens (T={T}, Hm={Hm}, Wm={Wm}), "
        f"got {video_embeds.shape[0]}"
    )

    if expected == 0:
        return torch.zeros(0, dtype=torch.bool, device=video_embeds.device)

    tokens = video_embeds.reshape(T, m, video_embeds.shape[-1])
    scores = _compute_surprise_scores(tokens, T, Hm, Wm, cfg)
    return _topk_mask_from_scores(scores, retain_num, cfg.keep_first_frame)
