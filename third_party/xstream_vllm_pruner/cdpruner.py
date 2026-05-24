"""CDPruner patch-level keep mask (conditional DPP fast MAP).

Faithful port of the CDPruner core idea from
``CDPruner/llava/model/llava_arch.py``: rank visual tokens by combining
instruction relevance with intra-token diversity and pick a subset that
maximizes the determinant of a conditional DPP kernel via the fast MAP
greedy described in ``fast-map-dpp``.

In the vLLM hook we receive post-merger video embeddings (``[T*Hm*Wm, D]``)
but no instruction string. The instruction is propagated via a ``ContextVar``
populated in ``patch_processor.py``. The CLIP text tower used to encode the
instruction is lazy-loaded on first call so SURGE-only deployments never pay
its memory cost.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CDPrunerConfig:
    """CDPruner configuration."""

    clip_model: str = "openai/clip-vit-large-patch14-336"
    keep_first_frame: bool = True


# ---------------------------------------------------------------------------
# CLIP text-tower handling
# ---------------------------------------------------------------------------

_clip_lock = threading.Lock()
_clip_tokenizer = None
_clip_text_model = None
_clip_device: Optional[torch.device] = None


def _ensure_clip_text_tower(model_name: str, device: torch.device) -> None:
    """Lazy-load a CLIP text tokenizer + text projection model.

    Loaded on CPU first then moved to ``device`` to avoid the ``meta`` weight
    pitfall that has bitten the existing MLLMFlow utility.
    """
    global _clip_tokenizer, _clip_text_model, _clip_device
    if _clip_tokenizer is not None and _clip_text_model is not None:
        if _clip_device == device:
            return
        with _clip_lock:
            _clip_text_model = _clip_text_model.to(device)  # type: ignore[union-attr]
            _clip_device = device
            return

    with _clip_lock:
        if _clip_tokenizer is not None and _clip_text_model is not None:
            return
        # Local import so unrelated deployments do not depend on transformers.
        from transformers.models.clip import (  # type: ignore
            CLIPTextModelWithProjection,
            CLIPTokenizerFast,
        )

        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(
            model_name, device_map=None, low_cpu_mem_usage=False
        )
        text_model.to(device)
        text_model.eval()

        _clip_tokenizer = tokenizer
        _clip_text_model = text_model
        _clip_device = device


def _encode_instruction(text: str, device: torch.device, model_name: str) -> Optional[torch.Tensor]:
    """Return an L2-normalized CLIP text embedding ``[D_text]`` or ``None``."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        _ensure_clip_text_tower(model_name, device)
    except Exception:
        # When transformers / weights are unavailable, CDPruner falls back to
        # pure-visual diversity (relevance becomes constant).
        return None

    assert _clip_tokenizer is not None
    assert _clip_text_model is not None

    inputs = _clip_tokenizer(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _clip_text_model(**inputs)
    embed = getattr(out, "text_embeds", None)
    if embed is None:
        last_hidden = out.last_hidden_state  # type: ignore[attr-defined]
        embed = last_hidden[:, 0, :]
    embed = embed.squeeze(0).to(torch.float32)
    return embed / (embed.norm() + 1e-6)


# ---------------------------------------------------------------------------
# Conditional DPP fast MAP greedy
# ---------------------------------------------------------------------------


def _project_visual_to_text_dim(visual: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project visual tokens to ``target_dim`` via a deterministic linear map.

    The vLLM video embeddings live in the LLM hidden-state space which has a
    different dimensionality than the CLIP text tower. Real CDPruner runs CLIP
    on the raw pixels, but we cannot re-encode images inside the vLLM hook, so
    we build a fixed, reproducible projection (an orthonormal slice / pad) so
    that relevance computations are well-defined without training a head.
    """
    N, D = visual.shape
    if D == target_dim:
        return visual
    if D > target_dim:
        # Deterministic truncation: take the first ``target_dim`` channels.
        return visual[:, :target_dim].contiguous()
    pad = torch.zeros(N, target_dim - D, dtype=visual.dtype, device=visual.device)
    return torch.cat([visual, pad], dim=1)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _conditional_dpp_kernel(
    visual_embeds: torch.Tensor,
    text_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """Construct the CDPruner conditional DPP kernel ``[N, N]``.

    Follows the original CDPruner formulation: ``L = q * S * q.T`` where
    ``S`` is the cosine-similarity Gram matrix of visual features and ``q``
    is a [0, 1] relevance score derived from negative CLIP cosine similarity.
    """
    visual_norm = _normalize_rows(visual_embeds.to(torch.float32))
    similarity = visual_norm @ visual_norm.t()  # [N, N]

    if text_embed is None:
        # Without instruction information the kernel reduces to the visual
        # similarity itself, i.e. pure diversity selection.
        return similarity

    text_proj = _project_visual_to_text_dim(visual_norm, text_embed.shape[0])
    text_proj = _normalize_rows(text_proj)
    # CDPruner negates the cosine score so high similarity to the instruction
    # ends up with a low relevance value before re-scaling.
    relevance = -(text_proj @ text_embed.to(text_proj.dtype))
    rel_min = relevance.min()
    rel_max = relevance.max()
    relevance = (relevance - rel_min + 1e-6) / (rel_max - rel_min + 1e-6)

    rel_col = relevance.unsqueeze(1)
    rel_row = relevance.unsqueeze(0)
    return rel_col * similarity * rel_row


def _fast_map_dpp_greedy(kernel: torch.Tensor, k: int) -> torch.Tensor:
    """Pick ``k`` indices maximizing the DPP MAP via the standard fast greedy.

    Reference: Chen et al., *Fast Greedy MAP Inference for Determinantal Point
    Process to Improve Recommendation Diversity*. Adapted from
    ``fast-map-dpp`` and the CDPruner codebase.
    """
    N = kernel.shape[0]
    device = kernel.device
    k = max(0, min(k, N))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone().to(torch.float32)
    cis = torch.zeros((k, N), device=device, dtype=torch.float32)
    selected = torch.empty(k, dtype=torch.long, device=device)

    for i in range(k):
        j = int(torch.argmax(di2s).item())
        selected[i] = j
        if di2s[j].item() <= 0:
            di2s[j] = float("-inf")
            continue
        # Update the orthogonalized representation.
        ei = (kernel[j].to(torch.float32) - cis[:i, j] @ cis[:i]) / torch.sqrt(
            di2s[j] + 1e-6
        )
        cis[i] = ei
        di2s = di2s - ei * ei
        di2s[j] = float("-inf")

    return selected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cdpruner(
    video_embeds: torch.Tensor,
    T: int,
    Hm: int,
    Wm: int,
    retain_num: int,
    instruction: str,
    cfg: CDPrunerConfig,
) -> torch.Tensor:
    """Compute the CDPruner retention mask for a single video.

    The returned tensor is a 1-D bool mask of length ``T*Hm*Wm`` with
    exactly ``retain_num`` ``True`` entries.
    """
    m = Hm * Wm
    expected = T * m
    assert video_embeds.shape[0] == expected, (
        f"cdpruner: expected {expected} tokens (T={T}, Hm={Hm}, Wm={Wm}), "
        f"got {video_embeds.shape[0]}"
    )

    device = video_embeds.device
    keep_mask = torch.zeros(expected, dtype=torch.bool, device=device)
    if expected == 0 or retain_num <= 0:
        return keep_mask

    retain_num = min(retain_num, expected)

    # Force-keep the first frame to mirror SURGE / EVS conventions and the
    # CDPruner intuition that the first observation is always informative.
    forced = 0
    if cfg.keep_first_frame and m > 0 and retain_num >= m:
        keep_mask[:m] = True
        forced = m

    remaining_budget = retain_num - forced
    if remaining_budget <= 0:
        return keep_mask

    # Build the candidate pool (everything that is not yet forced-kept).
    if forced > 0:
        candidate_idx = torch.arange(forced, expected, device=device)
    else:
        candidate_idx = torch.arange(expected, device=device)

    candidate_embeds = video_embeds.index_select(0, candidate_idx).to(torch.float32)

    text_embed: Optional[torch.Tensor] = None
    if instruction.strip():
        text_embed = _encode_instruction(instruction, device, cfg.clip_model)

    kernel = _conditional_dpp_kernel(candidate_embeds, text_embed)
    picked_local = _fast_map_dpp_greedy(kernel, remaining_budget)
    picked_global = candidate_idx.index_select(0, picked_local)
    keep_mask[picked_global] = True

    # Defensive: ensure exactly ``retain_num`` True entries even when the DPP
    # marks fewer (e.g. degenerate kernels collapsed all di2s to <= 0).
    short = retain_num - int(keep_mask.sum().item())
    if short > 0:
        fallback_pool = (~keep_mask).nonzero(as_tuple=True)[0]
        if fallback_pool.numel() > 0:
            extra = fallback_pool[: min(short, fallback_pool.numel())]
            keep_mask[extra] = True

    return keep_mask
