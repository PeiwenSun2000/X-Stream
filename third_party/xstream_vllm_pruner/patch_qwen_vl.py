"""Monkey-patch ``vllm.multimodal.evs.compute_retention_mask``.

This is the single narrow seam we depend on inside vLLM. Both
``Qwen2_5_VLForConditionalGeneration._postprocess_video_embeds_evs`` and the
Qwen3-VL counterpart call ``compute_retention_mask`` to decide which
post-merger video tokens survive into the LLM. Replacing this function lets us
inject SURGE / CDPruner without touching any vLLM source file.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable

logger = logging.getLogger("xstream_vllm_pruner.patch_qwen_vl")

_PATCHED = False
_orig_compute_retention_mask: Callable | None = None


def install() -> bool:
    """Replace ``vllm.multimodal.evs.compute_retention_mask`` in-place.

    Returns ``True`` if the patch is now active, ``False`` if it was a no-op.
    Safe to call multiple times.
    """
    global _PATCHED, _orig_compute_retention_mask
    if _PATCHED:
        return True

    try:
        import vllm.multimodal.evs as _evs  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only in misuse
        logger.warning("xstream_vllm_pruner: cannot import vllm.multimodal.evs (%s)", exc)
        return False

    _orig_compute_retention_mask = _evs.compute_retention_mask

    def _patched_compute_retention_mask(
        video_embeds, video_size_thw, spatial_merge_size, q
    ):
        # Local imports keep the cold path cheap and avoid surfacing optional
        # deps (e.g. torch) before vLLM has been imported.
        import torch  # type: ignore

        from .config import get_runtime_config
        from .context import get_instruction

        cfg = get_runtime_config()
        if not cfg.enabled:
            return _orig_compute_retention_mask(  # type: ignore[misc]
                video_embeds, video_size_thw, spatial_merge_size, q
            )

        T, H, W = map(int, video_size_thw)
        Hm = H // spatial_merge_size
        Wm = W // spatial_merge_size
        tokens_per_frame = Hm * Wm

        # vLLM stipulates that the True count of the returned mask equals the
        # number of placeholders reserved at prompt-construction time.
        retain_num = _evs.compute_retained_tokens_count(
            tokens_per_frame=tokens_per_frame, num_frames=T, q=q
        )

        try:
            if cfg.algo == "surge":
                from .surge import SurgeConfig, run_surge

                surge_cfg = SurgeConfig(keep_first_frame=cfg.keep_first_frame)
                mask = run_surge(video_embeds, T, Hm, Wm, retain_num, surge_cfg)
            else:
                from .cdpruner import CDPrunerConfig, run_cdpruner

                cdp_cfg = CDPrunerConfig(
                    clip_model=cfg.clip_model,
                    keep_first_frame=cfg.keep_first_frame,
                )
                instruction = get_instruction()
                mask = run_cdpruner(
                    video_embeds, T, Hm, Wm, retain_num, instruction, cdp_cfg
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "xstream_vllm_pruner: %s pruning failed, falling back to EVS (%s)",
                cfg.algo,
                exc,
            )
            return _orig_compute_retention_mask(  # type: ignore[misc]
                video_embeds, video_size_thw, spatial_merge_size, q
            )

        # Strict invariant: True count must match retain_num so vLLM's
        # placeholder accounting stays consistent.
        true_count = int(mask.sum().item()) if isinstance(mask, torch.Tensor) else int(sum(bool(x) for x in mask))
        if true_count != retain_num:
            logger.warning(
                "xstream_vllm_pruner: %s produced %d kept tokens, expected %d. Falling back to EVS.",
                cfg.algo,
                true_count,
                retain_num,
            )
            return _orig_compute_retention_mask(  # type: ignore[misc]
                video_embeds, video_size_thw, spatial_merge_size, q
            )

        if cfg.debug:
            msg = (
                f"xstream_vllm_pruner: algo={cfg.algo} T={T} "
                f"Hm={Hm} Wm={Wm} kept={true_count} / {T * tokens_per_frame}"
            )
            logger.info(msg)
            print(msg, file=sys.stderr, flush=True)
        return mask

    _evs.compute_retention_mask = _patched_compute_retention_mask

    # Also override any already-imported references (Qwen2.5-VL / Qwen3-VL
    # capture the symbol via ``from vllm.multimodal.evs import ...``).
    _maybe_patch_model_modules(_patched_compute_retention_mask)

    _PATCHED = True
    msg = "xstream_vllm_pruner: compute_retention_mask patch installed"
    logger.info(msg)
    print(msg, file=sys.stderr, flush=True)
    return True


def uninstall() -> bool:
    """Restore the original ``compute_retention_mask`` if currently patched."""
    global _PATCHED, _orig_compute_retention_mask
    if not _PATCHED or _orig_compute_retention_mask is None:
        return False
    try:
        import vllm.multimodal.evs as _evs  # type: ignore

        _evs.compute_retention_mask = _orig_compute_retention_mask
        _maybe_patch_model_modules(_orig_compute_retention_mask)
    except Exception:  # pragma: no cover
        return False
    _PATCHED = False
    return True


def _maybe_patch_model_modules(target: Callable) -> None:
    """Rebind already-imported ``compute_retention_mask`` references.

    Both Qwen2.5-VL and Qwen3-VL import the function at module load time::

        from vllm.multimodal.evs import compute_retention_mask

    That binding is independent of ``vllm.multimodal.evs.compute_retention_mask``
    once the module is loaded, so we walk known model modules and rebind the
    attribute when present. Models loaded later go through the patched module
    attribute by default.
    """
    for mod_name in (
        "vllm.model_executor.models.qwen2_5_vl",
        "vllm.model_executor.models.qwen3_vl",
        "vllm.model_executor.models.qwen3_vl_moe",
        "vllm.model_executor.models.qwen3_omni_moe_thinker",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "compute_retention_mask"):
            try:
                setattr(mod, "compute_retention_mask", target)
            except Exception:  # pragma: no cover
                continue


def is_installed() -> bool:
    return _PATCHED
