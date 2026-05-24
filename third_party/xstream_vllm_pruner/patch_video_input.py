"""Soft patch-level pruning hook for Qwen models that do NOT use vLLM's EVS.

vLLM only routes Qwen2.5-VL / Qwen3-VL / Qwen3-VL MoE through
``vllm.multimodal.evs.compute_retention_mask``. The Omni Thinker family
(Qwen2.5-Omni, Qwen3-Omni MoE Thinker) and Qwen2-VL never call EVS, so the
``patch_qwen_vl.py`` seam alone cannot prune their video tokens.

To extend X-Stream's patch-level pruner to the full Qwen family without
modifying any vLLM source, we wrap ``_process_video_input`` on the relevant
classes. Inside the wrapper we apply a **soft** mask (zeroing out the
embedding of dropped patches) instead of physically dropping rows. Soft mask
keeps the token count identical to the unmodified path, which leaves vLLM's
placeholder bookkeeping and ``_merge_multimodal_embeddings`` untouched.

This is a deliberate trade-off:
- For EVS-capable models we still get the original *hard* prune through
  ``compute_retention_mask`` (KV-cache shrinks).
- For non-EVS Qwen models we get *soft* pruning (KV-cache size unchanged,
  but the dropped patches contribute zero signal to attention).

Both behaviors are token-level inside a frame, which is the user requirement.
"""
from __future__ import annotations

import functools
import logging
import sys
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger("xstream_vllm_pruner.patch_video_input")

_PATCHED = False
_orig_methods: dict[str, Callable] = {}


def _debug_print(message: str) -> None:
    logger.info(message)
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Soft-prune core
# ---------------------------------------------------------------------------


def _retain_num_from_rho(expected: int, rho: float) -> int:
    """Compute the kept-token budget from a drop ratio ``rho``.

    Mirrors the spirit of ``vllm.multimodal.evs.compute_retained_tokens_count``
    but does not require it to be importable - we always keep at least one
    frame worth of tokens when ``expected > 0``.
    """
    if expected <= 0:
        return 0
    keep_ratio = max(0.0, min(1.0, 1.0 - float(rho)))
    return max(0, min(expected, int(round(expected * keep_ratio))))


def _soft_prune_one_video(
    emb,
    grid_thw_size: Sequence[int],
    spatial_merge_size: int,
):
    """Apply SURGE / CDPruner to a single video's ``[N, D]`` embeddings.

    Returns a tensor of the **same shape** as ``emb`` with dropped patches
    zeroed out. Failures fall back to returning ``emb`` unchanged.
    """
    import torch  # type: ignore

    from .config import get_runtime_config
    from .context import get_instruction

    cfg = get_runtime_config()
    if not cfg.enabled or cfg.rho <= 0.0:
        return emb

    if emb is None or not isinstance(emb, torch.Tensor) or emb.dim() < 2:
        return emb

    try:
        T = int(grid_thw_size[0])
        H = int(grid_thw_size[1])
        W = int(grid_thw_size[2])
    except Exception:
        return emb

    if spatial_merge_size is None or spatial_merge_size <= 0:
        return emb

    Hm = H // spatial_merge_size
    Wm = W // spatial_merge_size
    m = Hm * Wm
    expected = T * m

    if expected == 0:
        return emb
    if emb.shape[0] != expected:
        # Some Qwen variants emit different layouts (e.g. before merger).
        # We deliberately bail out instead of guessing.
        if cfg.debug:
            _debug_print(
                "soft-prune: shape mismatch "
                f"emb={tuple(emb.shape)} expected={expected} "
                f"(T={T} Hm={Hm} Wm={Wm}); skip"
            )
        return emb

    retain_num = _retain_num_from_rho(expected, cfg.rho)
    if retain_num >= expected:
        return emb
    if retain_num <= 0:
        # Force-keep at least one frame so the model still sees something.
        retain_num = min(expected, m if m > 0 else 1)

    try:
        if cfg.algo == "surge":
            from .surge import SurgeConfig, run_surge

            surge_cfg = SurgeConfig(keep_first_frame=cfg.keep_first_frame)
            mask = run_surge(emb, T, Hm, Wm, retain_num, surge_cfg)
        else:
            from .cdpruner import CDPrunerConfig, run_cdpruner

            cdp_cfg = CDPrunerConfig(
                clip_model=cfg.clip_model,
                keep_first_frame=cfg.keep_first_frame,
            )
            instruction = get_instruction()
            mask = run_cdpruner(emb, T, Hm, Wm, retain_num, instruction, cdp_cfg)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "soft-prune: %s failed (%s); leaving video embeddings untouched",
            cfg.algo,
            exc,
        )
        return emb

    if not isinstance(mask, torch.Tensor) or mask.numel() != expected:
        logger.warning(
            "soft-prune: %s produced mask with wrong shape %s (expected %d); skip",
            cfg.algo,
            tuple(getattr(mask, "shape", ())) or "?",
            expected,
        )
        return emb

    mask_bool = mask.to(torch.bool).to(emb.device)
    if mask_bool.all():
        return emb

    # Use multiplication instead of in-place mutation to avoid disturbing any
    # autograd / cache graph that might wrap ``emb``.
    multiplier = mask_bool.to(emb.dtype).unsqueeze(-1)
    pruned = emb * multiplier

    if cfg.debug:
        kept = int(mask_bool.sum().item())
        msg = (
            f"soft-prune: algo={cfg.algo} T={T} Hm={Hm} Wm={Wm} "
            f"kept={kept}/{expected}"
        )
        _debug_print(msg)
    return pruned


# ---------------------------------------------------------------------------
# _process_video_input wrapper
# ---------------------------------------------------------------------------


def _normalize_results(
    result: Any,
) -> tuple[list, str]:
    """Split a ``_process_video_input`` return into a list + container kind.

    The Qwen2-VL / Qwen2.5-Omni / Qwen3-Omni implementations return either a
    ``tuple[torch.Tensor, ...]``, a ``list[torch.Tensor]`` or a single
    ``torch.Tensor``. We standardise on a list while remembering the original
    container so the wrapper can restore it.
    """
    import torch  # type: ignore

    if isinstance(result, tuple):
        return list(result), "tuple"
    if isinstance(result, list):
        return list(result), "list"
    if isinstance(result, torch.Tensor):
        # Qwen2.5-Omni Mixin returns a single concatenated tensor when the
        # caller fed a ``video_embeds`` payload; treat the whole tensor as one
        # video for masking purposes.
        return [result], "tensor"
    return [], "other"


def _restore_container(tensors: list, kind: str) -> Any:
    if kind == "tuple":
        return tuple(tensors)
    if kind == "list":
        return tensors
    if kind == "tensor":
        return tensors[0] if tensors else None
    return tensors


def _grid_thw_list(grid_thw: Any) -> list[list[int]]:
    if grid_thw is None:
        return []
    if hasattr(grid_thw, "tolist"):
        out = grid_thw.tolist()
    elif isinstance(grid_thw, (list, tuple)):
        out = list(grid_thw)
    else:
        return []
    if out and isinstance(out[0], (int, float)):
        # A single [T, H, W] row.
        return [list(map(int, out))]
    return [list(map(int, row)) for row in out]


def _get_video_grid_thw(video_input: Any, debug: bool) -> Any:
    """Read ``video_grid_thw`` from vLLM's video input container.

    vLLM model inputs are usually ``TypedDict`` instances, but some execution
    paths wrap them in mapping-like containers. Avoid a strict ``dict`` check
    so the Qwen3-Omni worker path can still see the grid metadata.
    """
    if hasattr(video_input, "get"):
        for key in ("video_grid_thw", "grid_thw"):
            try:
                value = video_input.get(key)  # type: ignore[attr-defined]
            except Exception:
                value = None
            if value is not None:
                return value

    for key in ("video_grid_thw", "grid_thw"):
        value = getattr(video_input, key, None)
        if value is not None:
            return value

    if debug:
        keys = None
        if hasattr(video_input, "keys"):
            try:
                keys = list(video_input.keys())  # type: ignore[attr-defined]
            except Exception:
                keys = None
        _debug_print(
            "soft-prune: video_grid_thw missing; skip "
            f"type={type(video_input).__name__} keys={keys}"
        )
    return None


def _wrap_process_video_input(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, video_input, *args, **kwargs):
        result = original(self, video_input, *args, **kwargs)

        from .config import get_runtime_config

        cfg = get_runtime_config()
        if not cfg.enabled or cfg.rho <= 0.0:
            return result

        # Resolve spatial_merge_size from the visual tower; fall back to
        # whatever attribute the model exposes.
        spatial_merge_size = None
        visual = getattr(self, "visual", None)
        if visual is not None:
            spatial_merge_size = getattr(visual, "spatial_merge_size", None)
        if spatial_merge_size is None:
            cfg_obj = getattr(self, "config", None)
            vision_cfg = getattr(cfg_obj, "vision_config", None)
            spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", None)
        if spatial_merge_size is None or int(spatial_merge_size) <= 0:
            if cfg.debug:
                _debug_print("soft-prune: cannot resolve spatial_merge_size; skip")
            return result

        grid_thw_raw = _get_video_grid_thw(video_input, cfg.debug)
        if grid_thw_raw is None:
            return result

        grid_list = _grid_thw_list(grid_thw_raw)
        if not grid_list:
            if cfg.debug:
                _debug_print("soft-prune: empty video_grid_thw; skip")
            return result

        tensors, kind = _normalize_results(result)
        if not tensors:
            if cfg.debug:
                _debug_print(
                    f"soft-prune: wrapper entered but no tensor output (kind={kind}); skip"
                )
            return result

        if cfg.debug:
            _debug_print(
                "soft-prune: wrapper entered "
                f"kind={kind} tensors={len(tensors)} grids={len(grid_list)} "
                f"spatial_merge_size={int(spatial_merge_size)}"
            )

        # Cardinality mismatch is treated as "unsafe to prune".
        if len(tensors) != len(grid_list) and kind != "tensor":
            if cfg.debug:
                _debug_print(
                    "soft-prune: video count mismatch "
                    f"({len(tensors)} tensors vs {len(grid_list)} grids); skip"
                )
            return result

        try:
            pruned_tensors: list = []
            for idx, emb in enumerate(tensors):
                if kind == "tensor":
                    # Concatenated case: derive a synthetic grid by summing.
                    # We zero out per-video grids in order to mimic
                    # ``_process_video_input``'s usual split-by-size shape.
                    if len(grid_list) == 1:
                        pruned_tensors.append(
                            _soft_prune_one_video(
                                emb, grid_list[0], int(spatial_merge_size)
                            )
                        )
                    else:
                        # Walk through grids and prune contiguous chunks.
                        cursor = 0
                        chunks: list = []
                        for size in grid_list:
                            T = int(size[0])
                            H = int(size[1])
                            W = int(size[2])
                            Hm = H // int(spatial_merge_size)
                            Wm = W // int(spatial_merge_size)
                            n = T * Hm * Wm
                            chunk = emb[cursor : cursor + n]
                            cursor += n
                            if chunk.shape[0] == 0:
                                chunks.append(chunk)
                                continue
                            chunks.append(
                                _soft_prune_one_video(
                                    chunk, size, int(spatial_merge_size)
                                )
                            )
                        if cursor != emb.shape[0]:
                            # Bail: layout assumption broken; leave untouched.
                            if cfg.debug:
                                _debug_print(
                                    "soft-prune: concat layout "
                                    f"cursor={cursor} total={emb.shape[0]}; skip"
                                )
                            return result
                        import torch  # type: ignore

                        pruned_tensors.append(torch.cat(chunks, dim=0))
                else:
                    pruned_tensors.append(
                        _soft_prune_one_video(
                            emb, grid_list[idx], int(spatial_merge_size)
                        )
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("soft-prune wrapper failed: %s", exc)
            return result

        return _restore_container(pruned_tensors, kind)

    return wrapper


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


# (module_name, class_name, method_name) targets that are known NOT to run
# vLLM's EVS path. EVS-capable classes intentionally stay off this list so we
# do not double-prune their embeddings.
_TARGETS: tuple[tuple[str, str, str], ...] = (
    # Qwen2-VL family
    (
        "vllm.model_executor.models.qwen2_vl",
        "Qwen2VLForConditionalGeneration",
        "_process_video_input",
    ),
    # Qwen2.5-Omni Thinker (mixin shared with Qwen3-Omni MoE Thinker)
    (
        "vllm.model_executor.models.qwen2_5_omni_thinker",
        "Qwen2_5OmniConditionalGenerationMixin",
        "_process_video_input",
    ),
)


def _try_patch_method(mod_name: str, cls_name: str, method_name: str) -> bool:
    mod = sys.modules.get(mod_name)
    if mod is None:
        try:
            __import__(mod_name)
            mod = sys.modules.get(mod_name)
        except Exception:
            return False
    if mod is None:
        return False
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return False
    key = f"{mod_name}.{cls_name}.{method_name}"
    if key in _orig_methods:
        return True
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    try:
        setattr(cls, method_name, _wrap_process_video_input(original))
    except Exception:  # pragma: no cover - defensive
        return False
    _orig_methods[key] = original
    return True


def install() -> bool:
    """Patch known non-EVS Qwen video-processing classes in-place."""
    global _PATCHED
    if _PATCHED:
        return True
    any_hit = False
    for mod_name, cls_name, method_name in _TARGETS:
        if _try_patch_method(mod_name, cls_name, method_name):
            any_hit = True
            logger.info(
                "xstream_vllm_pruner: patched %s.%s.%s",
                mod_name,
                cls_name,
                method_name,
            )
    _PATCHED = any_hit
    return any_hit


def uninstall() -> bool:
    global _PATCHED
    if not _PATCHED:
        return False
    for key, original in list(_orig_methods.items()):
        try:
            head, _, attr = key.rpartition(".")
            mod_head, _, cls_head = head.rpartition(".")
            mod = sys.modules.get(mod_head)
            if mod is None:
                continue
            cls = getattr(mod, cls_head, None)
            if cls is None:
                continue
            setattr(cls, attr, original)
        except Exception:  # pragma: no cover
            continue
    _orig_methods.clear()
    _PATCHED = False
    return True


def is_installed() -> bool:
    return _PATCHED


def install_for_class(cls: Any, method_name: str = "_process_video_input") -> bool:
    """Public helper for out-of-tree models with the same contract."""
    mod_name = getattr(cls, "__module__", "<unknown>")
    cls_name = cls.__name__
    key = f"{mod_name}.{cls_name}.{method_name}"
    if key in _orig_methods:
        return True
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    setattr(cls, method_name, _wrap_process_video_input(original))
    _orig_methods[key] = original
    return True
