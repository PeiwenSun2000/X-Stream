"""Monkey-patch the Qwen multimodal processors to relay ``xstream_instruction``.

We accept the per-request instruction text via vLLM's existing
``mm_processor_kwargs`` channel, then bind it to a ``ContextVar`` so the
patched ``compute_retention_mask`` can consume it. The key has to be popped
out of ``mm_kwargs`` before delegating to the HF processor, because HF
processors raise on unknown kwargs.

Note on transport semantics: ``ContextVar`` propagates within the same
process. Single-process deployments (e.g. ``--tp 1`` without an isolated
worker) therefore get the full conditional CDPruner relevance term. With
multi-process workers the instruction simply degrades to an empty string and
CDPruner falls back to visual-only diversity, which is documented in
``README.md``.
"""
from __future__ import annotations

import functools
import logging
import sys
from typing import Any, Callable

from .context import set_instruction

logger = logging.getLogger("xstream_vllm_pruner.patch_processor")

_PATCHED = False
_orig_methods: dict[str, Callable] = {}


def _wrap_call_hf_processor(original: Callable) -> Callable:
    """Strip ``xstream_instruction`` from ``mm_kwargs`` and set the ContextVar."""

    @functools.wraps(original)
    def wrapper(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        instruction = None
        if isinstance(mm_kwargs, dict) and "xstream_instruction" in mm_kwargs:
            instruction = mm_kwargs.pop("xstream_instruction")
        elif hasattr(mm_kwargs, "pop") and "xstream_instruction" in mm_kwargs:  # Mapping
            try:
                instruction = mm_kwargs.pop("xstream_instruction")  # type: ignore[arg-type]
            except Exception:
                instruction = None

        token = set_instruction(instruction if isinstance(instruction, str) else "")
        try:
            return original(self, prompt, mm_data, mm_kwargs, tok_kwargs)
        finally:
            # Best-effort reset; reset can fail across thread boundaries.
            try:
                from .context import reset_instruction

                reset_instruction(token)
            except Exception:  # pragma: no cover - defensive
                pass

    return wrapper


def _try_patch_class(mod_name: str, cls_name: str) -> bool:
    mod = sys.modules.get(mod_name)
    if mod is None:
        return False
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return False
    key = f"{mod_name}.{cls_name}._call_hf_processor"
    if key in _orig_methods:
        return True
    original = getattr(cls, "_call_hf_processor", None)
    if original is None:
        return False
    wrapped = _wrap_call_hf_processor(original)
    try:
        setattr(cls, "_call_hf_processor", wrapped)
    except Exception:  # pragma: no cover - defensive
        return False
    _orig_methods[key] = original
    return True


def _try_import_and_patch(mod_name: str, cls_name: str) -> bool:
    if _try_patch_class(mod_name, cls_name):
        return True
    try:
        __import__(mod_name)
    except Exception:
        return False
    return _try_patch_class(mod_name, cls_name)


def install() -> bool:
    """Patch known Qwen2.5-VL / Qwen3-VL multimodal processors in-place."""
    global _PATCHED
    if _PATCHED:
        return True

    targets = [
        # EVS-capable VL families.
        ("vllm.model_executor.models.qwen2_vl", "Qwen2VLMultiModalProcessor"),
        ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5_VLMultiModalProcessor"),
        ("vllm.model_executor.models.qwen3_vl", "Qwen3VLMultiModalProcessor"),
        ("vllm.model_executor.models.qwen3_vl_moe", "Qwen3VLMoEMultiModalProcessor"),
        # Omni Thinker families (soft prune path).
        (
            "vllm.model_executor.models.qwen2_5_omni_thinker",
            "Qwen2_5OmniThinkerMultiModalProcessor",
        ),
        (
            "vllm.model_executor.models.qwen3_omni_moe_thinker",
            "Qwen3OmniMoeThinkerMultiModalProcessor",
        ),
    ]
    any_hit = False
    for mod_name, cls_name in targets:
        if _try_import_and_patch(mod_name, cls_name):
            any_hit = True
            logger.info("xstream_vllm_pruner: patched %s.%s", mod_name, cls_name)

    _PATCHED = any_hit
    if not any_hit:
        logger.warning(
            "xstream_vllm_pruner: no known multimodal processor classes were found; "
            "instruction relay will be unavailable for this run"
        )
    return any_hit


def uninstall() -> bool:
    """Restore the original processor methods."""
    global _PATCHED
    if not _PATCHED:
        return False
    for key, original in list(_orig_methods.items()):
        try:
            mod_name, cls_name, _ = key.split(".", 2)[0], key.split(".")[-2], None
            # Rebuild ``module.Class`` from key.
            head, _, _attr = key.rpartition(".")
            mod_head, _, cls_head = head.rpartition(".")
            mod = sys.modules.get(mod_head)
            if mod is None:
                continue
            cls = getattr(mod, cls_head, None)
            if cls is None:
                continue
            setattr(cls, "_call_hf_processor", original)
        except Exception:  # pragma: no cover
            continue
    _orig_methods.clear()
    _PATCHED = False
    return True


def is_installed() -> bool:
    return _PATCHED


def install_for_class(cls: Any) -> bool:
    """Public helper: patch an arbitrary processor class.

    Useful for forks / out-of-tree models that mirror the Qwen processor
    contract.
    """
    mod_name = getattr(cls, "__module__", "<unknown>")
    cls_name = cls.__name__
    key = f"{mod_name}.{cls_name}._call_hf_processor"
    if key in _orig_methods:
        return True
    original = getattr(cls, "_call_hf_processor", None)
    if original is None:
        return False
    setattr(cls, "_call_hf_processor", _wrap_call_hf_processor(original))
    _orig_methods[key] = original
    return True
