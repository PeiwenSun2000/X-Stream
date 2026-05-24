"""X-Stream vLLM patch-level pruner plugin.

Installs SURGE / CDPruner as a drop-in replacement for vLLM's built-in EVS
``compute_retention_mask`` so frame-internal patch tokens of Qwen2.5-VL and
Qwen3-VL videos can be pruned without modifying vLLM source files.

The plugin is opt-in via the ``XSTREAM_VLLM_PRUNER=1`` environment variable.
When the variable is not set, ``install()`` is a no-op and the worker keeps
its original EVS behaviour. See ``README.md`` for the full env-var contract.
"""
from __future__ import annotations

import logging
import sys

from .config import (
    RuntimeConfig,
    get_active_algo,
    get_runtime_config,
)
from .context import (
    current_instruction,
    get_instruction,
    reset_instruction,
    set_instruction,
)
from .surge import SurgeConfig
from .cdpruner import CDPrunerConfig
from . import patch_processor, patch_qwen_vl, patch_video_input

logger = logging.getLogger("xstream_vllm_pruner")

__all__ = [
    "RuntimeConfig",
    "SurgeConfig",
    "CDPrunerConfig",
    "current_instruction",
    "get_active_algo",
    "get_instruction",
    "get_runtime_config",
    "install",
    "is_installed",
    "reset_instruction",
    "set_instruction",
    "uninstall",
]


def install() -> bool:
    """Install both the EVS and processor patches.

    Returns ``True`` when the EVS retention-mask patch ends up active.
    Processor patches are best-effort: failing to find them just disables
    instruction relay (CDPruner falls back to visual-only diversity).
    """
    cfg = get_runtime_config()
    if not cfg.enabled:
        logger.info(
            "xstream_vllm_pruner: install() called but XSTREAM_VLLM_PRUNER is off; no-op"
        )
        return False

    ok_mask = patch_qwen_vl.install()
    patch_processor.install()
    ok_video = patch_video_input.install()
    if ok_mask or ok_video:
        msg = (
            "xstream_vllm_pruner: ready "
            f"(algo={cfg.algo} rho={cfg.rho:.3f} "
            f"keep_first_frame={cfg.keep_first_frame} "
            f"evs_hook={'on' if ok_mask else 'off'} "
            f"soft_hook={'on' if ok_video else 'off'})"
        )
        logger.info(msg)
        print(msg, file=sys.stderr, flush=True)
    return ok_mask or ok_video


def uninstall() -> bool:
    """Undo every patch the plugin has applied."""
    a = patch_qwen_vl.uninstall()
    b = patch_processor.uninstall()
    c = patch_video_input.uninstall()
    return bool(a or b or c)


def is_installed() -> bool:
    return patch_qwen_vl.is_installed() or patch_video_input.is_installed()
