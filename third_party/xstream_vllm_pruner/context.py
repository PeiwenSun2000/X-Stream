"""Per-request context propagation for the xstream_vllm_pruner plugin.

vLLM's video pruning hook (``compute_retention_mask``) does not receive any
per-request metadata such as the user instruction text. CDPruner needs the
instruction to compute conditional diversity, so we relay it via a
``ContextVar`` set inside the processor patch and consumed inside the
retention-mask patch.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional


# Default to an empty string so callers can always ``.get("")`` without guards.
current_instruction: ContextVar[str] = ContextVar(
    "xstream_vllm_pruner_current_instruction", default=""
)


def set_instruction(text: Optional[str]) -> object:
    """Bind the per-request instruction string. Returns a reset token."""
    return current_instruction.set((text or "").strip())


def reset_instruction(token: object) -> None:
    """Restore the previous instruction binding."""
    try:
        current_instruction.reset(token)  # type: ignore[arg-type]
    except (ValueError, LookupError):
        # Reset may fail across thread boundaries; falling back is harmless.
        current_instruction.set("")


def get_instruction() -> str:
    """Read the current instruction string (defaults to ``""``)."""
    return current_instruction.get("")
