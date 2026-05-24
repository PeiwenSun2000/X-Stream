"""Runtime configuration for the xstream_vllm_pruner plugin.

All settings are read from environment variables so the plugin can be installed
into a vLLM worker process without modifying any library code. When the
``XSTREAM_VLLM_PRUNER`` env var is not set, every public helper here reports
``enabled=False`` and the plugin must be a strict no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in _TRUE_STRINGS


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    """Snapshot of plugin configuration sampled from the environment."""

    enabled: bool
    algo: str  # "surge" or "cdpruner"
    rho: float
    keep_first_frame: bool
    debug: bool
    clip_model: str

    def __post_init__(self) -> None:
        # frozen dataclass cannot mutate, validation only.
        if self.algo not in ("surge", "cdpruner"):
            raise ValueError(
                f"XSTREAM_VLLM_PRUNER_ALGO must be 'surge' or 'cdpruner', got {self.algo!r}"
            )
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(
                f"XSTREAM_VLLM_PRUNER_RHO must be in [0, 1), got {self.rho}"
            )


def get_runtime_config() -> RuntimeConfig:
    """Read the current plugin runtime configuration from the environment.

    The configuration is intentionally sampled per call so test harnesses or
    upstream tools can flip env vars at runtime without restarting the worker.
    """
    enabled = _env_bool("XSTREAM_VLLM_PRUNER", default=False)
    algo = os.environ.get("XSTREAM_VLLM_PRUNER_ALGO", "surge").strip().lower() or "surge"
    rho = _env_float("XSTREAM_VLLM_PRUNER_RHO", 0.25)
    rho = max(0.0, min(0.9999, rho))
    keep_first_frame = _env_bool("XSTREAM_VLLM_PRUNER_KEEP_FIRST_FRAME", default=True)
    debug = _env_bool("XSTREAM_VLLM_PRUNER_DEBUG", default=False)
    clip_model = os.environ.get(
        "XSTREAM_VLLM_PRUNER_CLIP_MODEL", "openai/clip-vit-large-patch14-336"
    ).strip()
    return RuntimeConfig(
        enabled=enabled,
        algo=algo if algo in ("surge", "cdpruner") else "surge",
        rho=rho,
        keep_first_frame=keep_first_frame,
        debug=debug,
        clip_model=clip_model,
    )


def get_active_algo() -> Optional[str]:
    """Return the active algorithm name, or ``None`` when the plugin is off."""
    cfg = get_runtime_config()
    return cfg.algo if cfg.enabled else None
