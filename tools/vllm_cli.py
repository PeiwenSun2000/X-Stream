#!/usr/bin/env python3
"""Thin wrapper around ``vllm`` CLI that registers HuggingFace tokenizer aliases.

Some checkpoints declare ``tokenizer_class: TokenizersBackend`` in
``tokenizer_config.json``, which upstream ``transformers`` does not provide.
We patch ``transformers.models.auto.tokenization_auto.tokenizer_class_from_name``
to map known aliases to real classes (default: ``TokenizersBackend`` ->
``Qwen2TokenizerFast``) before ``vllm`` imports its tokenizer machinery.

Extra aliases can be supplied via the ``VLLMFLOW_TOKENIZER_ALIASES`` env var,
e.g. ``{"FooBar": "Qwen2TokenizerFast"}`` (value must be a transformers class
resolvable by ``tokenizer_class_from_name``).
"""
from __future__ import annotations

import json
import os
import sys

from transformers import Qwen2TokenizerFast
import transformers.models.auto.tokenization_auto as _tokenization_auto

_orig = _tokenization_auto.tokenizer_class_from_name

_ALIASES: dict[str, type] = {"TokenizersBackend": Qwen2TokenizerFast}

_raw = os.environ.get("VLLMFLOW_TOKENIZER_ALIASES", "").strip()
if _raw:
    try:
        for name, target in json.loads(_raw).items():
            cls = _orig(target)
            if cls is None:
                raise ValueError(f"unknown tokenizer class in alias: {target!r}")
            _ALIASES[str(name)] = cls
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"vllm_cli: invalid VLLMFLOW_TOKENIZER_ALIASES: {e}", file=sys.stderr)
        sys.exit(2)


def _patched(class_name: str):
    if class_name in _ALIASES:
        return _ALIASES[class_name]
    return _orig(class_name)


_tokenization_auto.tokenizer_class_from_name = _patched

from vllm.entrypoints.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
