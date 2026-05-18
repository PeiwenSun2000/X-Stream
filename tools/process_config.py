#!/usr/bin/env python3
"""Inject local vLLM endpoints into a models.json for a single logical model.

Reads the source config, locates the entry for ``model_name`` (creating a
default celie-adapter entry if missing), then replaces it with one entry per
port pointing to ``http://localhost:<port>/v1/chat/completions``. The
``model_name`` of every replicated entry is forced to match the logical name so
vLLM ``--served-model-name`` and the request body agree.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <input_config> <model_name> <port_list> <output_config>",
            file=sys.stderr,
        )
        return 1

    src, model_name, port_list, dst = sys.argv[1:5]
    with open(src, "r", encoding="utf-8") as f:
        config = json.load(f)

    ports = [p.strip() for p in port_list.split(";") if p.strip()]
    if not ports:
        print(f"invalid port list: {port_list!r}", file=sys.stderr)
        return 1

    if model_name not in config or not config[model_name]:
        config[model_name] = [{
            "adapter": "celie",
            "model_name": model_name,
            "endpoint": "http://localhost:8901/v1/chat/completions",
        }]

    template = config[model_name][0].copy()
    config[model_name] = [
        {
            **template,
            "model_name": model_name,
            "endpoint": f"http://localhost:{port}/v1/chat/completions",
            "is_vllm_local": True,
        }
        for port in ports
    ]

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
