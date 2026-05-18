#!/usr/bin/env python3
"""Print ``export`` statements that re-attach to a previous run directory.

Reads ``<run_dir>/run_env.json`` and emits shell-safe ``export KEY=...``
lines for the keys needed to resume that run. Intended to be consumed via
``eval "$(print_continue_exports.py /path/to/run_dir)"``.
"""
from __future__ import annotations

import json
import os
import sys


_KEYS = ("RUN_DIR", "RUN_ID_WITH_TIMESTAMP", "FLOW_OUTPUT", "FLOW_CONFIG", "RUN_ID")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: print_continue_exports.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = os.path.abspath(sys.argv[1])
    path = os.path.join(run_dir, "run_env.json")
    if not os.path.isfile(path):
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in _KEYS:
        if key in data:
            safe = str(data[key]).replace("'", "'\"'\"'")
            print(f"export {key}='{safe}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
