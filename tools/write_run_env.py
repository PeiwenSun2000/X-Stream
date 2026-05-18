#!/usr/bin/env python3
"""Write alternating key/value pairs to a JSON file.

Usage: write_run_env.py <output_path> <key1> <value1> <key2> <value2> ...
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 2 or (len(sys.argv) - 2) % 2 != 0:
        print(
            f"Usage: {sys.argv[0]} <output_path> <key> <value> [<key> <value> ...]",
            file=sys.stderr,
        )
        return 1
    out = sys.argv[1]
    rest = sys.argv[2:]
    data = {rest[i]: rest[i + 1] for i in range(0, len(rest), 2)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
