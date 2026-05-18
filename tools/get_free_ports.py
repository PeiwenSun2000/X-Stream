#!/usr/bin/env python3
"""Print N free TCP ports as a semicolon-separated list.

Avoids ports stuck in TIME_WAIT and verifies a successful bind+listen.
"""
from __future__ import annotations

import random
import socket
import subprocess
import sys


def _has_time_wait(port: int) -> bool:
    try:
        p = subprocess.run(
            ["ss", "-tan", "state", "time-wait", "sport", "=", f":{port}"],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except Exception:
        return False
    return p.returncode == 0 and len([ln for ln in p.stdout.splitlines() if ln.strip()]) > 1


def _can_listen(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: get_free_ports.py <n>", file=sys.stderr)
        return 1
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("n must be a positive integer", file=sys.stderr)
        return 1
    if n < 1:
        print("n must be a positive integer", file=sys.stderr)
        return 1

    lo, hi = 5000, 65000
    chosen: list[int] = []
    for _ in range((hi - lo) * 2):
        if len(chosen) >= n:
            break
        port = random.randint(lo, hi)
        if port in chosen or _has_time_wait(port) or not _can_listen(port):
            continue
        chosen.append(port)

    if len(chosen) < n:
        print("could not find enough free ports", file=sys.stderr)
        return 1
    print(";".join(map(str, chosen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
