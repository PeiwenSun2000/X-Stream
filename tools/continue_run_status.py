#!/usr/bin/env python3
"""Locate completed or resumable runs that match the current environment.

Subcommands:
  status      print "complete\\t<dir>" / "resume\\t<dir>" / "none\\t" (rc 0)
  resume-dir  print only the resumable run dir (rc 0); rc 1 if none / completed

Match keys come from environment variables and are compared against the
``run_env.json`` written by ``pipeline.sh`` on every run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_MATCH_KEYS = (
    "RUN_ID",
    "VLLM_SERVE_MODEL",
    "VLLM_MODEL_NAME",
    "FLOW_INPUT",
    "FLOW_MULTI_STREAM_MODE",
    "FLOW_PROMPT_ROOT",
    "FLOW_REPLACEMENT",
)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _line_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _current_env() -> dict[str, str]:
    return {k: os.environ.get(k, "") for k in _MATCH_KEYS}


def _matches(data: dict, current: dict[str, str]) -> bool:
    for key, value in current.items():
        if value and str(data.get(key, "")) != value:
            return False
    return True


def _has_eval_marker(run_dir: Path, data: dict) -> bool:
    candidates = [run_dir / "eval.json"]
    for stem_src in (str(data.get("FLOW_INPUT", "")), str(data.get("FLOW_OUTPUT", ""))):
        stem = Path(stem_src).stem
        if stem:
            candidates += [run_dir / f"{stem}.json", run_dir / f"eval_{stem}.json"]
    return any(p.is_file() and p.stat().st_size > 0 for p in candidates)


def _input_done(data: dict) -> bool:
    flow_input = Path(str(data.get("FLOW_INPUT", "")))
    flow_output = Path(str(data.get("FLOW_OUTPUT", "")))
    if not flow_input.is_file() or not flow_output.is_file():
        return False
    n_in = _line_count(flow_input)
    n_out = _line_count(flow_output)
    return n_in is not None and n_out is not None and n_out >= n_in


def _is_complete(run_dir: Path, data: dict) -> bool:
    if not _input_done(data):
        return False
    if os.environ.get("ENABLE_STREAM_EVAL", "true") == "false":
        return True
    return _has_eval_marker(run_dir, data)


def _matching_runs() -> list[tuple[Path, dict, bool]]:
    output_dir = Path(os.environ.get("FLOW_OUTPUT_DIR", "")).expanduser()
    if not output_dir.is_dir():
        return []
    current = _current_env()
    runs: list[tuple[Path, dict, bool]] = []
    for env_path in output_dir.glob("*/run_env.json"):
        data = _read_json(env_path)
        if not data or not _matches(data, current):
            continue
        run_dir = env_path.parent
        runs.append((run_dir, data, _is_complete(run_dir, data)))
    runs.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return runs


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    runs = _matching_runs()

    complete = next((d for d, _, ok in runs if ok), None)
    if complete is not None:
        if mode == "resume-dir":
            return 1
        print(f"complete\t{complete}")
        return 0

    resumable = next((d for d, _, ok in runs if not ok), None)
    if resumable is not None:
        if mode == "resume-dir":
            print(resumable)
        else:
            print(f"resume\t{resumable}")
        return 0

    if mode == "resume-dir":
        return 1
    print("none\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
