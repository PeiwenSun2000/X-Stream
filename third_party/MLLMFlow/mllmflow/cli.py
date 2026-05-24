#!/usr/bin/env python3
"""MLLMFlow CLI"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from mllmflow import MLLMFlow
from mllmflow.utils import _parse_args_str
from mllmflow.video_utils.probe_video import probe_video


_VIDEO_PROBE_CACHE = {}
_VIDEO_PROBE_CACHE_LOCK = threading.Lock()


def parse_model_replacement(s: str) -> dict:
    """Parse a model replacement string, such as 'gpt-4o>gemini-pro-3-preview'"""
    if not s:
        return {}
    replacements = {}
    for pair in s.split(","):
        if ">" in pair:
            old, new = pair.split(">", 1)
            replacements[old.strip()] = new.strip()
    return replacements


def _join_root(root: str, path: str) -> str:
    """Join path under root only when path is relative; return absolute paths unchanged."""
    root = (root or "").strip()
    path = path.strip()
    if not root or not path:
        return path
    if path.startswith("/"):
        return path
    return os.path.normpath(os.path.join(root, path))


def modify_template(
    template: Dict[str, Any],
    prompt_root: str = "",
    image_root: str = "",
    video_root: str = "",
) -> Dict[str, Any]:
    """Resolve relative file/image/video placeholders in the template under their roots without changing absolute paths."""
    import copy

    template = copy.deepcopy(template)
    pr = (prompt_root or "").strip()
    ir = (image_root or "").strip()
    vr = (video_root or "").strip()

    for round_data in template.get("rounds", []):
        for msg in round_data.get("messages", []):
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            if pr:

                def repl_file(m: re.Match) -> str:
                    p = m.group(1).strip()
                    return f"{{{{file:{_join_root(pr, p)}}}}}"

                content = re.sub(r"\{\{file:([^}]+)\}\}", repl_file, content)

            if ir:

                def repl_image(m: re.Match) -> str:
                    path_part = m.group(1).strip()
                    rest = m.group(2) or ""
                    joined = _join_root(ir, path_part)
                    return f"{{{{image:{joined}{rest}}}}}"

                content = re.sub(r"\{\{image:([^,}]+)([^}]*)\}\}", repl_image, content)

            if vr:

                def repl_video(m: re.Match) -> str:
                    inner = m.group(1)
                    parts = inner.split(",", 1)
                    path0 = parts[0].strip()
                    tail = f",{parts[1]}" if len(parts) > 1 else ""
                    joined = _join_root(vr, path0)
                    return f"{{{{video:{joined}{tail}}}}}"

                content = re.sub(r"\{\{video:([^}]+)\}\}", repl_video, content)

            msg["content"] = content

    return template


def _template_id(template: Dict[str, Any]) -> str:
    round_ids = [
        str(round_data.get("round_id", "round"))
        for round_data in template.get("rounds", [])
        if isinstance(round_data, dict)
    ]
    return ",".join(round_ids[:3]) or "<unknown>"


def _iter_template_video_paths(template: Dict[str, Any]):
    for round_data in template.get("rounds", []):
        for msg in round_data.get("messages", []):
            content = msg.get("content", "")
            if not isinstance(content, str) or "{{video:" not in content:
                continue
            for match in re.finditer(r"\{\{video:([^}]+)\}\}", content):
                resource, _ = _parse_args_str(match.group(1))
                yield resource


def _probe_video_cached(video_path: str):
    with _VIDEO_PROBE_CACHE_LOCK:
        if video_path in _VIDEO_PROBE_CACHE:
            cached = _VIDEO_PROBE_CACHE[video_path]
            if isinstance(cached, Exception):
                raise cached
            return cached

    path = Path(video_path)
    if not path.exists():
        result = FileNotFoundError(f"FileNotFile {video_path}")
    else:
        info = probe_video(video_path)
        result = info if info is not None else ValueError(f"Failed to probe original video: {video_path}")

    with _VIDEO_PROBE_CACHE_LOCK:
        _VIDEO_PROBE_CACHE[video_path] = result

    if isinstance(result, Exception):
        raise result
    return result


def validate_template_videos(template: Dict[str, Any]) -> None:
    """Fail fast before MoviePy starts trimming if any referenced video is missing or invalid."""
    checked = set()
    for video_path in _iter_template_video_paths(template):
        if video_path in checked:
            continue
        checked.add(video_path)
        _probe_video_cached(video_path)


def handle_template(template, args):
    # Modify the template (path replacement)
    template = modify_template(
        template, args.prompt_root, args.image_root, args.video_root
    )

    model_replacement = parse_model_replacement(args.model_replacement)
    flow = MLLMFlow(
        models_config=args.model_config,
        cache_dir=args.cache_dir,
        model_replacement=model_replacement,
        multi_stream_mode=getattr(args, "multi_stream_mode", "pixel"),
        memory_bank=getattr(args, "memory_bank", False),
        memory_bank_model=getattr(args, "memory_bank_model", ""),
        memory_bank_log_dir=str(Path(args.output).parent / "memory_bank_logs"),
    )

    result = flow.run(template)

    log_dir = Path(args.output).parent / "mllmflow_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    template_id = (template.get("rounds") or [{}])[0].get("round_id", "templ")

    def get_log_name(name: str) -> str:
        return f"{name}_{int(time.time())}_{str(uuid.uuid4())[0:8]}.json"

    with open(log_dir / get_log_name(str(template_id)), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def warm_template_video_cache(template, args):
    try:
        template = modify_template(
            template, args.prompt_root, args.image_root, args.video_root
        )
        validate_template_videos(template)
        model_replacement = parse_model_replacement(args.model_replacement)
        flow = MLLMFlow(
            models_config=args.model_config,
            cache_dir=args.cache_dir,
            model_replacement=model_replacement,
            multi_stream_mode=getattr(args, "multi_stream_mode", "pixel"),
        )
        return flow.warm_video_cache(template)
    except Exception as exc:
        template_id = _template_id(template)
        error = f"{type(exc).__name__}: {exc}"
        print(f"cache warm skipped: template={template_id} error={error}", file=sys.stderr, flush=True)
        return {
            "video_messages": 0,
            "video_segments": 0,
            "skipped": 1,
            "template_id": template_id,
            "error": error,
        }


def _run_cache_warm(args) -> int:
    workers = int(getattr(args, "cache_warm_workers", 0) or args.n_workers or 1)
    workers = max(1, workers)
    pending_limit = max(1, workers * 2)
    progress_interval = max(1, min(16, workers))
    total_templates = 0
    total_video_messages = 0
    total_video_segments = 0
    total_skipped = 0
    skipped_errors = []

    def iter_templates():
        if args.input.endswith(".jsonl"):
            for chunk in iter_jsonl_chunks(args.input, chunk_size=pending_limit):
                yield from chunk
        else:
            yield from load_jsonl_or_json(args.input)

    def record_result(result):
        nonlocal total_templates, total_video_messages, total_video_segments, total_skipped
        total_templates += 1
        total_video_messages += result.get("video_messages", 0)
        total_video_segments += result.get("video_segments", 0)
        total_skipped += result.get("skipped", 0)
        if result.get("skipped"):
            skipped_errors.append(
                {"template_id": result.get("template_id", "<unknown>"), "error": result.get("error", "")}
            )

    def print_progress():
        print(
            f"cache warm progress: templates={total_templates} "
            f"video_messages={total_video_messages} video_segments={total_video_segments} "
            f"skipped={total_skipped}",
            flush=True,
        )

    templates = iter_templates()
    if workers == 1:
        for template in templates:
            record_result(warm_template_video_cache(template, args))
            if total_templates % progress_interval == 0:
                print_progress()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            exhausted = False

            def fill_pending():
                nonlocal exhausted
                while not exhausted and len(pending) < pending_limit:
                    try:
                        template = next(templates)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(executor.submit(warm_template_video_cache, template, args))

            fill_pending()
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    record_result(future.result())
                    if total_templates % progress_interval == 0:
                        print_progress()
                fill_pending()

    if total_templates % progress_interval != 0:
        print_progress()

    max_errors_in_summary = 200
    summary = {
        "mode": "warm_cache_only",
        "workers": workers,
        "templates": total_templates,
        "video_messages": total_video_messages,
        "video_segments": total_video_segments,
        "skipped": total_skipped,
        "skipped_errors": skipped_errors[:max_errors_in_summary],
        "skipped_errors_truncated": max(0, len(skipped_errors) - max_errors_in_summary),
        "cache_dir": args.cache_dir,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"cache warm done -> {args.output}")
    return 0

def load_jsonl_or_json(filename):
    assert filename.endswith(".json") or filename.endswith(".jsonl")

    if filename.endswith(".jsonl"):
        templates = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip():
                    templates.append(json.loads(line))
        return templates
    else:
        with open(filename, "r", encoding="utf-8") as f:
            templates = [json.load(f)]
        return templates


def iter_jsonl_chunks(input_path: str, skip: int = 0, chunk_size: int = 64):
    """Stream jsonl in chunks to avoid loading large files into memory at once. Skip the first skip lines, then yield template lists chunk by chunk."""
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in range(skip):
            next(f, None)
        chunk = []
        for line in f:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def dump_jsonl_or_json(data, filename):
    assert filename.endswith(".json") or filename.endswith(".jsonl")

    with open(filename, "w", encoding="utf-8") as f:
        if filename.endswith(".jsonl"):
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False)+"\n")
        else:
            json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="MLLMFlow CLI")
    parser.add_argument("--model-config", required=True, help="Path to the model configuration file")
    parser.add_argument("--input", required=True, help="Path to the input template file")
    parser.add_argument("--output", required=True, help="Path to the output result file (JSON format)")
    parser.add_argument("--model-replacement", default="", help="Model replacement, such as 'gpt-4o>gemini-pro-3-preview'")
    parser.add_argument("--prompt-root", default="", help="Root directory for prompt files")
    parser.add_argument("--image-root", default="", help="Root directory for image assets")
    parser.add_argument("--video-root", default="", help="Root directory for video assets")
    parser.add_argument("--cache-dir", default="media_dir", help="Cache directory")
    parser.add_argument("--n-workers", default=4, type=int, help="Number of worker threads")
    parser.add_argument(
        "--warm-cache-only",
        action="store_true",
        help="Only pre-generate video segment cache; do not call any model",
    )
    parser.add_argument(
        "--cache-warm-workers",
        default=0,
        type=int,
        help="Worker count for --warm-cache-only (default: --n-workers)",
    )
    parser.add_argument(
        "--multi-stream-mode",
        choices=[
            "pixel",
            "time",
            "code",
            "code_adaptive",
            "cdpruner",
            "surge",
            "cdpruner_token",
            "surge_token",
        ],
        default="pixel",
        help=(
            "Multi-stream mode: "
            "pixel=default non-multi-stream; "
            "time=dual streams interleaved by time as A1 B1 A2 B2; "
            "code=for each segment, input the stream with larger changes and replace the other with Stream N: Unchanged; "
            "code_adaptive=adaptively adjusts pixel scale based on the change magnitude of both video streams per segment, using higher pixel scale for the stream with larger changes (up to 2.0x),"
            "and lower pixel scale for the stream with smaller changes (can approach 0). If one stream has no changes, output Unchanged for it and use full pixel scale for the other; "
            "cdpruner=token selection inspired by CDPruner, combined with time multi-stream mode, to prune video tokens for Qwen3-Omni-30B-A3B; "
            "surge=SURGE-style temporal surprise strategy for video token pruning (combined with time multi-stream mode); "
            "cdpruner_token/surge_token=patch-level token pruning inside frames, local vLLM only"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: if output already exists and is jsonl, skip the first N existing results, run only the remaining inputs, and append the results",
    )
    parser.add_argument(
        "--memory-bank",
        action="store_true",
        help="Enable the multi-stream Memory Bank (Stream + Global) and inject distilled memory into the model context",
    )
    parser.add_argument(
        "--memory-bank-model",
        default="",
        help="Model name used by the Memory Bank (defaults to the current model placeholder model)",
    )
    args = parser.parse_args()

    if getattr(args, "warm_cache_only", False):
        return _run_cache_warm(args)

    done_count = 0
    if getattr(args, "resume", False) and args.output.endswith(".jsonl"):
        out_path = Path(args.output)
        if out_path.exists() and out_path.stat().st_size > 0:
            with open(args.output, "r", encoding="utf-8") as f:
                done_count = sum(1 for line in f if line.strip())
            if done_count > 0:
                print(f"Resume: {done_count} lines already in {args.output}, skipping those inputs.")

    # jsonl input + jsonl output: stream in chunks to avoid loading large files (such as 2,974 phostream records) into memory at once and causing OOM
    if args.input.endswith(".jsonl") and args.output.endswith(".jsonl"):
        chunk_size = max(32, min(128, args.n_workers * 16))
        first_chunk = True
        total_written = done_count
        for chunk in iter_jsonl_chunks(args.input, skip=done_count, chunk_size=chunk_size):
            if not chunk:
                continue
            if len(chunk) == 1 or args.n_workers == 1:
                results = [handle_template(tmpl, args) for tmpl in chunk]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                    futures = [executor.submit(handle_template, tmpl, args) for tmpl in chunk]
                    results_sorted = [None] * len(chunk)
                    future_to_idx = {f: idx for idx, f in enumerate(futures)}
                    for future in futures:
                        idx = future_to_idx[future]
                        results_sorted[idx] = future.result()
                    results = results_sorted
            mode = "w" if (done_count == 0 and first_chunk) else "a"
            with open(args.output, mode, encoding="utf-8") as f:
                for x in results:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            total_written += len(results)
            first_chunk = False
        if total_written == done_count and done_count > 0:
            print("All templates already done, nothing to run.")
        else:
            print(f"done -> {args.output} ({total_written} lines)")
        return 0

    # Single-file or .json output: keep the original logic (load the full table)
    templates = load_jsonl_or_json(args.input)
    if not templates:
        print("template is empty", file=sys.stderr)
        return 1

    templates_to_run = templates[done_count:] if done_count < len(templates) else []
    if not templates_to_run:
        if done_count >= len(templates):
            print("All templates already done, nothing to run.")
        else:
            templates_to_run = templates
        if not templates_to_run:
            return 0
    else:
        templates = templates_to_run

    if len(templates) == 1 or args.n_workers == 1:
        results = [handle_template(tmpl, args) for tmpl in templates]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(handle_template, tmpl, args) for tmpl in templates]
            results_sorted = [None] * len(templates)
            future_to_idx = {f: idx for idx, f in enumerate(futures)}
            for future in futures:
                idx = future_to_idx[future]
                results_sorted[idx] = future.result()
            results = results_sorted

    if len(results) == 1:
        result = results[0]
    else:
        result = results

    if done_count > 0 and args.output.endswith(".jsonl"):
        with open(args.output, "a", encoding="utf-8") as f:
            for x in (result if isinstance(result, list) else [result]):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    else:
        dump_jsonl_or_json(result, filename=args.output)

    print(f"done -> {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main() or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
