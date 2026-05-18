#!/usr/bin/env python3
"""MLLMFlow CLI"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from mllmflow import MLLMFlow


def parse_model_replacement(s: str) -> dict:
    """解析模型替换字符串，如 'gpt-4o>gemini-pro-3-preview'"""
    if not s:
        return {}
    replacements = {}
    for pair in s.split(","):
        if ">" in pair:
            old, new = pair.split(">", 1)
            replacements[old.strip()] = new.strip()
    return replacements


def _join_root(root: str, path: str) -> str:
    """仅在 path 为相对路径时拼到 root 下；已是绝对路径则原样返回。"""
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
    """将模板里相对路径的 file/image/video 占位符解析到各 root 下（不破坏绝对路径）。"""
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


def handle_template(template, args):
    # 修改模板（路径替换）
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
    """流式按块读取 jsonl，避免大文件一次性进内存。先跳过 skip 行，再逐块 yield 模板列表。"""
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
    parser.add_argument("--model-config", required=True, help="模型配置文件路径")
    parser.add_argument("--input", required=True, help="输入模板文件路径")
    parser.add_argument("--output", required=True, help="输出结果文件路径（JSON格式）")
    parser.add_argument("--model-replacement", default="", help="模型替换，如 'gpt-4o>gemini-pro-3-preview'")
    parser.add_argument("--prompt-root", default="", help="prompt文件根目录")
    parser.add_argument("--image-root", default="", help="图片资源根目录")
    parser.add_argument("--video-root", default="", help="视频资源根目录")
    parser.add_argument("--cache-dir", default="media_dir", help="缓存目录")
    parser.add_argument("--n-workers", default=4, type=int, help="多线程数量")
    parser.add_argument(
        "--multi-stream-mode",
        choices=["pixel", "time", "code", "code_adaptive", "cdpruner", "surge"],
        default="pixel",
        help=(
            "多流模式: "
            "pixel=默认非多流; "
            "time=双流按时间交错 A1 B1 A2 B2; "
            "code=每段选变化较大的一路输入，另一路用 Stream N: Unchanged 代替; "
            "code_adaptive=依据每段两路视频变化量自适应调节像素大小，对变化大的一路使用更高像素（最多 2.0 倍），"
            "变化小的一路使用更低像素（可接近 0），若某一路完全无变化则输出 Unchanged，另一条使用满像素; "
            "cdpruner=基于 CDPruner 思想的 token 选择，结合 time 多流，对 Qwen3-Omni-30B-A3B 进行视频 token 剪枝; "
            "surge=基于 SURGE 风格的时间惊讶度策略进行 video token 剪枝（结合 time 多流）"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑: 若 output 已存在且为 jsonl，则跳过前 N 条已存在结果，只跑剩余输入并追加写入",
    )
    parser.add_argument(
        "--memory-bank",
        action="store_true",
        help="启用多流 Memory Bank（Stream + Global）并将蒸馏记忆注入模型上下文",
    )
    parser.add_argument(
        "--memory-bank-model",
        default="",
        help="Memory Bank 使用的模型名（默认跟随当前 model 占位符模型）",
    )
    args = parser.parse_args()

    done_count = 0
    if getattr(args, "resume", False) and args.output.endswith(".jsonl"):
        out_path = Path(args.output)
        if out_path.exists() and out_path.stat().st_size > 0:
            with open(args.output, "r", encoding="utf-8") as f:
                done_count = sum(1 for line in f if line.strip())
            if done_count > 0:
                print(f"Resume: {done_count} lines already in {args.output}, skipping those inputs.")

    # jsonl 输入 + jsonl 输出：按块流式处理，避免大文件（如 2974 条 phostream）一次性进内存导致 OOM
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

    # 单文件或 .json 输出：沿用原逻辑（整表加载）
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
