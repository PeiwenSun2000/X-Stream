#!/usr/bin/env python3
"""测试 CLI 功能"""
import subprocess
import json
from pathlib import Path

demo_dir = Path(__file__).parent

# 创建完整测试模板（JSON 格式）
template_file = demo_dir / "template.json"
template = {
    "vars": {
        "instruction": "请用中文回答"
    },
    "rounds": [
        {
            "round_id": "cli_1",
            "messages": [
                {"role": "user", "content": "{{file:prompt.txt}}"},
                {"role": "assistant", "content": "{{model:gpt-4o,as=answer1}}"}
            ]
        },
        {
            "round_id": "cli_2",
            "messages": [
                {"role": "user", "content": "{{var:instruction}}这幅图片{{image:land.png}}描述了什么？"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=image_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "cli_3",
            "messages": [
                {"role": "user", "content": "{{image:land.mp4,time=1.0}}这帧画面如何？"},
                {"role": "assistant", "content": "{{model:doubao-seed-1-8-251228,as=frame_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "cli_4",
            "messages": [
                {"role": "user", "content": "简单介绍一下这个视频{{video:land.mp4,start=0,end=2,step=1,fps=1}}"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=video_summary,media_limit=1}}"},
                {"role": "user", "content": "{{var:instruction}}综合以上信息，给出最终结论。"},
                {"role": "assistant", "content": "{{model:gpt-4o,as=conclusion,return=1}}"}
            ]
        }
    ]
}
with open(template_file, "w", encoding="utf-8") as f:
    json.dump(template, f, indent=2, ensure_ascii=False)

# 运行 CLI（测试模型替换、路径替换等功能）
subprocess.run([
    "mllmflow",
    "--model-config", str(demo_dir / "models.json"),
    "--input", str(template_file),
    "--output", str(demo_dir / "cli_result.json"),
    "--model-replacement", "gpt-4o>gemini-3-pro-preview",
    "--prompt-root", str(demo_dir),
    "--image-root", str(demo_dir),
    "--video-root", str(demo_dir),
    "--cache-dir", str(demo_dir / "media_dir")
], check=True, cwd=demo_dir)

# 显示结果
print("=" * 60)
print("CLI 测试结果")
print("=" * 60)
with open(demo_dir / "cli_result.json", "r", encoding="utf-8") as f:
    result = json.load(f)
    print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("变量内容：")
for key, value in result.get("vars", {}).items():
    print(f"  {key}: {value}")
print("=" * 60)
