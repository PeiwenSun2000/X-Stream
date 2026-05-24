#!/usr/bin/env python3
"""Test CLI functionality"""
import subprocess
import json
from pathlib import Path

demo_dir = Path(__file__).parent

# Create a complete test template (JSON format)
template_file = demo_dir / "template.json"
template = {
    "vars": {
        "instruction": "Please answer in English"
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
                {"role": "user", "content": "{{var:instruction}}What does this image {{image:land.png}}depict?"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=image_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "cli_3",
            "messages": [
                {"role": "user", "content": "{{image:land.mp4,time=1.0}}What is shown in this frame?"},
                {"role": "assistant", "content": "{{model:doubao-seed-1-8-251228,as=frame_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "cli_4",
            "messages": [
                {"role": "user", "content": "Briefly introduce this video {{video:land.mp4,start=0,end=2,step=1,fps=1}}"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=video_summary,media_limit=1}}"},
                {"role": "user", "content": "{{var:instruction}}Based on the information above, provide the final conclusion."},
                {"role": "assistant", "content": "{{model:gpt-4o,as=conclusion,return=1}}"}
            ]
        }
    ]
}
with open(template_file, "w", encoding="utf-8") as f:
    json.dump(template, f, indent=2, ensure_ascii=False)

# Run the CLI (test model replacement, path replacement, and related features)
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

# Display results
print("=" * 60)
print("CLI Test Result")
print("=" * 60)
with open(demo_dir / "cli_result.json", "r", encoding="utf-8") as f:
    result = json.load(f)
    print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("Variable contents:")
for key, value in result.get("vars", {}).items():
    print(f"  {key}: {value}")
print("=" * 60)
