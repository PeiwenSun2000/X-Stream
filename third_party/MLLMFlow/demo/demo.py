#!/usr/bin/env python3
"""Complete MLLMFlow feature demo"""
import json
from pathlib import Path
from mllmflow import MLLMFlow

demo_dir = Path(__file__).parent

# Create a test file
with open(demo_dir / "prompt.txt", "w", encoding="utf-8") as f:
    f.write("You are a large language model. Please answer in English. What is 101 + 24?")

# Full feature test template (JSON format)
template = {
    "vars": {
        "instruction": "Please answer in English"
    },
    "rounds": [
        {
            "round_id": "1",
            "messages": [
                {"role": "user", "content": "{{file:prompt.txt}}"},
                {"role": "assistant", "content": "{{model:gpt-4o,as=answer1}}"}
            ]
        },
        {
            "round_id": "2",
            "messages": [
                {"role": "user", "content": "{{var:instruction}}What does this image {{image:land.png}}depict?"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=image_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "3",
            "messages": [
                {"role": "user", "content": "{{image:land.mp4,time=1.0}}What is shown in this frame?"},
                {"role": "assistant", "content": "{{model:doubao-seed-1-8-251228,as=frame_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "4",
            "messages": [
                {"role": "user", "content": "Briefly introduce this video {{video:land.mp4,start=0,end=2,step=1,fps=1}}"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=video_summary,media_limit=1}}"}
            ]
        },
        {
            "round_id": "5",
            "messages": [
                {"role": "user", "content": "{{var:instruction}}Based on the information above, provide the final conclusion."},
                {"role": "assistant", "content": "{{model:gpt-4o,as=conclusion,return=1}}"}
            ]
        }
    ]
}

if __name__ == "__main__":
    flow = MLLMFlow(str(demo_dir / "models.json"))

    print("=" * 60)
    print("MLLMFlow Feature Demo")
    print("=" * 60)

    result = flow.run(template)

    print("\nExecution result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Variable contents:")
    for key, value in result["vars"].items():
        print(f"  {key}: {value}")
    print("=" * 60)
