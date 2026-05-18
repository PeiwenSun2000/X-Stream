#!/usr/bin/env python3
"""MLLMFlow 完整功能演示"""
import json
from pathlib import Path
from mllmflow import MLLMFlow

demo_dir = Path(__file__).parent

# 创建测试文件
with open(demo_dir / "prompt.txt", "w", encoding="utf-8") as f:
    f.write("你是一个大模型，请用中文回答。101+24=？")

# 完整功能测试模板（JSON 格式）
template = {
    "vars": {
        "instruction": "请用中文回答"
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
                {"role": "user", "content": "{{var:instruction}}这幅图片{{image:land.png}}描述了什么？"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=image_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "3",
            "messages": [
                {"role": "user", "content": "{{image:land.mp4,time=1.0}}这帧画面如何？"},
                {"role": "assistant", "content": "{{model:doubao-seed-1-8-251228,as=frame_desc,media_limit=1}}"}
            ]
        },
        {
            "round_id": "4",
            "messages": [
                {"role": "user", "content": "简单介绍一下这个视频{{video:land.mp4,start=0,end=2,step=1,fps=1}}"},
                {"role": "assistant", "content": "{{model:gemini-3-pro-preview,as=video_summary,media_limit=1}}"}
            ]
        },
        {
            "round_id": "5",
            "messages": [
                {"role": "user", "content": "{{var:instruction}}综合以上信息，给出最终结论。"},
                {"role": "assistant", "content": "{{model:gpt-4o,as=conclusion,return=1}}"}
            ]
        }
    ]
}

if __name__ == "__main__":
    flow = MLLMFlow(str(demo_dir / "models.json"))

    print("=" * 60)
    print("MLLMFlow 功能演示")
    print("=" * 60)

    result = flow.run(template)

    print("\n执行结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("变量内容：")
    for key, value in result["vars"].items():
        print(f"  {key}: {value}")
    print("=" * 60)
