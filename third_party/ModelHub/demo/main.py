from model_hub import ModelHub

hub = ModelHub("models.json")

# 测试模型列表
models = [
    # "gemini-3-pro-preview",
    # "qwen3-235b-a22b-instruct-2507",
    # "doubao-seed-1-8-251228",
    "gpt-4o"
    # "Qwen3-Omni-30B-A3B-Instruct",
    # "doubao-seedream-4-0-250828",
    # "echo"
]

# 测试图片和视频
image_path = "land.png"  # 替换为实际图片路径
video_path = "land.mp4"  # 替换为实际视频路径

# 测试消息
text = "星際穿越風格，電影級畫面，巨大黑洞，一輛支離破碎的復古列車從黑洞視界線衝出，極強視覺衝擊力，末日廢土風，動態模糊，誇張的廣角透視，強引力，吞噬感。OC渲染，光線追踪，景深，超現實主義，深藍色調，豐富的色彩層次，真實材質質感，暗黑系背景光影，耀光，反射，极致光影效果。"
messages = [
    {
        "role": "user",
        "content": text
    }
]

# 测试每个模型
for model_name in models:
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"{'='*50}")

    response = hub.call(
        model_name=model_name,
        messages=messages,
        request_params={"temperature": 0.7},
        request_id=f"{model_name}_debug"
    )
    print(response)

    # if "error" in response:
    #     print(f"错误: {response['error']}")
    # else:
    #     print(f"内容: {response['content'][:200]}...")
    #     print(f"Token使用: {response['usage']}")
