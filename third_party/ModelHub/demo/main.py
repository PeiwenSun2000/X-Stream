from model_hub import ModelHub

hub = ModelHub("models.json")

# Test model list
models = [
    # "gemini-3-pro-preview",
    # "qwen3-235b-a22b-instruct-2507",
    # "doubao-seed-1-8-251228",
    "gpt-4o"
    # "Qwen3-Omni-30B-A3B-Instruct",
    # "doubao-seedream-4-0-250828",
    # "echo"
]

# Test images and videos
image_path = "land.png"  # Replace with the actual image path
video_path = "land.mp4"  # Replace with the actual video path

# Test message
text = "Interstellar style, cinematic visuals, a massive black hole, a shattered vintage train bursting out from the black hole event horizon, extremely strong visual impact, post-apocalyptic wasteland style, motion blur, exaggerated wide-angle perspective, intense gravity, a sense of being swallowed. OC rendering, ray tracing, depth of field, surrealism, deep blue tones, rich color layering, realistic material texture, dark background lighting and shadows, lens flare, reflections, and ultimate light-and-shadow effects."
messages = [
    {
        "role": "user",
        "content": text
    }
]

# Test each model
for model_name in models:
    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")

    response = hub.call(
        model_name=model_name,
        messages=messages,
        request_params={"temperature": 0.7},
        request_id=f"{model_name}_debug"
    )
    print(response)

    # if "error" in response:
    #     print(f"Error: {response['error']}")
    # else:
    #     print(f"content: {response['content'][:200]}...")
    #     print(f"TokenUsage: {response['usage']}")
