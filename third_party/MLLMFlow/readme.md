# MLLMFlow

多模态智能对话流程构建工具，支持文本、图像、视频、本地文件以及大语言模型的灵活组合调用。

## 安装

```bash
pip install -e .
```

或者手动安装依赖：

```bash
pip install git+https://github.com/guanhuankang/ModelHub.git
pip install requests moviepy==2.2.1 json_repair
```

## 1. 模型配置

创建 `models.json`（遵循 ModelHub 配置格式）：

```json
{
    "gpt-4o": [{
        "adapter": "openai",
        "model_name": "gpt-4o",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_key": "your-api-key",
        "request_params": {
            "temperature": 0.7
        }
    }],
    "gemini-3-pro-preview": [{
        "weight": 0.5,
        "adapter": "gemini",
        "model_name": "gemini-3-pro-preview",
        "endpoint": "https://api.example.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        "api_key": "your-api-key",
        "max_video_size_mb": 20
    }]
}
```

配置说明：
- `adapter`: 适配器类型（openai, gemini, doubao, qwen 等）
- `model_name`: 模型名称
- `endpoint`: API 端点（支持 `{model_name}` 和 `{api_key}` 模板变量）
- `api_key`: API 密钥（可选）
- `weight`: 负载均衡权重（可选，默认 1.0）
- `request_params`: 模型请求参数（可选，如 temperature, top_k 等）
- `max_video_size_mb`: 最大视频大小限制（可选）

## 2. 模板示例

创建 `template.json`（JSON 格式）：

```json
{
  "vars": {
    "instruction": "请用中文回答"
  },
  "rounds": [
    {
      "round_id": "1",
      "messages": [
        {
          "role": "user",
          "content": "{{file:prompt.txt}}"
        },
        {
          "role": "assistant",
          "content": "{{model:gpt-4o,as=answer1}}"
        }
      ]
    },
    {
      "round_id": "2",
      "messages": [
        {
          "role": "user",
          "content": "{{var:instruction}}这幅图片{{image:land.png}}描述了什么？"
        },
        {
          "role": "assistant",
          "content": "{{model:gemini-3-pro-preview,as=image_desc,media_limit=1}}"
        }
      ]
    },
    {
      "round_id": "3",
      "messages": [
        {
          "role": "user",
          "content": "{{image:land.mp4,time=1.0}}这帧画面如何？"
        },
        {
          "role": "assistant",
          "content": "{{model:doubao-seed-1-8-251228,as=frame_desc,media_limit=1}}"
        }
      ]
    },
    {
      "round_id": "4",
      "messages": [
        {
          "role": "user",
          "content": "简单介绍一下这个视频{{video:land.mp4,start=0,end=2,step=1,fps=1}}"
        },
        {
          "role": "assistant",
          "content": "{{model:gemini-3-pro-preview,as=video_summary,media_limit=1}}"
        }
      ]
    },
    {
      "round_id": "5",
      "messages": [
        {
          "role": "user",
          "content": "{{var:instruction}}综合以上信息，给出最终结论。"
        },
        {
          "role": "assistant",
          "content": "{{model:gpt-4o,as=conclusion,return=1}}"
        },
        {
          "role": "user",
          "content": "{{var:instruction}}再见。"
        },
        {
          "role": "assistant",
          "content": "{{model:gpt-4o,as=byebye,return=1}}"
        }
      ]
    }
  ]
}
```

**注意**：
- 每一轮新对话开始，会清空上下文，但 `vars` 不会清空。可以通过 `vars` 跨轮次传递数据。
- 每一轮支持多组对话（多个 user/assistant 对）。
- `round_id` 可以是数字或字符串。
- `content` 字段是字符串，可以包含占位符（如 `{{file:...}}`、`{{model:...}}` 等）。

## 3. CLI 运行

### 基本用法

`usage: mllmflow [-h] --model-config MODEL_CONFIG --input INPUT [--output OUTPUT] [--model-replacement MODEL_REPLACEMENT] [--prompt-root PROMPT_ROOT] [--image-root IMAGE_ROOT] [--video-root VIDEO_ROOT] [--cache-dir CACHE_DIR]`

### 完整示例

```bash
mllmflow --model-config models.json --input template.json --output output.json

mllmflow \
  --model-config models.json \
  --input template.json \
  --output result.json \
  --model-replacement "gpt-4o>gemini-3-pro-preview" \
  --prompt-root ./prompts \
  --image-root ./images \
  --video-root ./videos \
  --cache-dir ./cache
```

### 参数说明

- `--model-config`: 模型配置文件路径（必需）
- `--input`: 输入模板文件路径（必需）
- `--output`: 输出结果文件路径（可选，不指定则输出到 stdout）
- `--model-replacement`: 模型替换，格式 `old>new`，多个用逗号分隔
  - 示例：`"gpt-4o>gemini-3-pro-preview,doubao-seed>qwen3-vl"`
- `--prompt-root`: prompt 文件根目录（可选）
- `--image-root`: 图片资源根目录（可选）
- `--video-root`: 视频资源根目录（可选）
- `--cache-dir`: 缓存目录（默认 `media_dir`）

### CLI 输出

CLI 会显示每个模型调用的进度和延迟：

```
[round-1_2][gpt-4o] Sending Request ...
[round-1_2][gpt-4o] latency: 2.35s
```

## 4. Python SDK 使用

```python
from mllmflow import MLLMFlow
import json

# 初始化
flow = MLLMFlow("models.json", cache_dir="media_dir")

# 定义模板（JSON 格式）
template = {
    "vars": {
        "instruction": "请用中文回答"
    },
    "rounds": [
        {
            "round_id": "1",
            "messages": [
                {
                    "role": "user",
                    "content": "{{file:prompt.txt}}"
                },
                {
                    "role": "assistant",
                    "content": "{{model:gpt-4o,as=answer}}"
                }
            ]
        },
        {
            "round_id": "2",
            "messages": [
                {
                    "role": "user",
                    "content": "{{image:photo.jpg}}描述这张图片"
                },
                {
                    "role": "assistant",
                    "content": "{{model:gpt-4o,as=desc,media_limit=1}}"
                }
            ]
        }
    ]
}

# 或者从文件读取
# with open("template.json", "r", encoding="utf-8") as f:
#     template = json.load(f)

# 运行流程
result = flow.run(template)

# 查看结果
print(result["vars"])  # 变量
print(result["rounds"])  # 对话轮次
```

### 模型替换

```python
flow = MLLMFlow(
    "models.json",
    model_replacement={"gpt-4o": "gemini-3-pro-preview"}
)
```

## 5. 模板语法

### JSON 格式

模板采用 JSON 格式，包含两个主要部分：

```json
{
  "vars": {
    "key": "value"
  },
  "rounds": [
    {
      "round_id": "1",
      "messages": [
        {
          "role": "user",
          "content": "..."
        },
        {
          "role": "assistant",
          "content": "..."
        }
      ]
    }
  ]
}
```

### 变量定义

在 `vars` 对象中定义变量：

```json
{
  "vars": {
    "instruction": "请用中文回答",
    "temperature": "0.7"
  }
}
```

变量可在模板中通过 `{{var:key}}` 引用。

### 对话轮次

在 `rounds` 数组中定义对话轮次：

```json
{
  "rounds": [
    {
      "round_id": "1",
      "messages": [
        {
          "role": "user",
          "content": "内容"
        },
        {
          "role": "assistant",
          "content": "内容"
        }
      ]
    }
  ]
}
```

- `round_id`: 轮次标识，可以是数字或字符串
- `messages`: 消息数组，每个消息包含 `role` 和 `content`
- `role`: 角色，通常为 `user`、`assistant` 或 `system`
- `content`: 内容字符串，可以包含占位符

### 占位符

#### `{{var:name}}` - 引用变量

```
user: {{var:instruction}}
```

#### `{{file:path}}` - 读取文件

```
user: {{file:prompt.txt}}
```

#### `{{image:path}}` - 插入图片

```
# 插入图片
user: {{image:photo.jpg}}

# 从视频截帧
user: {{image:video.mp4,time=1.5}}

# 指定缓存目录
user: {{image:photo.jpg,cache_dir=./cache}}
```

参数：
- `time=秒数`: 从视频指定时间截取一帧
- `cache_dir=路径`: 缓存目录

#### `{{video:path}}` - 插入视频

```
# 插入视频片段
user: {{video:demo.mp4,start=0,end=10}}

# 切分成多段
user: {{video:demo.mp4,start=0,end=10,step=2}}

# 指定帧率和缓存
user: {{video:demo.mp4,start=0,end=10,fps=1,cache_dir=./cache}}
```

参数：
- `start=秒数`: 起始时间（默认 0）
- `end=秒数`: 结束时间（默认视频全长）
- `step=秒数`: 切分步长，指定后会将视频切分成多段
- `fps=帧率`: 抽帧率（可选）
- `cache_dir=路径`: 缓存目录

#### `{{model:name}}` - 调用模型

```
assistant: {{model:gpt-4o,as=answer,media_limit=1,return=1}}
```

参数：
- `as=变量名`: 将模型输出保存到变量
- `return=0/1`: 是否将回复加入后续对话上下文（1=加入，默认；0=不加入）
- `media_limit=数量`: 本次调用允许的最大多媒体数量

## 完整示例

运行 `demo/demo.py` 查看完整功能演示。

运行 `demo/test_cli.py` 查看 CLI 测试示例。
