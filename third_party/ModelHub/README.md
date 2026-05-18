# Model Hub

统一的模型中心，用于管理和调用多个 AI 模型 API。

## 安装

```bash
uv pip install -e .
```

## 使用

```python
from model_hub import ModelHub

# 创建实例
hub = ModelHub("models.json")

# 调用 API
response = hub.call(
    model_name="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    request_params={"temperature": 0.7},
    request_id="req-123"
)
print(response)
```

## 配置

`models.json` 使用扁平结构，示例：

```json
{
    "gpt-4o": [
        {
            "adapter": "openai",
            "model_name": "gpt-4o",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "api_key": "your-api-key",
            "max_retries": 3,
            "timeout": 600,
            "request_params": {
                "temperature": 0.7,
                "top_k": 50
            }
        }
    ],
    "gemini-3-pro-preview": [
        {
            "weight": 0.5,
            "adapter": "gemini",
            "model_name": "gemini-3-pro-preview",
            "endpoint": "https://api.example.com/v1beta/models/{model_name}:generateContent?key={api_key}",
            "api_key": "your-api-key",
            "max_video_size_mb": 20,
            "request_params": {}
        }
    ]
}
```

### 配置字段说明

**系统配置字段（顶层）：**
- `adapter`: 适配器类型（必需），支持：`openai`, `gemini`, `doubao`, `qwen`
- `model_name`: 模型名称（必需）
- `endpoint`: API 端点 URL（必需），支持模板变量 `{model_name}` 和 `{api_key}`
- `api_key`: API 密钥（可选）
- `max_retries`: 最大重试次数（默认：3）
- `timeout`: 超时时间，单位秒（默认：600）
- `max_video_size_mb`: 视频大小限制，单位 MB（默认：100）
- `weight`: 权重，用于多配置负载均衡（默认：1.0）

**请求参数字段：**
- `request_params`: 默认请求参数字典（可选），包含如 `temperature`、`top_k` 等参数，调用时可通过 `request_params` 参数覆盖

### 多配置与权重

可以为同一个模型配置多个端点，使用 `weight` 字段控制选择概率：

```json
{
    "model-name": [
        {"weight": 0.7, "adapter": "openai", ...},
        {"weight": 0.3, "adapter": "openai", ...}
    ]
}
```

## 消息格式

支持文本、图片和视频输入：

```python
# 纯文本
messages = [{"role": "user", "content": "Hello!"}]

# 多模态（文本 + 图片 + 视频）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片"},
            {"type": "image", "image": "path/to/image.png"},
            {"type": "video", "video": "path/to/video.mp4"}
        ]
    }
]
```

## 响应格式

```python
{
    "content": "模型返回的文本内容",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30
    },
    "raw_response": {...}  # 原始 API 响应
}
```

## 日志功能

当提供 `request_id` 时，会自动记录请求日志到 `logs/{request_id}.json`，包含：

- `payload`: 请求负载
- `response`: 原始响应
- `attempts_history`: 失败尝试历史
- `request_details`: 请求详情（模型名称、endpoint、超时等）

```python
hub.call(
    model_name="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    request_id="debug"  # 日志将保存到 logs/debug.json
)
```

## 支持的适配器

- **openai**: OpenAI 兼容 API（包括 Qwen 等）
- **gemini**: Google Gemini API
- **doubao**: 豆包 API
- **qwen**: 通义千问 API
