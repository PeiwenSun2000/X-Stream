# Model Hub

A unified model hub for managing and calling multiple AI model APIs.

## Installation

```bash
uv pip install -e .
```

## Usage

```python
from model_hub import ModelHub

# Create an instance
hub = ModelHub("models.json")

# Call the API
response = hub.call(
    model_name="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    request_params={"temperature": 0.7},
    request_id="req-123"
)
print(response)
```

## Configuration

`models.json` uses a flat structure, for example:

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

### Configuration Fields

**System configuration fields (top level):**
- `adapter`: Adapter type (required), supports: `openai`, `gemini`, `doubao`, `qwen`
- `model_name`: Model name (required)
- `endpoint`: API endpoint URL (required), supports template variables `{model_name}` and `{api_key}`
- `api_key`: API key (optional)
- `max_retries`: Maximum retry count (default: 3)
- `timeout`: Timeout in seconds (default: 600)
- `max_video_size_mb`: Video size limit in MB (default: 100)
- `weight`: Weight for load balancing across multiple configurations (default: 1.0)

**Request parameter fields:**
- `request_params`: Default request parameter dictionary (optional), containing parameters such as `temperature` and `top_k`; can be overridden by the `request_params` argument when calling

### Multiple Configurations And Weights

You can configure multiple endpoints for the same model and use the `weight` field to control selection probability:

```json
{
    "model-name": [
        {"weight": 0.7, "adapter": "openai", ...},
        {"weight": 0.3, "adapter": "openai", ...}
    ]
}
```

## Message Format

Supports text, image, and video inputs:

```python
# Plain text
messages = [{"role": "user", "content": "Hello!"}]

# Multimodal (text + image + video)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe this image"},
            {"type": "image", "image": "path/to/image.png"},
            {"type": "video", "video": "path/to/video.mp4"}
        ]
    }
]
```

## Response Format

```python
{
    "content": "Text content returned by the model",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30
    },
    "raw_response": {...}  # Raw API response
}
```

## Logging

When `request_id` is provided, request logs are automatically written to `logs/{request_id}.json`, including:

- `payload`: Request payload
- `response`: Raw response
- `attempts_history`: Failed attempt history
- `request_details`: Request details (model name, endpoint, timeout, etc.)

```python
hub.call(
    model_name="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    request_id="debug"  # The log will be saved to logs/debug.json
)
```

## Supported Adapters

- **openai**: OpenAI compatible API (including Qwen, etc.)
- **gemini**: Google Gemini API
- **doubao**: Doubao API
- **qwen**: Qwen API
