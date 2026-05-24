# MLLMFlow

Multimodal intelligent conversation workflow builder that supports flexible composition of text, images, videos, local files, and large language model calls.

## Installation

```bash
pip install -e .
```

Alternatively, install dependencies manually:

```bash
pip install git+https://github.com/guanhuankang/ModelHub.git
pip install requests moviepy==2.2.1 json_repair
```

## 1. Model Configuration

Create `models.json` (following the ModelHub configuration format):

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

Configuration Notes:
- `adapter`: Adapter type (openai, gemini, doubao, qwen, etc.)
- `model_name`: Model name
- `endpoint`: API endpoint (supports `{model_name}` and `{api_key}` template variables)
- `api_key`: API key (optional)
- `weight`: Load-balancing weight (optional, default 1.0)
- `request_params`: Model request parameters (optional, such as temperature, top_k, etc.)
- `max_video_size_mb`: Maximum video size limit (optional)

## 2. Template Example

Create `template.json` (JSON format):

```json
{
  "vars": {
    "instruction": "Please answer in English"
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
          "content": "{{var:instruction}}What does this image {{image:land.png}}depict?"
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
          "content": "{{image:land.mp4,time=1.0}}What is shown in this frame?"
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
          "content": "Briefly introduce this video {{video:land.mp4,start=0,end=2,step=1,fps=1}}"
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
          "content": "{{var:instruction}}Based on the information above, provide the final conclusion."
        },
        {
          "role": "assistant",
          "content": "{{model:gpt-4o,as=conclusion,return=1}}"
        },
        {
          "role": "user",
          "content": "{{var:instruction}}Goodbye."
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

**Note**：
- At the start of each new conversation round, context is cleared, but `vars` is not. Use `vars` to pass data across rounds.
- Each round supports multiple conversation pairs (multiple user/assistant pairs).
- `round_id` can be a number or string。
- The `content` field is a string and may contain placeholders such as `{{file:...}}` and `{{model:...}}`.

## 3. CLI Usage

### Basic Usage

`usage: mllmflow [-h] --model-config MODEL_CONFIG --input INPUT [--output OUTPUT] [--model-replacement MODEL_REPLACEMENT] [--prompt-root PROMPT_ROOT] [--image-root IMAGE_ROOT] [--video-root VIDEO_ROOT] [--cache-dir CACHE_DIR]`

### Complete Example

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

### Parameter Reference

- `--model-config`: Path to the model configuration file（required）
- `--input`: Path to the input template file（required）
- `--output`: Output result file path (optional; outputs to stdout if omitted)
- `--model-replacement`: Model replacement in `old>new` format; separate multiple entries with commas
  - Example:`"gpt-4o>gemini-3-pro-preview,doubao-seed>qwen3-vl"`
- `--prompt-root`: Root directory for prompt files (optional)
- `--image-root`: Root directory for image assets (optional)
- `--video-root`: Root directory for video assets (optional)
- `--cache-dir`: Cache directory (default: `media_dir`)

### CLI Output

The CLI displays progress and latency for each model call:

```
[round-1_2][gpt-4o] Sending Request ...
[round-1_2][gpt-4o] latency: 2.35s
```

## 4. Python SDK Usage

```python
from mllmflow import MLLMFlow
import json

# Initialize
flow = MLLMFlow("models.json", cache_dir="media_dir")

# Define a template (JSON format)
template = {
    "vars": {
        "instruction": "Please answer in English"
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
                    "content": "{{image:photo.jpg}}Describe this image"
                },
                {
                    "role": "assistant",
                    "content": "{{model:gpt-4o,as=desc,media_limit=1}}"
                }
            ]
        }
    ]
}

# Or read from a file
# with open("template.json", "r", encoding="utf-8") as f:
#     template = json.load(f)

# Run the workflow
result = flow.run(template)

# Inspect results
print(result["vars"])  # Variables
print(result["rounds"])  # Conversation Rounds
```

### Model Replacement

```python
flow = MLLMFlow(
    "models.json",
    model_replacement={"gpt-4o": "gemini-3-pro-preview"}
)
```

## 5. Template Syntax

### JSON Format 

Templates use JSON format and contain two main sections:

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

### Variable Definition

Define variables in the `vars` object:

```json
{
  "vars": {
    "instruction": "Please answer in English",
    "temperature": "0.7"
  }
}
```

Variables can be referenced in templates with `{{var:key}}`.

### Conversation Rounds

Define conversation rounds in the `rounds` array:

```json
{
  "rounds": [
    {
      "round_id": "1",
      "messages": [
        {
          "role": "user",
          "content": "content"
        },
        {
          "role": "assistant",
          "content": "content"
        }
      ]
    }
  ]
}
```

- `round_id`: Round identifier，can be a number or string
- `messages`: Message array; each message contains `role` and `content`
- `role`: Role, usually `user`, `assistant`, or `system`
- `content`: Content string that may contain placeholders

### Placeholders

#### `{{var:name}}` - Reference a Variable

```
user: {{var:instruction}}
```

#### `{{file:path}}` - Read a file

```
user: {{file:prompt.txt}}
```

#### `{{image:path}}` - Insert an image

```
# Insert an image
user: {{image:photo.jpg}}

# Extract a frame from a video
user: {{image:video.mp4,time=1.5}}

# Specify the cache directory
user: {{image:photo.jpg,cache_dir=./cache}}
```

Parameters:
- `time=seconds`: Extract one frame from the specified time in a video
- `cache_dir=path`: Cache directory

#### `{{video:path}}` - Insert a video

```
# Insert a video clip
user: {{video:demo.mp4,start=0,end=10}}

# Split into multiple segments
user: {{video:demo.mp4,start=0,end=10,step=2}}

# Specify frame rate and cache
user: {{video:demo.mp4,start=0,end=10,fps=1,cache_dir=./cache}}
```

Parameters:
- `start=seconds`: Start time (default 0)
- `end=seconds`: End time (default: full video duration)
- `step=seconds`: Step size for splitting; when specified, the video is split into multiple segments
- `fps=frame_rate`: Frame sampling rate (optional)
- `cache_dir=path`: Cache directory

#### `{{model:name}}` - Call a model

```
assistant: {{model:gpt-4o,as=answer,media_limit=1,return=1}}
```

Parameters:
- `as=variable_name`: Save model output to a variable
- `return=0/1`: Whether to add the response to subsequent conversation context (1=add, default; 0=do not add)
- `media_limit=count`: Maximum number of multimedia items allowed for this call

## Complete Example

Run `demo/demo.py` to view the complete feature demo.

Run `demo/test_cli.py` to view the CLI test example.
