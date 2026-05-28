# X-Stream Inference

Official inference and evaluation code for **X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding**. This package runs online multi-stream video QA with local vLLM checkpoints or hosted API models.

## Introduction

X-Stream is a multi-stream streaming understanding benchmark for evaluating how multimodal large language models handle concurrent video streams. It contains 4,220 curated QA pairs across 932 videos and covers 11 subtasks in multi-window, multi-view, and multi-device scenarios. The paper frames current MLLMs as naive multiplexers and studies spatial, temporal, and semantic ways to combine multiple streams into one model-consumable token sequence.

The `inference/` package keeps the runtime simple: most users only need `run.sh`.

Supported multi-stream modes:

| Mode                    | Multiplexing term                                | Meaning                                                                                                                                                                                             | Input file                                  |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `pixel`                 | Spatial Division Multiplexing                    | Uses the pre-merged video input and sends each tiled visual stream as one spatial canvas without multi-stream segment expansion.                                                                    | `eval_relative_merged_phostream_type.jsonl` |
| `time`                  | Time Division Multiplexing                       | Splits step-based video placeholders into segments and interleaves them as `Stream 1: A1`, `Stream 2: B1`, `Stream 1: A2`, `Stream 2: B2`, and so on.                                               | `eval_relative_multi_phostream_type.jsonl`  |
| `code`, `code_adaptive` | Extra Exploration                                | `code` keeps the stream segment with the larger video-change score and marks the others as unchanged, while `code_adaptive` scales each changed stream's FPS between 0x and 2x based on that score. | `eval_relative_multi_phostream_type.jsonl`  |
| `cdpruner`              | Semantic Division Multiplexing (Dropping frames) | Reuses time-style interleaving, then applies client-side media selection with CDPruner-style instruction relevance and diversity before the model call.                                             | `eval_relative_multi_phostream_type.jsonl`  |
| `surge`                 | Extra Exploration                                | Reuses time-style interleaving, then applies client-side SURGE-style temporal surprise selection before the model call.                                                                             | `eval_relative_multi_phostream_type.jsonl`  |
| `cdpruner_token`        | Semantic Division Multiplexing                   | Reuses time-style interleaving and forwards pruning metadata to the local vLLM worker, where the X-Stream pruner performs patch-level CDPruner token selection inside video frames.                 | `eval_relative_multi_phostream_type.jsonl`  |
| `surge_token`           | Extra Exploration                                | Reuses time-style interleaving and forwards pruning metadata to the local vLLM worker, where the X-Stream pruner performs patch-level SURGE token selection inside video frames.                    | `eval_relative_multi_phostream_type.jsonl`  |

`cdpruner_token` and `surge_token` are only available with local vLLM. Hosted API models cannot run patch-level token pruning because the pruning hook must be installed inside the vLLM worker.

## Environment Setup

### 1. Common Base Environment

Use this base setup before running inference.

Requirements:

- Linux.
- Python `>=3.12,<3.13`.
- `uv >= 0.4`.
- `ffmpeg` and `ffprobe` on `PATH` for video probing and segment-cache generation.
- NVIDIA GPU and CUDA-compatible drivers for local vLLM runs. API-only runs and cache prewarming can run without GPUs.

Install the project environment:

```bash
git clone https://github.com/PeiwenSun2000/X-Stream.git
cd X-Stream/inference
uv sync --extra local
```

Use `uv run` for commands:

```bash
uv run bash run.sh --help
```

Or activate the environment manually:

```bash
source .venv/bin/activate
bash run.sh --help
```

Create a local model configuration:

```bash
cp configs/models.example.json configs/models.json
```

### 2. Download Data

Download the X-Stream dataset from [Hugging Face](https://huggingface.co/datasets/spw2000/X-stream). The dataset is distributed as JSONL manifests plus compressed video archives. In this repository, the examples place the downloaded dataset root directly at `data/`, so commands launched from `inference/` can refer to it as `../data`.

```bash
cd X-Stream
pip install -U huggingface_hub
huggingface-cli download spw2000/X-stream \
  --repo-type dataset \
  --local-dir data
```

If you also download video archives, install `zstd` and extract the archives from that dataset root:

```bash
sudo apt-get update
sudo apt-get install -y zstd
python data/scripts/extract_archives.py --dataset-root data
```

To extract only the evaluation split or only the lightweight 2 fps model-input videos:

```bash
python data/scripts/extract_archives.py --dataset-root data --splits eval
python data/scripts/extract_archives.py --dataset-root data --kinds reencoded
```

### 3. Data Files

After download and extraction, the expected dataset root looks like this:

```text
data/
|-- eval_relative.json
|-- train_relative.json
|-- eval_relative_merged_phostream_type.jsonl
|-- eval_relative_multi_phostream_type.jsonl
|-- archives/
|   |-- SHA256SUMS
|   |-- archives.json
|   |-- eval/
|   `-- train/
|-- scripts/
|   `-- extract_archives.py
`-- data/
    |-- eval/
    |   |-- merged/
    |   |-- reencoded/
    |   `-- original/
    `-- train/
        |-- merged/
        |-- reencoded/
        `-- original/
```

The official dataset manifests are JSON Lines files with a `.json` extension:

```text
../data/eval_relative.json
../data/train_relative.json
```

Each record stores video paths relative to the dataset root. The main fields are:

- `merged_video_path`: a merged multi-stream video under `data/eval/merged/` or `data/train/merged/`.
- `encoded_video_path`: synchronized per-stream videos under `data/eval/reencoded/` or `data/train/reencoded/`.
- `original_video_path`: higher-fps source videos under `data/eval/original/` or `data/train/original/`.
- `verified_responses`: verified QA annotations.

The inference runner uses the MLLMFlow-ready evaluation JSONL files:

```text
../data/eval_relative_merged_phostream_type.jsonl
../data/eval_relative_multi_phostream_type.jsonl
```

Use `eval_relative_merged_phostream_type.jsonl` with `--multi-stream pixel`. Use `eval_relative_multi_phostream_type.jsonl` with `time`, `code`, `code_adaptive`, `cdpruner`, `surge`, `cdpruner_token`, and `surge_token`.

Use `eval_relative.json` for dataset inspection, upload checks, and release validation. `run.sh` expects the MLLMFlow-ready JSONL files above for inference.

### 4. Local Runtime Variables

For local vLLM checkpoint runs:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLOW_CONFIG=configs/models.json
export STREAM_EVAL_JUDGER=qwen3-235b-a22b-instruct-2507
```

If `STREAM_EVAL_JUDGER=qwen3-235b-a22b-instruct-2507`, provide a Qwen-compatible judge endpoint and key:

```bash
export QWEN_ENDPOINT=https://<your-qwen-compatible-endpoint>/v1/chat/completions
export QWEN_API_KEY=<your-qwen-api-key>
```

For hosted API models, export only the provider credentials used by the selected model:

```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export OPENAI_API_KEY=<your-openai-api-key>
export GEMINI_API_KEY=<your-gemini-api-key>
```

For a quick smoke test or inference-only run, add `--no-stream-eval` to avoid judge credentials.

### 5. Model Checkpoints

Local vLLM runs need a downloaded checkpoint. The examples below use `Qwen3-Omni-30B-A3B-Instruct`, whose logical model name must match the key in `configs/models.json`.

Download the checkpoint with the Hugging Face CLI:

```bash
cd X-Stream/inference
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir checkpoints/Qwen3-Omni-30B-A3B-Instruct
```

Then use the checkpoint root as `--vllm-model-path`:

```bash
--vllm-model-path ./checkpoints
```

The expected structure is:

```text
inference/
`-- checkpoints/
    `-- Qwen3-Omni-30B-A3B-Instruct/
        |-- config.json
        |-- tokenizer_config.json
        |-- generation_config.json
        |-- model-00001-of-*.safetensors
        `-- ...
```

`pipeline.sh` also supports pointing `--vllm-model-path` directly at one checkpoint directory if that directory contains `config.json`:

```bash
--vllm-model-path ./checkpoints/Qwen3-Omni-30B-A3B-Instruct
```

For other local models, keep the same rule: the directory name under `checkpoints/` should match the logical model key passed through `--model`, or `--vllm-model-path` should point directly to a checkpoint directory with `config.json`.

### 6. CLIP Weights For Pruning Modes

Some pruning modes use CLIP features in addition to the main MLLM checkpoint:

| Mode | CLIP usage |
| --- | --- |
| `cdpruner` | Uses `openai/clip-vit-large-patch14-336` for instruction relevance and visual diversity before the model call. |
| `surge` | Uses `openai/clip-vit-large-patch14-336` to embed representative video frames before the model call; this is a client-side segment-level SURGE approximation, not vLLM-internal token pruning. |
| `cdpruner_token` | Uses the CLIP text tower lazily when instruction text is available; if CLIP is unavailable, it falls back to visual-only diversity inside the local vLLM worker. |
| `surge_token` | Does not require CLIP; it runs inside the local vLLM worker and computes surprise directly from post-vision-encoder video token embeddings. |

If the machine has internet access, the CLIP weights are downloaded lazily through `transformers`. For offline or firewalled machines, pre-populate the Hugging Face cache before running pruning modes:

```bash
pip install -U huggingface_hub
huggingface-cli download openai/clip-vit-large-patch14-336
```

If you use a custom Hugging Face cache location, set it before downloading and before running inference:

```bash
export HF_HOME=/path/to/hf-cache
huggingface-cli download openai/clip-vit-large-patch14-336
```

Segment-level `cdpruner` and `surge` fall back to the default media-limit behavior if CLIP cannot be loaded, so the run may continue but it will not use the intended pruning strategy.

### 7. Runtime Option Checklist

For full local vLLM runs, keep these options together unless you intentionally change the experiment:

| Option                                                                          | Why it matters                                                                                                         |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `--model Qwen3-Omni-30B-A3B-Instruct`                                           | Selects the logical model key from `configs/models.json`.                                                              |
| `--vllm-model-path /path/to/checkpoints`                                        | Points vLLM to the local checkpoint root.                                                                              |
| `--input ../data/eval_relative_*.jsonl`                                         | Selects the MLLMFlow-ready task file.                                                                                  |
| `--multi-stream MODE`                                                           | Selects `pixel`, `time`, `code`, `code_adaptive`, `cdpruner`, `surge`, `cdpruner_token`, or `surge_token`.             |
| `--video-root ../data`                                                          | Resolves `{{video:...}}` placeholders in the JSONL inputs.                                                             |
| `--prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt` | Resolves `{{file:system_prompt.txt}}` in the JSONL inputs.                                                             |
| `--tp 2`                                                                        | Sets vLLM tensor parallel size per service. Match this to your GPU count and memory.                                   |
| `--workers 4`                                                                   | Sets MLLMFlow request concurrency. Use `1` for token-level pruning modes unless you have validated higher concurrency. |
| `--max-model-len 200000`                                                        | Allows long multi-stream contexts; keep `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` when using this value.                       |
| `--run-id NAME`                                                                 | Gives the output directory a reproducible, readable name.                                                              |

For hosted API or smoke-test runs, add `--no-vllm`. For inference-only runs, add `--no-stream-eval`. Do not use `cdpruner_token` or `surge_token` with hosted API models; use `pixel`, `time`, `code`, `code_adaptive`, `cdpruner`, or `surge` instead.

## Usage

Start with the API-free smoke test, then add vLLM, hosted API models, or evaluation as needed.

### 1. API-Free Smoke Test

This command verifies the Python environment, CLI path, input parsing, output writing, and video-root resolution. It does not start vLLM and does not call any hosted API.

```bash
cd X-Stream/inference
uv run bash run.sh \
  --model echo \
  --no-vllm \
  --no-stream-eval \
  --input ../data/eval_relative_merged_phostream_type.jsonl \
  --multi-stream pixel \
  --workers 2 \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data \
  --run-id smoke_echo_pixel
```

### 2. Local vLLM With Merged Pixel Input

Use this when your checkpoint is available on the same machine. Keep `/path/to/checkpoints` as a placeholder for the local checkpoint root.

```bash
cd X-Stream/inference
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLOW_CONFIG=configs/models.json
export STREAM_EVAL_JUDGER=qwen3-235b-a22b-instruct-2507

uv run bash run.sh \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --vllm-model-path /path/to/checkpoints \
  --input ../data/eval_relative_merged_phostream_type.jsonl \
  --multi-stream pixel \
  --tp 2 \
  --workers 4 \
  --max-model-len 200000 \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data \
  --run-id qwen3omni_pixel
```

### 3. Local vLLM With Multi-Stream Input

Switch to `eval_relative_multi_phostream_type.jsonl` for temporal, semantic, and token-reduction modes.

```bash
cd X-Stream/inference
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLOW_CONFIG=configs/models.json
export STREAM_EVAL_JUDGER=qwen3-235b-a22b-instruct-2507

uv run bash run.sh \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --vllm-model-path /path/to/checkpoints \
  --input ../data/eval_relative_multi_phostream_type.jsonl \
  --multi-stream time \
  --tp 2 \
  --workers 4 \
  --max-model-len 200000 \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data \
  --run-id qwen3omni_time
```

To run another non-token mode, change only `--multi-stream` and `--run-id`, for example `code`, `code_adaptive`, `cdpruner`, or `surge`.

### 4. Local vLLM With Token-Level Pruning

`surge_token` and `cdpruner_token` require a local vLLM backend. Do not pass `--no-vllm`, and do not use these modes with hosted API models.

```bash
cd X-Stream/inference
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLOW_CONFIG=configs/models.json

uv run bash run.sh \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --vllm-model-path /path/to/checkpoints \
  --input ../data/eval_relative_multi_phostream_type.jsonl \
  --multi-stream cdpruner_token \
  --xstream-rho 0.25 \ # Choose the ratio that fits your data
  --tp 2 \
  --workers 1 \
  --max-model-len 200000 \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data \
  --run-id qwen3omni_cdpruner_token
```

### 5. Hosted API Model

Hosted API models do not start vLLM. Pass `--no-vllm` and make sure the relevant provider key exists in the environment. Patch-level token pruning modes (`cdpruner_token` and `surge_token`) are not supported for API models.

```bash
cd X-Stream/inference
export FLOW_CONFIG=configs/models.json
export OPENROUTER_API_KEY=<your-openrouter-api-key>

uv run bash run.sh \
  --model qwen3-vl-30b-a3b-instruct \
  --no-vllm \
  --input ../data/eval_relative_merged_phostream_type.jsonl \
  --multi-stream pixel \
  --workers 8 \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data \
  --run-id api_qwen3vl_pixel
```

### 6. CPU Cache Prewarming

Long multi-stream videos are split into cached MP4 segments before model inference. This MoviePy and ffmpeg stage is CPU-bound. Prewarm the cache on a CPU machine before a GPU run.

```bash
cd X-Stream/inference
uv run bash run.sh \
  --input ../data/eval_relative_multi_phostream_type.jsonl \
  --warm-cache-only \
  --workers 64 \
  --cache-warm-workers 64 \
  --cache-dir ./cache \
  --run-id prewarm_multi \
  --prompt-root third_party/MLLMFlow/annotations/system_prompt/streaming_prompt \
  --video-root ../data
```

`--warm-cache-only` does not start vLLM and does not call any model. It only resolves `{{video:...}}` placeholders and writes the segment cache. Keep `--cache-dir`, `--input`, `--video-root`, and video placeholder parameters identical between prewarming and the later GPU run.

For `time`, `cdpruner`, `surge`, `cdpruner_token`, and `surge_token`, `--multi-stream` can be omitted during prewarming because the same base video segments are generated. For `code_adaptive`, pass the same `--multi-stream code_adaptive` as the later run because it may create additional fps-scaled segments.

## Outputs And Evaluation

Each run writes to:

```text
outputs/<RUN_ID>_<YYYYMMDD-HHMMSS>/
```

Typical contents:

```text
run_env.json
models.json
output_<input>.jsonl
eval.sh
eval.json
vllm_pids.txt
vllmlogs/
```

Useful flags:

- `--resume`: continue a compatible incomplete run.
- `--no-stream-eval`: skip `stream-eval` and write only raw model outputs.
- `--stream-eval-judger MODEL`: choose the judge model.
- `--output-dir DIR`: change the output root.
- `--warm-cache-only`: pre-generate video segment cache on CPU and exit.
- `--cache-warm-workers N`: set CPU prewarming concurrency.

`run_env.json` records resolved runtime paths and options. Use it to reproduce a run or inspect which config, input, cache directory, and multi-stream mode were used.

## Directory Structure

```text
inference/
|-- README.md
|-- run.sh                         # Main entrypoint for inference runs
|-- pipeline.sh                    # vLLM startup, resume, evaluation, cleanup
|-- pyproject.toml                 # uv environment and dependency pins
|-- configs/
|   |-- models.example.json        # Public model-config template
|   `-- models.json                # Local model config
|-- tools/
|-- third_party/
|   |-- MLLMFlow
|   |-- ModelHub
|   |-- stream-eval
|   `-- xstream_vllm_pruner
|-- outputs/                       # Generated runs
`-- cache/                         # Generated video segment cache
```

## Token Rate Guidelines

Different model providers account for video tokens differently. If a run exceeds the model's video-token or token-per-second budget, reduce the input load by lowering the resolution, lowering the FPS, shortening clips, or changing playback speed according to the model family:

1. Gemini: Fixed 258 tokens/sec (independent of resolution/FPS).
2. GPT: 85 tokens/frame + 170 tokens per 512$\times$512 tile.
3. Qwen3+: 28$\times$28 pixel patches per token with token merging.

Use these rules to estimate the effective token rate for your target model, then choose the resolution, FPS, clip length, or playback-speed adjustment that keeps the input within that model's limit.

## FAQ

### Which dataset file should I use?

Use `../data/eval_relative_merged_phostream_type.jsonl` for `pixel`. Use `../data/eval_relative_multi_phostream_type.jsonl` for all multi-stream modes. Use `../data/eval_relative.json` when you need to inspect or validate the dataset manifest itself.

### Why does `eval_relative.json` not appear in `run.sh` examples?

`eval_relative.json` is the release manifest. The inference runner consumes the MLLMFlow-ready JSONL task files, so the executable examples use `eval_relative_merged_phostream_type.jsonl` or `eval_relative_multi_phostream_type.jsonl`.

### `ModuleNotFoundError: No module named 'vllm'`

Run through `uv` or activate this project environment:

```bash
cd X-Stream/inference
uv sync --extra local
uv run bash run.sh --help
```

### vLLM never becomes healthy

Check `outputs/<run>/vllmlogs/<port>.log`. Common fixes are reducing `--gpu-mem-util`, reducing `--max-model-len`, increasing `--tp`, or setting:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

This repo commonly uses `--max-model-len 200000` for long multi-stream runs.

### `KeyError: 'QWEN_ENDPOINT'` or `Invalid URL '${QWEN_ENDPOINT}'`

The default judge model uses `${QWEN_ENDPOINT}` and `${QWEN_API_KEY}` placeholders. Export both variables, choose a different `--stream-eval-judger`, or pass `--no-stream-eval` for inference-only runs.

### API model returns 401 or 429

401 usually means the API key or endpoint is invalid. 429 usually means rate limit or quota exhaustion. Check the relevant environment variable and provider quota for the model configured in `configs/models.json`.

### `invalid choice: 'surge_token'` or `invalid choice: 'cdpruner_token'`

The environment is loading an older MLLMFlow. Run from this `inference/` directory and reinstall local editable packages:

```bash
uv sync --extra local
```

### Token-level pruning seems inactive

Make sure you are using local vLLM and did not pass `--no-vllm`. Then enable debug logs:

```bash
XSTREAM_VLLM_PRUNER_DEBUG=1 uv run bash run.sh ...
```

Look for `xstream_vllm_pruner: enabled` and `compute_retention_mask patch installed` in `outputs/<run>/vllmlogs/*.log`.

### Can I use token-level pruning with API models?

No. `cdpruner_token` and `surge_token` require the X-Stream pruning hook to run inside a local vLLM worker. Hosted API providers do not expose that internal worker path, so API models can only use non-token pruning modes such as `pixel`, `time`, `code`, `code_adaptive`, `cdpruner`, or `surge`.

### Input path fails

Verify both `--input` and `--video-root`. `run.sh` converts paths to absolute paths before launching workers, but it cannot fix a missing file or a video root that does not match the placeholders inside the JSONL.

### How do I resume a failed or interrupted run?

Rerun the same command with:

```bash
--resume
```

Finished JSONL rows are skipped when the existing output is compatible with the current command.

## Discussion

1. Drawback of semantic multiplexing.

```text
In a typical streaming setting, the question is provided only after the frames have already appeared. This means that, when a frame is first observed, the question cannot be used as a query to determine which salient tokens should be retained.

However, most existing methods for identifying salient tokens rely on question-based importance ranking and keep only the tokens deemed important. As a result, they cannot fundamentally address this limitation. We leave this issue for the community to further explore.
```

## Acknowledgements

This inference package builds on ideas and components from the following open-source projects:

- [PhoStream](https://github.com/Lucky-Lance/PhoStream) and [AURA](https://github.com/aurateam2026/AURA) for streaming video understanding infrastructure and evaluation design.
- [CDPruner](https://github.com/Theia-4869/CDPruner) and [SURGE](https://github.com/BarryTang22/SURGE) for visual token pruning.

## Citation

```bibtex
@inproceedings{sun2026xstream,
  title     = {X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding},
  author    = {Sun, Peiwen and Lu, Xudong and Liu, Huadai and Bo, Yang and Wu, Dongming and Guan, Huankang and Cai, Minghong and Chen, Jinpeng and Guo, Xintong and Li, Shuhan and Liu, Rui and Yue, Xiangyu},
  booktitle = {arXiv},
  year      = {2026}
}
```

## License

This inference package is released under the [MIT License](LICENSE). Third-party packages under `third_party/` keep their original licenses and notices.
