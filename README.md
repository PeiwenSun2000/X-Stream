<!-- markdownlint-disable MD033 MD041 -->

# <img src="assets/logo.png" alt="X-Stream logo" width="10%"> X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding

**Authors:** Peiwen Sun<sup>*1</sup>, Xudong Lu<sup>*1</sup>, Huadai Liu<sup>*3</sup>, Yang Bo<sup>2</sup>, Dongming Wu<sup>1</sup>, Huankang Guan<sup>2</sup>, Minghong Cai<sup>1</sup>, Jinpeng Chen<sup>2</sup>, Xintong Guo<sup>2</sup>, Shuhan Li<sup>2</sup>, Rui Liu<sup>2</sup>, and Xiangyu Yue<sup>&dagger;1</sup>.

**Affiliations:** <sup>1</sup>MMLab, The Chinese University of Hong Kong; <sup>2</sup>Huawei Inc.; <sup>3</sup>Independent.

Official inference and evaluation code for **X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding**. This package runs online multi-stream video QA with local vLLM checkpoints or hosted API models.


<p align="center">
  <a href="https://peiwensun2000.github.io/xstream/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Website"></a>
  <a href="https://huggingface.co/datasets/spw2000/X-stream"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset HuggingFace"></a>
  <a href="https://peiwensun2000.github.io/xstream/"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="X-Stream multi-stream scenarios">
</p>

## Introduction

X-Stream is a multi-stream streaming understanding benchmark for evaluating how multimodal large language models handle concurrent video streams. It contains 4,220 curated QA pairs across 932 videos and covers 11 subtasks in multi-window, multi-view, and multi-device scenarios. The paper frames current MLLMs as naive multiplexers and studies spatial, temporal, and semantic ways to combine multiple streams into one model-consumable token sequence.

The `inference/` package keeps the runtime simple: most users only need `run.sh`.

## Pipeline

<p align="center">
  <img src="assets/multiplexing_pipeline.png" alt="X-Stream multiplexing pipeline" width="90%">
</p>

X-Stream evaluates online inference where synchronized streams are multiplexed into a single model-consumable sequence under a fixed average video-token rate. The runner supports spatial division for tiled video inputs, time division for stream-wise interleaving, and semantic division or token-level pruning for reducing redundant visual content before or inside the model call.

Supported multi-stream modes:

<!-- markdownlint-disable MD033 -->
<table>
  <thead>
    <tr>
      <th width="10%">Mode</th>
      <th width="22%">Multiplexing term</th>
      <th width="50%">Meaning</th>
      <th width="18%">Input file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>pixel</code></td>
      <td>Spatial Division Multiplexing</td>
      <td>Uses the pre-merged video input and sends each tiled visual stream as one spatial canvas without multi-stream segment expansion.</td>
      <td><code>eval_relative_merged_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>time</code></td>
      <td>Time Division Multiplexing</td>
      <td>Splits step-based video placeholders into segments and interleaves them as <code>Stream 1: A1</code>, <code>Stream 2: B1</code>, <code>Stream 1: A2</code>, <code>Stream 2: B2</code>, and so on.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>code</code><br><code>code_adaptive</code></td>
      <td>Extra Exploration</td>
      <td><code>code</code> keeps the stream segment with the larger video-change score and marks the others as unchanged, while <code>code_adaptive</code> scales each changed stream's FPS between 0x and 2x based on that score.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>cdpruner</code></td>
      <td>Semantic Division Multiplexing (Dropping frames)</td>
      <td>Reuses time-style interleaving, then applies client-side media selection with CDPruner-style instruction relevance and diversity before the model call.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>surge</code></td>
      <td>Extra Exploration</td>
      <td>Reuses time-style interleaving, then applies client-side SURGE-style temporal surprise selection before the model call.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>cdpruner_token</code></td>
      <td>Semantic Division Multiplexing</td>
      <td>Reuses time-style interleaving and forwards pruning metadata to the local vLLM worker, where the X-Stream pruner performs patch-level CDPruner token selection inside video frames.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
    <tr>
      <td><code>surge_token</code></td>
      <td>Extra Exploration</td>
      <td>Reuses time-style interleaving and forwards pruning metadata to the local vLLM worker, where the X-Stream pruner performs patch-level SURGE token selection inside video frames.</td>
      <td><code>eval_relative_multi_<br>phostream_type.jsonl</code></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-enable MD033 -->

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
cd X-Stream
uv sync --extra local
```

If you are working from a monorepo checkout where this package lives under an `inference/` subdirectory, run the same `uv sync --extra local` command from that `inference/` directory instead. If your default `python3` is not Python 3.12, point uv at a 3.12 interpreter explicitly:

```bash
UV_PYTHON=/path/to/python3.12 uv sync --extra local
uv run python --version
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

For reproducible environment comparisons, prefer uv over invoking `pip` directly because uv-created virtual environments may not include the `pip` Python module:

```bash
uv pip freeze --python .venv/bin/python > env.freeze.txt
```

### 2. Download Data

Download the X-Stream dataset from [Hugging Face](https://huggingface.co/datasets/spw2000/X-stream). The dataset is distributed as JSONL manifests plus compressed video archives. In a monorepo checkout, the examples place the downloaded dataset root directly at the repository-level `data/`, so commands launched from `inference/` can refer to it as `../data`. In a standalone public `X-Stream` checkout, either place the dataset as a sibling directory and keep using `../data`, or place it at `X-Stream/data` and change the example `--input` and `--video-root` paths from `../data` to `data`.

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

### 3. Local Runtime Variables

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

### 4. Model Checkpoints

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

### 5. (Optional) CLIP Weights For Pruning Modes

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

## Usage

Start with the API-free smoke test, then add vLLM, hosted API models, or evaluation as needed.

### 0. (Optional) CPU Cache Prewarming

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
  --prompt-root prompts/streaming_prompt \
  --video-root ../data
```

`--warm-cache-only` does not start vLLM and does not call any model. It only resolves `{{video:...}}` placeholders and writes the segment cache. Keep `--cache-dir`, `--input`, `--video-root`, and video placeholder parameters identical between prewarming and the later GPU run.

For `time`, `cdpruner`, `surge`, `cdpruner_token`, and `surge_token`, `--multi-stream` can be omitted during prewarming because the same base video segments are generated. For `code_adaptive`, pass the same `--multi-stream code_adaptive` as the later run because it may create additional fps-scaled segments.

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
  --prompt-root prompts/streaming_prompt \
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
  --prompt-root prompts/streaming_prompt \
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
  --prompt-root prompts/streaming_prompt \
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
  --xstream-rho 0.25 \
  --tp 2 \
  --workers 1 \
  --max-model-len 200000 \
  --prompt-root prompts/streaming_prompt \
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
  --prompt-root prompts/streaming_prompt \
  --video-root ../data \
  --run-id api_qwen3vl_pixel
```

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
|-- LICENSE
|-- assets/
|   |-- logo.png                    # X-Stream logo used in this README
|   |-- teaser.png                  # Scenario overview figure
|   `-- multiplexing_pipeline.png   # Online inference and multiplexing pipeline
|-- run.sh                         # Main entrypoint for inference runs
|-- pipeline.sh                    # vLLM startup, resume, evaluation, cleanup
|-- pyproject.toml                 # uv environment and dependency pins
|-- configs/
|   |-- models.example.json        # Public model-config template
|   `-- models.json                # Local model config
|-- prompts/
|   |-- streaming_prompt/
|   |   `-- system_prompt.txt      # Streaming QA prompt resolved by {{file:system_prompt.txt}}
|   `-- general_prompt/
|       `-- system_prompt.txt      # General-purpose prompt for custom runs
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
2. GPT: 85 tokens/frame + 170 tokens per 512 $\times$ 512 tile.
3. Qwen3+: 28 $\times$ 28 pixel patches per token with token merging.

Use these rules to estimate the effective token rate for your target model, then choose the resolution, FPS, clip length, or playback-speed adjustment that keeps the input within that model's limit.

## FAQ

### Which dataset file should I use?

Use `../data/eval_relative_merged_phostream_type.jsonl` for `pixel`. Use `../data/eval_relative_multi_phostream_type.jsonl` for all multi-stream modes. Use `../data/eval_relative.json` when you need to inspect or validate the dataset manifest itself.

### Why does `eval_relative.json` not appear in `run.sh` examples?

`eval_relative.json` is the release manifest. The inference runner consumes the MLLMFlow-ready JSONL task files, so the executable examples use `eval_relative_merged_phostream_type.jsonl` or `eval_relative_multi_phostream_type.jsonl`.

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
