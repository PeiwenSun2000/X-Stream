# <img src="assets/logo.png" alt="X-Stream logo" width="10%"> X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding.

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://peiwensun2000.github.io/xstream/)
[![Dataset HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/spw2000/X-stream)
[![Paper ECCV 2026](https://img.shields.io/badge/Paper-ECCV%202026-red)](https://peiwensun2000.github.io/xstream/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official inference and evaluation code for **X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding**. This package runs online multi-stream video QA with local vLLM checkpoints or hosted API models.

![X-Stream teaser](assets/teaser.png)

## Abstract

X-Stream is a multi-stream streaming understanding benchmark for evaluating how multimodal large language models handle concurrent video streams. It contains 4,220 curated QA pairs across 932 videos and covers 11 subtasks in multi-window, multi-view, and multi-device scenarios. The paper frames current MLLMs as naive multiplexers and studies spatial, temporal, and semantic ways to combine multiple streams into one model-consumable token sequence.

## Pipeline

![X-Stream multiplexing pipeline](assets/multiplexing_pipeline.png)

```mermaid
flowchart LR
    inputJsonl["X-Stream JSONL"] --> runner["run.sh"]
    runner --> backend{"Backend"}
    backend -->|"local"| vllm["vLLM"]
    backend -->|"API"| modelApi["Hosted API"]
    vllm --> flow["MLLMFlow"]
    modelApi --> flow
    flow --> outputJsonl["Output JSONL"]
    outputJsonl --> evalStep{"stream-eval"}
    evalStep -->|"optional"| scores["eval.json"]
```

Supported multi-stream modes:

| Mode | Meaning | Input type |
| --- | --- | --- |
| `pixel` | Spatial division; use merged/tiled videos. | merged JSONL |
| `time` | Time division; interleave synchronized streams. | multi-stream JSONL |
| `code`, `code_adaptive` | Semantic stream selection. | multi-stream JSONL |
| `cdpruner`, `surge` | Token-reduction baselines. | multi-stream JSONL |

## Repository Layout

```text
inference/
|-- run.sh                  # main entrypoint
|-- pipeline.sh             # vLLM, resume, and evaluation helpers
|-- configs/
|   `-- models.example.json
|-- tests/
|   `-- make_samples.sh
|-- tools/
|-- third_party/
|   |-- MLLMFlow
|   |-- ModelHub
|   `-- stream-eval
`-- assets/
```

## Installation

Requirements:

- Linux with NVIDIA GPU support for local vLLM runs.
- `uv >= 0.4`.
- Python 3.12, resolved by `uv`.

Install dependencies:

```bash
cd X-Stream-open-source/inference
uv sync --extra local
```

Run commands either through `uv run` or an activated environment:

```bash
source .venv/bin/activate
bash run.sh --help
```

## Data And Model Setup

The inference scripts expect MLLMFlow-ready JSONL files. Prepared evaluation files are included in the repository:

```text
../data/v1/eval_relative_merged_phostream_type.jsonl
../data/v1/eval_relative_multi_phostream_type.jsonl
../data/v2/strict/eval_relative_merged_phostream_type.jsonl
../data/v2/strict/eval_relative_multi_phostream_type.jsonl
../data/v2/loose/eval_relative_merged_phostream_type.jsonl
../data/v2/loose/eval_relative_multi_phostream_type.jsonl
```

The v3 dataset release uses manifest files such as `../data/v3/eval_relative.json`; see `../data/v3/readme.md` for the dataset format. Convert v3 manifests to the MLLMFlow JSONL format before using them with this runner.

Create a local model config:

```bash
cp configs/models.example.json configs/models.json
```

Set API keys only for hosted models:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
export QWEN_ENDPOINT=...
export QWEN_API_KEY=...
```

## Quickstart

### Smoke Test

```bash
uv run bash tests/make_samples.sh 10
uv run bash run.sh \
  --model echo \
  --no-vllm \
  --input tests/sample_10_merged.jsonl \
  --multi-stream pixel \
  --no-stream-eval \
  --workers 2 \
  --video-root ../data/v1
```

### Local vLLM Model

```bash
uv run bash run.sh \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --vllm-model-path /path/to/checkpoint \
  --input ../data/v1/eval_relative_multi_phostream_type.jsonl \
  --multi-stream time \
  --tp 2 \
  --workers 4 \
  --max-model-len 65536 \
  --video-root ../data/v1
```

### Hosted API Model

```bash
uv run bash run.sh \
  --model qwen3-vl-30b-a3b-instruct \
  --no-vllm \
  --input ../data/v1/eval_relative_merged_phostream_type.jsonl \
  --multi-stream pixel \
  --workers 8 \
  --video-root ../data/v1
```

## Outputs And Evaluation

Each run writes to `outputs/<RUN_ID>_<YYYYMMDD-HHMMSS>/`:

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
- `--no-stream-eval`: skip LLM-as-judge evaluation.
- `--stream-eval-judger MODEL`: choose the judge model.
- `--output-dir DIR`: change the output root.

## Troubleshooting

- **vLLM is not ready**: check `outputs/<run>/vllmlogs/<port>.log`; reduce `--gpu-mem-util` or `--max-model-len`, or increase `--tp`.
- **Input path fails**: verify `--input` and `--video-root`. `run.sh` converts paths to absolute paths before launching workers.
- **API errors**: check API keys, endpoint URLs, and multimodal quota in `configs/models.json`.
- **Wrong multi-stream behavior**: use `eval_relative_multi_phostream_type.jsonl` for `time`, `code`, `cdpruner`, and `surge`.

## Citation

```bibtex
@inproceedings{sun2026xstream,
  title     = {X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding},
  author    = {Sun, Peiwen and Lu, Xudong and Liu, Huadai and Bo, Yang and Wu, Dongming and Guan, Huankang and Cai, Minghong and Chen, Jinpeng and Guo, Xintong and Li, Shuhan and Liu, Rui and Yue, Xiangyu},
  booktitle = {ECCV},
  year      = {2026}
}
```

## License

This inference package is released under the [MIT License](LICENSE). Third-party packages under `third_party/` keep their original licenses and notices.
