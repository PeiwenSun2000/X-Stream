# xstream_vllm_pruner

Patch-level token pruner for X-Stream's local vLLM backend. Reproduces
[SURGE](https://github.com/) and [CDPruner](https://github.com/) at the
**single-frame patch granularity**, replacing vLLM's built-in EVS retention
mask. The plugin is fully isolated: it does not modify any vLLM source file,
does not touch hosted-API model adapters, and is a no-op unless explicitly
enabled.

## When does this kick in?

Only when **all** of the following hold:

1. `XSTREAM_VLLM_PRUNER=1` is exported before `tools/vllm_cli.py serve` runs
   (set automatically by `pipeline.sh` when the X-Stream multi-stream mode is
   `cdpruner_token` or `surge_token`).
2. The vLLM CLI is launched with `--video-pruning-rate <rho>` for `rho > 0`
   (also set automatically by `pipeline.sh`).
3. The model is a Qwen multimodal architecture supported by vLLM. The
   plugin auto-detects which seam each model exposes:

   | Family | vLLM module | Seam used | Mode |
   | --- | --- | --- | --- |
   | Qwen2.5-VL | `qwen2_5_vl` | `compute_retention_mask` (EVS) | **hard** (KV-cache shrinks) |
   | Qwen3-VL | `qwen3_vl` | `compute_retention_mask` (EVS) | **hard** |
   | Qwen3-VL MoE | `qwen3_vl_moe` | `compute_retention_mask` (EVS) | **hard** |
   | Qwen2-VL | `qwen2_vl` | `_process_video_input` wrap | **soft** (dropped patches zeroed) |
   | Qwen2.5-Omni Thinker | `qwen2_5_omni_thinker` | `_process_video_input` wrap | **soft** |
   | Qwen3-Omni MoE Thinker | `qwen3_omni_moe_thinker` (inherits `Qwen2_5OmniConditionalGenerationMixin`) | `_process_video_input` wrap | **soft** |

   "Hard" mode removes pruned tokens from the LLM sequence entirely (this is
   the upstream EVS contract). "Soft" mode leaves the token count untouched
   and zeroes the embedding of dropped patches — vLLM's placeholder /
   `_merge_multimodal_embeddings` accounting therefore stays consistent on
   models that have no EVS support. Soft mode does not shrink the KV-cache
   but still removes visual signal at the per-patch granularity that SURGE
   and CDPruner require.

If none of the supported seams matches the loaded model, vLLM falls back to
its native EVS behaviour (or skips pruning entirely when
`--video-pruning-rate` is not provided).

## Architecture overview

```
client (MLLMFlow)
  └─ multi_stream_mode = cdpruner_token / surge_token
        ├─ request_params._xstream_pruner   ──┐
        └─ doubao adapter                     ▼
                payload.mm_processor_kwargs.xstream_instruction
                          │
                          ▼
              vLLM OpenAI server (one process)
                          │  → patched Qwen MultiModalProcessor
                          │       sets ContextVar (best-effort)
                          ▼
              vLLM workers (separate processes)
                          │  Qwen{2.5,3}VL._postprocess_video_embeds_evs
                          ▼
              patched vllm.multimodal.evs.compute_retention_mask
                          │
                          └─ SURGE / CDPruner (this package)
```

The seams we depend on are all stable across the v0.10+ series of vLLM and
are listed below. Unknown architectures fall through to native EVS (or no
pruning when EVS is not enabled):

- `vllm.multimodal.evs.compute_retention_mask` (replaced) — drives the
  *hard* EVS-style prune for the Qwen2.5-VL / Qwen3-VL / Qwen3-VL MoE
  families.
- `Qwen2VLForConditionalGeneration._process_video_input` (wrapped) — drives
  the *soft* prune for Qwen2-VL.
- `Qwen2_5OmniConditionalGenerationMixin._process_video_input` (wrapped) —
  drives the *soft* prune for Qwen2.5-Omni Thinker and, by inheritance via
  `Qwen3OmniMoeConditionalGenerationMixin`, Qwen3-Omni MoE Thinker.
- `Qwen{2_5,3}VLMultiModalProcessor._call_hf_processor` and the
  `Qwen{2_5,3}Omni*MultiModalProcessor` counterparts (wrapped) — strip
  `xstream_instruction` from `mm_kwargs` and bind it to a `ContextVar` so
  CDPruner can read the user instruction.

## Environment variables

| Name | Default | Meaning |
| --- | --- | --- |
| `XSTREAM_VLLM_PRUNER` | `0` | Master switch (`1` to enable). |
| `XSTREAM_VLLM_PRUNER_ALGO` | `surge` | `surge` or `cdpruner`. |
| `XSTREAM_VLLM_PRUNER_RHO` | `0.25` | Pruning rate forwarded to vLLM's `--video-pruning-rate`; sets the fraction of tokens *dropped*. |
| `XSTREAM_VLLM_PRUNER_KEEP_FIRST_FRAME` | `1` | Force-keep every patch of the first frame. |
| `XSTREAM_VLLM_PRUNER_DEBUG` | `0` | Log every retention-mask call (algo / shapes / kept count). |
| `XSTREAM_VLLM_PRUNER_CLIP_MODEL` | `openai/clip-vit-large-patch14-336` | CLIP text tower used by CDPruner. Loaded lazily; ignored for SURGE. |

The pruning *retention* count is **always** computed by
`vllm.multimodal.evs.compute_retained_tokens_count(...)` so the placeholder
accounting that vLLM did at prompt-stage stays consistent. Our SURGE /
CDPruner output is forced to that exact top-k.

## Algorithm notes

- **SURGE** (`surge.py`): inference-time port of
  [`SURGE/surge/surge_core.py`](../../../../SURGE/surge/surge_core.py). We
  keep drift correction (least-squares spatial detrending of the per-token
  delta) and EMA variance normalization but drop the offline diagnostics
  that need `scipy`. Frame 0 is forced into the keep set; the rest of the
  budget is filled by the global top-k of surprise scores.
- **CDPruner** (`cdpruner.py`): conditional DPP fast MAP greedy port of
  [`CDPruner/llava/model/llava_arch.py`](../../../../CDPruner/llava/model/llava_arch.py).
  Visual similarity is the cosine Gram of post-merger video embeddings.
  Relevance comes from a CLIP text embedding of the user instruction; the
  visual side is deterministically projected to CLIP-text dim by
  truncate / zero-pad. The kernel is `q · S · qᵀ`. Selection uses the
  standard fast greedy MAP from Chen et al.

## Known limitations

1. **CDPruner instruction relay only spans one process.** vLLM v1 launches
   workers in separate processes; the ContextVar we use to thread the
   instruction string from the API server to the model worker does not
   cross that boundary. When the instruction is unavailable in the worker,
   CDPruner gracefully degrades to **visual-only diversity DPP** (the
   relevance term becomes constant). This is still token-level diversity
   pruning, just non-conditional.
2. **Model coverage**: Qwen2.5-VL, Qwen3-VL, Qwen3-VL MoE, Qwen2-VL,
   Qwen2.5-Omni Thinker and Qwen3-Omni MoE Thinker are all supported. The
   EVS-capable families (Qwen2.5-VL / Qwen3-VL / Qwen3-VL MoE) get hard
   pruning; the rest get soft pruning that leaves token count unchanged but
   zeros dropped patches. Out-of-tree models that mirror the
   `_process_video_input` contract can be opted in at runtime via
   `xstream_vllm_pruner.patch_video_input.install_for_class(MyModelCls)`.
3. **Hosted-API models**: completely unsupported (there is no way to inject
   pruning logic into a third-party endpoint). MLLMFlow raises early when
   `multi_stream_mode ∈ {cdpruner_token, surge_token}` is paired with a
   non-vLLM backend.
4. **CLIP text tower memory**: CDPruner lazy-loads CLIP on first call. If
   you only want SURGE, the CLIP weights are never loaded.

## Manual smoke test

```bash
export XSTREAM_VLLM_PRUNER=1
export XSTREAM_VLLM_PRUNER_ALGO=surge
export XSTREAM_VLLM_PRUNER_RHO=0.5
export XSTREAM_VLLM_PRUNER_DEBUG=1
python3 inference/tools/vllm_cli.py serve <Qwen2.5-VL or Qwen3-VL path> \
    --tensor-parallel-size 1 --video-pruning-rate 0.5 --trust-remote-code
```

You should see one `xstream_vllm_pruner: ready (algo=surge ...)` line at
startup, followed by per-request `xstream_vllm_pruner: algo=surge T=... kept=...`
messages in debug mode.

## Disabling the plugin

Unset `XSTREAM_VLLM_PRUNER` (or set it to `0`). The plugin is then a
strict no-op and vLLM's original EVS behaviour returns unchanged.
