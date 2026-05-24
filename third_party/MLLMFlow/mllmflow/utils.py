import os
import hashlib
import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mllmflow.video_utils.trim_video import trim_video
from mllmflow.video_utils.probe_video import probe_video
from mllmflow.time_utils.time_utils import seconds_to_time

def _parse_args_str(args_str: str) -> Tuple[str, Dict[str, str]]:
    if not args_str:
        return "", {}
    parts = args_str.split(",", 1)
    resource = parts[0].strip()
    kwargs = {
        k.strip(): v.strip()
        for kv in (parts[1].split(",") if len(parts) > 1 else [])
        if "=" in kv
        for k, v in [kv.split("=", 1)]
    }
    return resource, kwargs


def _merge_text_parts(parts: List[Dict]) -> List[Dict]:
    merged = []
    for part in parts:
        if part["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1]["text"] += part["text"]
        else:
            merged.append(copy.deepcopy(part))
    return merged


def _construct_conversation(parts: List[Dict]) -> List[Dict]:
    messages = []
    current = None
    for p in parts:
        if p["type"] == "role":
            current = {"role": p["role"], "content": []}
            messages.append(current)
        else:
            if current is None:
                raise ValueError("Content part before any role")
            current["content"].append(p)
    return messages

def _apply_media_limit(parts: List[Dict], media_limit: int) -> List[Dict]:
    media_types = {"image", "video", "audio"}

    # Allow audio to be globally dropped through an environment variable for upstream scripts
    drop_audio = os.environ.get("FLOW_DROP_AUDIO", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if drop_audio:
        # Replace all audio items with empty text so audio content is not sent to the model
        parts = [
            {"type": "text", "text": ""} if p.get("type") == "audio" else p
            for p in parts
        ]
        # Audio has already been cleared, so only image/video count limits are applied here
        media_types = {"image", "video"}

    media_indices = [i for i, p in enumerate(parts) if p.get("type") in media_types]

    if media_limit <= 0:
        keep_set = set()
    else:
        keep_set = set(media_indices[-media_limit:])

    return [
        p
        if (p.get("type") not in media_types or i in keep_set)
        else {"type": "text", "text": ""}
        for i, p in enumerate(parts)
    ]


# ---- CDPruner-style media limit implementation ----
_CDPRUNER_CLIP_MODEL = "openai/clip-vit-large-patch14-336"
_cdpruner_initialized = False
_cdpruner_device = None
_cdpruner_image_processor = None
_cdpruner_vision_model = None
_cdpruner_text_tokenizer = None
_cdpruner_text_model = None


def _init_cdpruner_clip():
    """Lazily initialize the CLIP model and tokenizer for CDPruner-style filtering.

    If dependencies (torch/transformers/PIL) are missing or model loading fails, raise an exception so the caller can fall back to the default logic.
    """
    global _cdpruner_initialized
    global _cdpruner_device
    global _cdpruner_image_processor
    global _cdpruner_vision_model
    global _cdpruner_text_tokenizer
    global _cdpruner_text_model

    if _cdpruner_initialized:
        return

    import torch  # type: ignore
    # Import directly from submodules to avoid import failures when some transformers versions do not export CLIP* from top-level __all__
    from transformers.models.clip import (  # type: ignore
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTokenizerFast,
        CLIPTextModelWithProjection,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Avoid .to(device) failures caused by from_pretrained using meta placeholders by forcing real weights to load on CPU before moving them
    load_kw = {"device_map": None, "low_cpu_mem_usage": False}

    image_processor = CLIPImageProcessor.from_pretrained(_CDPRUNER_CLIP_MODEL)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(
        _CDPRUNER_CLIP_MODEL, **load_kw
    )
    text_tokenizer = CLIPTokenizerFast.from_pretrained(_CDPRUNER_CLIP_MODEL)
    text_model = CLIPTextModelWithProjection.from_pretrained(
        _CDPRUNER_CLIP_MODEL, **load_kw
    )

    vision_model.to(device)
    text_model.to(device)
    vision_model.eval()
    text_model.eval()

    _cdpruner_device = device
    _cdpruner_image_processor = image_processor
    _cdpruner_vision_model = vision_model
    _cdpruner_text_tokenizer = text_tokenizer
    _cdpruner_text_model = text_model
    _cdpruner_initialized = True


def _apply_media_limit_cdpruner(
    parts: List[Dict],
    media_limit: int,
    instruction_text: str = "",
) -> List[Dict]:
    """Use CDPruner-inspired filtering when the number of image/video tokens exceeds media_limit.

    - Use CLIP to extract visual and text embeddings for each multimedia token;
    - Build a kernel matrix based on instruction relevance;
    - Use DPP-style greedy MAP inference to select a subset;
    - Only applies to image/video; audio is still controlled globally by the FLOW_DROP_AUDIO environment variable.

    Dependencies: torch, transformers, and PIL. If any is missing, automatically fall back to _apply_media_limit.
    """
    media_types = {"image", "video", "audio"}

    drop_audio = os.environ.get("FLOW_DROP_AUDIO", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if drop_audio:
        parts = [
            {"type": "text", "text": ""} if p.get("type") == "audio" else p
            for p in parts
        ]
        media_types = {"image", "video"}

    media_indices = [i for i, p in enumerate(parts) if p.get("type") in media_types]

    # No media or a zero limit: drop everything (no need to enter CDPruner logic)
    if media_limit <= 0 or not media_indices:
        keep_set = set()
        return [
            p
            if (p.get("type") not in media_types or i in keep_set)
            else {"type": "text", "text": ""}
            for i, p in enumerate(parts)
        ]

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        # Missing runtime dependencies: fall back to simple tail truncation
        print(
            f"[CDPruner] Missing torch/PIL, fallback to _apply_media_limit. Error: {e}",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    try:
        _init_cdpruner_clip()
    except Exception as e:
        # Also fall back when CLIP model loading fails
        print(
            f"[CDPruner] Failed to init CLIP model, fallback to _apply_media_limit. Error: {e}",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    assert _cdpruner_image_processor is not None
    assert _cdpruner_vision_model is not None
    assert _cdpruner_text_tokenizer is not None
    assert _cdpruner_text_model is not None
    assert _cdpruner_device is not None

    device = _cdpruner_device

    # Map each media token to one representative frame image (use the original image for image tokens and the middle frame for video tokens)
    from mllmflow.video_utils.extract_frame import (  # type: ignore
        extract_frame_ffmpeg,
    )

    images = []
    valid_media_indices: List[int] = []
    cache_dir = os.environ.get("FLOW_CACHE_DIR", "media_dir")

    for idx in media_indices:
        p = parts[idx]
        mtype = p.get("type")
        img_path: Optional[str] = None

        if mtype == "image":
            img_path = p.get("image")
        elif mtype == "video":
            video_path = p.get("video")
            if not video_path:
                continue
            try:
                duration = _probe_video_duration(video_path)
            except Exception:
                duration = 0
            mid_t = max(0.0, float(duration) / 2.0)
            fp = get_file_fingerprint(video_path)
            out_dir = Path(cache_dir) / "cdpruner_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{fp}_mid.jpg"
            if not out_path.exists():
                try:
                    ok = extract_frame_ffmpeg(video_path, str(out_path), mid_t)
                except Exception:
                    ok = False
                if not ok:
                    continue
            img_path = str(out_path)

        if not img_path or not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        images.append(img)
        valid_media_indices.append(idx)

    # If no media can be converted to images, fall back to the default strategy
    if not images:
        print(
            "[CDPruner] No media could be converted to images, fallback to _apply_media_limit.",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    N = len(images)
    if N <= 1:
        # When there are only 0/1 media items, pruning is not useful; use the default strategy directly
        return _apply_media_limit(parts, media_limit)

    # Control the keep ratio in CDPruner mode through an environment variable (relative to the current total media count N):
    # - FLOW_CDPRUNER_KEEP_RATIO ∈ (0, 1]：roughly keep ratio * N media items (also constrained by media_limit);
    # - ratio<=0：fall back to dropping everything;
    # - ratio>1：treat as 1.0 (no additional ratio-based trimming, only constrained by media_limit).
    ratio_env = os.environ.get("FLOW_CDPRUNER_KEEP_RATIO", "").strip()
    try:
        keep_ratio = float(ratio_env) if ratio_env else 0.5
    except ValueError:
        keep_ratio = 0.5
    keep_ratio = max(0.0, min(1.0, keep_ratio))

    # Normally media_limit <= 0 has already been handled above (drop everything), so assume media_limit > 0 here
    if keep_ratio <= 0.0:
        effective_limit = 0
    elif keep_ratio >= 1.0:
        # Do not additionally trim by ratio; only constrain by media_limit
        effective_limit = min(media_limit, N)
    else:
        # Estimate the number to keep by ratio and constrain it by both media_limit and N; keep at least 1
        effective_limit = int(N * keep_ratio)
        if effective_limit < 1:
            effective_limit = 1
        effective_limit = min(effective_limit, media_limit, N)

    # Use CLIP to extract features
    pixel_values = _cdpruner_image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        vision_out = _cdpruner_vision_model(pixel_values)
        # Prefer image_embeds; otherwise fall back to the CLS token
        if hasattr(vision_out, "image_embeds") and vision_out.image_embeds is not None:
            image_embeds = vision_out.image_embeds  # (N, D)
        else:
            last_hidden = vision_out.last_hidden_state  # type: ignore[attr-defined]
            image_embeds = last_hidden[:, 0, :]  # (N, D)

        if not instruction_text:
            instruction_text = ""

        text_inputs = _cdpruner_text_tokenizer(
            text=[instruction_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_out = _cdpruner_text_model(**text_inputs)
        if hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
            text_embeds = text_out.text_embeds  # (1, D)
        else:
            last_hidden_t = text_out.last_hidden_state  # type: ignore[attr-defined]
            text_embeds = last_hidden_t[:, 0, :]

    # ---- Build the kernel matrix and perform DPP greedy selection following the core CDPruner idea ----
    N, D = image_embeds.shape

    # Normalize
    img_norm = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-6)
    txt_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)

    # Inter-image similarity
    similarity = torch.matmul(img_norm, img_norm.t())  # (N, N)

    # Instruction relevance (following CDPruner, negate cosine similarity and normalize it)
    relevance = torch.matmul(img_norm, txt_norm.t()).squeeze(-1)  # (N,)
    relevance = -relevance
    relevance = (relevance - relevance.min() + 1e-6) / (
        relevance.max() - relevance.min() + 1e-6
    )

    # Conditional DPP kernel matrix
    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)  # (N, N)

    # Fast MAP inference（B=1 special case, simplified from the original CDPruner implementation）
    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()  # (N,)
    cis = torch.zeros((effective_limit, N), device=kernel.device)  # (T, N)
    select_idx = torch.empty((effective_limit,), dtype=torch.long, device=kernel.device)

    for i in range(effective_limit):
        j = torch.argmax(di2s)
        select_idx[i] = j
        if di2s[j] <= 0:
            di2s[j] = -float("inf")
            continue
        ei = (kernel[j] - torch.matmul(cis[:i, j], cis[:i])) / torch.sqrt(
            di2s[j] + 1e-6
        )
        cis[i] = ei
        di2s -= ei * ei
        di2s[j] = -float("inf")

    # Deduplicate and sort, then map back to the original media indices
    unique_idx = torch.unique(select_idx.cpu())
    if unique_idx.numel() > effective_limit:
        unique_idx = unique_idx[:effective_limit]
    keep_positions = sorted(int(x) for x in unique_idx.tolist())
    keep_set = {valid_media_indices[pos] for pos in keep_positions}

    return [
        p
        if (p.get("type") not in media_types or i in keep_set)
        else {"type": "text", "text": ""}
        for i, p in enumerate(parts)
    ]


def _compute_surge_temporal(
    features,
    rho: float = 0.25,
    ema_gamma: float = 0.9,
    ema_var_decay: float = 0.9,
    epsilon: float = 1e-8,
    enable_variance_norm: bool = True,
):
    """Simplified SURGE: model surprise only along the temporal dimension, assuming each token corresponds to global features at one time step.

    This implementation follows the core idea of the official SURGE `compute_surge`: 
    - Use two-frame differences to model short-term trends and apply variance normalization;
    - Use a quantile threshold to control the keep ratio (rho).
    """
    import torch  # type: ignore

    if features.dim() != 2:
        raise ValueError(f"_compute_surge_temporal expects [T, D] features, got shape {tuple(features.shape)}")

    device = features.device
    T, _ = features.shape
    compute_dtype = torch.float32
    feats = features.to(compute_dtype)

    scores = torch.zeros(T, device=device, dtype=compute_dtype)
    running_var = None

    for t in range(T):
        if t == 0:
            scores[t] = float("inf")
            continue

        if t == 1:
            error = feats[t] - feats[t - 1]
            score = (error * error).sum()
            running_var = score.clone()
            if enable_variance_norm:
                scores[t] = score / (running_var + epsilon)
            else:
                scores[t] = score
            continue

        raw_delta = feats[t - 1] - feats[t - 2]
        pred = feats[t - 1] + raw_delta
        error = feats[t] - pred
        score = (error * error).sum()

        if enable_variance_norm:
            running_var = ema_var_decay * running_var + (1 - ema_var_decay) * score
            scores[t] = score / (running_var + epsilon)
        else:
            scores[t] = score

    finite_scores = scores[torch.isfinite(scores)]
    if finite_scores.numel() > 0 and 0.0 < rho < 1.0:
        threshold = torch.quantile(finite_scores, 1.0 - rho)
        keep_mask = scores >= threshold
    else:
        keep_mask = torch.ones_like(scores, dtype=torch.bool)

    # Smooth the surprise curve for diagnostics; the current selection logic only depends on the raw scores
    if T > 1:
        smoothed = torch.zeros_like(scores)
        smoothed[0] = scores[0]
        for t in range(1, T):
            smoothed[t] = ema_gamma * smoothed[t - 1] + (1 - ema_gamma) * scores[t]
    return keep_mask


def _apply_media_limit_surge(
    parts: List[Dict],
    media_limit: int,
) -> List[Dict]:
    """SURGE-style video token reduction.

    Key implementation points:
    - Only applies to multimedia tokens with `type=="video"`; image/audio behavior matches the default strategy;
    - For multiple clips from the same video path, extract representative frames and use CLIP to obtain temporal sequence features;
    - Apply the simplified SURGE algorithm in chronological order within each video stream to obtain time steps to keep;
    - If the total retained count still exceeds media_limit, fall back to keeping the last media_limit video tokens;
    - Replace dropped video tokens with empty text so those clips are no longer sent to downstream models.
    """
    media_types = {"image", "video", "audio"}

    drop_audio = os.environ.get("FLOW_DROP_AUDIO", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if drop_audio:
        parts = [
            {"type": "text", "text": ""} if p.get("type") == "audio" else p
            for p in parts
        ]
        media_types = {"image", "video"}

    # Apply SURGE only to video tokens; other media keeps the count-limit strategy
    video_indices = [i for i, p in enumerate(parts) if p.get("type") == "video"]

    if media_limit <= 0 or not video_indices:
        keep_set = set()
        return [
            p
            if (p.get("type") != "video" or i in keep_set)
            else {"type": "text", "text": ""}
            for i, p in enumerate(parts)
        ]

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        print(
            f"[SURGE] Missing torch/PIL, fallback to _apply_media_limit. Error: {e}",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    try:
        _init_cdpruner_clip()
    except Exception as e:
        print(
            f"[SURGE] Failed to init CLIP model, fallback to _apply_media_limit. Error: {e}",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    assert _cdpruner_image_processor is not None
    assert _cdpruner_vision_model is not None
    assert _cdpruner_device is not None

    device = _cdpruner_device

    from mllmflow.video_utils.extract_frame import (  # type: ignore
        extract_frame_ffmpeg,
    )

    cache_dir = os.environ.get("FLOW_CACHE_DIR", "media_dir")

    # 1) Extract one frame image for each video token and compute CLIP features
    frame_images: List["Image.Image"] = []
    frame_meta: List[Tuple[int, str]] = []  # (global_index, video_path)

    for idx in video_indices:
        p = parts[idx]
        video_path = p.get("video")
        if not video_path:
            continue
        try:
            duration = _probe_video_duration(video_path)
        except Exception:
            duration = 0
        mid_t = max(0.0, float(duration) / 2.0)
        fp = get_file_fingerprint(video_path)
        out_dir = Path(cache_dir) / "surge_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fp}_mid.jpg"
        if not out_path.exists():
            try:
                ok = extract_frame_ffmpeg(video_path, str(out_path), mid_t)
            except Exception:
                ok = False
            if not ok:
                continue
        img_path = str(out_path)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")  # type: ignore[name-defined]
        except Exception:
            continue
        frame_images.append(img)
        frame_meta.append((idx, video_path))

    if not frame_images:
        print(
            "[SURGE] No video frames extracted, fallback to _apply_media_limit.",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    pixel_values = _cdpruner_image_processor(images=frame_images, return_tensors="pt")[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        vision_out = _cdpruner_vision_model(pixel_values)
        if hasattr(vision_out, "image_embeds") and vision_out.image_embeds is not None:
            frame_embeds = vision_out.image_embeds  # (N, D)
        else:
            last_hidden = vision_out.last_hidden_state  # type: ignore[attr-defined]
            frame_embeds = last_hidden[:, 0, :]  # (N, D)

    # 2) Group by video path, form temporal sequences by occurrence order within each group, and apply SURGE
    rho_env = os.environ.get("FLOW_SURGE_RHO", "").strip()
    try:
        surge_rho = float(rho_env) if rho_env else 0.25
    except ValueError:
        surge_rho = 0.25
    surge_rho = max(0.0, min(1.0, surge_rho))

    by_video: Dict[str, List[Tuple[int, int]]] = {}
    for feat_idx, (global_idx, vpath) in enumerate(frame_meta):
        by_video.setdefault(vpath, []).append((global_idx, feat_idx))

    keep_video_indices: List[int] = []

    for vpath, items in by_video.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        feat_indices = [feat_idx for _, feat_idx in items_sorted]
        feats = frame_embeds[feat_indices]
        if feats.shape[0] == 0:
            continue
        keep_mask = _compute_surge_temporal(feats, rho=surge_rho)
        for (global_idx, _), keep_flag in zip(items_sorted, keep_mask.bool().tolist()):
            if keep_flag:
                keep_video_indices.append(global_idx)

    if not keep_video_indices:
        # When all frames are judged low-surprise, keep at least the last video token to avoid losing the video entirely
        keep_video_indices = [video_indices[-1]]

    keep_video_indices = sorted(set(keep_video_indices))
    if len(keep_video_indices) > media_limit:
        keep_video_indices = keep_video_indices[-media_limit:]

    keep_set = set(keep_video_indices)

    return [
        p
        if (p.get("type") != "video" or i in keep_set)
        else {"type": "text", "text": ""}
        for i, p in enumerate(parts)
    ]


def _probe_video_duration(video_path: str) -> int:
    info = probe_video(video_path)
    if info is None:
        raise RuntimeError(
            f"Unable to probe video duration (ffprobe failed and probe returned empty): {video_path!s} — "
            "Please confirm the file exists, is a valid video, and `ffprobe` is available locally (installed with ffmpeg)."
        )
    d = info.get("duration")
    if d is None:
        return 0
    return int(float(d))


def get_file_fingerprint(filepath):
    filepath = str(Path(filepath).resolve())
    key = "/".join(filepath.split("/")[-4::])
    hash_value = hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]
    return hash_value
    # stat= os.stat(filepath)
    # fingerprint = f"{stat.st_dev:x}_{stat.st_ino:x}"
    # return fingerprint

def _gen_clip_path(video_path: str, start: int, end: int, fps: Optional[float]) -> str:
    s = seconds_to_time(start).replace(":", "")
    e = seconds_to_time(end).replace(":", "")
    fps_part = f"_fps_{fps}" if fps is not None else ""
    fingerprint = get_file_fingerprint(video_path)
    return f"{fingerprint}/{s}_{e}{fps_part}"

def _trim_video_cached(
    video_path: str,
    start: int,
    end: int,
    cache_dir: str,
    fps: Optional[float] = None,
) -> str:
    key = _gen_clip_path(video_path, start, end, fps)
    output = Path(cache_dir) / f"{key}.mp4"

    os.makedirs(output.parent, exist_ok=True)

    if output.exists():
        return str(output.resolve())

    lock_dir = output.with_suffix(output.suffix + ".lock")
    while True:
        if output.exists():
            return str(output.resolve())
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            # A previous killed prewarm may leave a stale lock behind.
            try:
                if time.time() - lock_dir.stat().st_mtime > 6 * 60 * 60:
                    lock_dir.rmdir()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            time.sleep(0.2)

    try:
        if not output.exists():
            trim_video(
                video_path=video_path,
                trim_path=str(output),
                start_time=seconds_to_time(start),
                end_time=seconds_to_time(end),
                temp_dir=str(Path(cache_dir) / "moviepy_tmp"),
                fps=fps,
            )
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass
    return str(output.resolve())
