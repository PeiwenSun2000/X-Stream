import os
import hashlib
import copy
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

    # 允许通过环境变量全局丢弃 audio，供上游脚本控制
    drop_audio = os.environ.get("FLOW_DROP_AUDIO", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if drop_audio:
        # 将所有 audio 类型直接替换为空文本，从而不向模型发送音频内容
        parts = [
            {"type": "text", "text": ""} if p.get("type") == "audio" else p
            for p in parts
        ]
        # audio 已经被清空，这里只对 image / video 做数量限制
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


# ---- CDPruner 风格的 media limit 实现 ----
_CDPRUNER_CLIP_MODEL = "openai/clip-vit-large-patch14-336"
_cdpruner_initialized = False
_cdpruner_device = None
_cdpruner_image_processor = None
_cdpruner_vision_model = None
_cdpruner_text_tokenizer = None
_cdpruner_text_model = None


def _init_cdpruner_clip():
    """惰性初始化 CLIP 模型与 tokenizer，供 CDPruner 风格的筛选使用。

    若依赖（torch/transformers/PIL）缺失或模型加载失败，将抛出异常，由上层回退到默认逻辑。
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
    # 直接从子模块导入，避免部分 transformers 版本顶部 __all__ 未导出 CLIP* 导致 import 失败
    from transformers.models.clip import (  # type: ignore
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTokenizerFast,
        CLIPTextModelWithProjection,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 避免 from_pretrained 使用 meta 占位导致 .to(device) 报错，强制在 CPU 上加载真实权重再迁移
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
    """使用 CDPruner 思想在 image/video token 数量超过 media_limit 时进行筛选。

    - 利用 CLIP 抽取每个多媒体 token 的视觉与文本嵌入；
    - 构造基于指令相关性的核矩阵；
    - 采用 DPP 风格的贪心 MAP 推断选择子集；
    - 仅对 image/video 生效，audio 仍按环境变量 FLOW_DROP_AUDIO 控制是否整体丢弃。

    依赖：torch、transformers、PIL，若任一缺失则自动回退到 _apply_media_limit。
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

    # 媒体为 0 或限制为 0：全部丢弃（此时无需进入 CDPruner 逻辑）
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
        # 环境缺少依赖：退化到简单的尾部截断策略
        print(
            f"[CDPruner] Missing torch/PIL, fallback to _apply_media_limit. Error: {e}",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    try:
        _init_cdpruner_clip()
    except Exception as e:
        # CLIP 模型加载失败时，同样退化
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

    # 将每个 media token 映射为 1 张代表帧的图像（image 直接使用原图，video 截取中间帧）
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

    # 如果所有媒体都无法成功解析为图像，则退化为默认策略
    if not images:
        print(
            "[CDPruner] No media could be converted to images, fallback to _apply_media_limit.",
            flush=True,
        )
        return _apply_media_limit(parts, media_limit)

    N = len(images)
    if N <= 1:
        # 只有 0/1 个媒体时，无意义进行剪枝，直接按默认策略处理
        return _apply_media_limit(parts, media_limit)

    # 通过环境变量控制 CDPruner 模式下的保留比例（相对于当前媒体总数 N）：
    # - FLOW_CDPRUNER_KEEP_RATIO ∈ (0, 1]：大致保留 ratio * N 个媒体（再受 media_limit 约束）；
    # - ratio<=0：退化为全部丢弃；
    # - ratio>1：按 1.0 处理（即不额外按比例裁剪，只受 media_limit 约束）。
    ratio_env = os.environ.get("FLOW_CDPRUNER_KEEP_RATIO", "").strip()
    try:
        keep_ratio = float(ratio_env) if ratio_env else 0.5
    except ValueError:
        keep_ratio = 0.5
    keep_ratio = max(0.0, min(1.0, keep_ratio))

    # 正常情况下 media_limit<=0 已在上面处理（直接全部丢弃），这里假设 media_limit>0
    if keep_ratio <= 0.0:
        effective_limit = 0
    elif keep_ratio >= 1.0:
        # 不按比例额外裁剪，仅受 media_limit 约束
        effective_limit = min(media_limit, N)
    else:
        # 按比例估算要保留的数量，并与 media_limit/N 双重约束；至少保留 1 个
        effective_limit = int(N * keep_ratio)
        if effective_limit < 1:
            effective_limit = 1
        effective_limit = min(effective_limit, media_limit, N)

    # 使用 CLIP 抽特征
    pixel_values = _cdpruner_image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        vision_out = _cdpruner_vision_model(pixel_values)
        # 优先使用 image_embeds，否则退回到 CLS token
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

    # ---- 按照 CDPruner 核心思想构造核矩阵并进行 DPP 贪心选择 ----
    N, D = image_embeds.shape

    # 归一化
    img_norm = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-6)
    txt_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)

    # 图像间相似度
    similarity = torch.matmul(img_norm, img_norm.t())  # (N, N)

    # 指令相关性（参照 CDPruner，将余弦相似度取负并归一化）
    relevance = torch.matmul(img_norm, txt_norm.t()).squeeze(-1)  # (N,)
    relevance = -relevance
    relevance = (relevance - relevance.min() + 1e-6) / (
        relevance.max() - relevance.min() + 1e-6
    )

    # 条件 DPP 的核矩阵
    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)  # (N, N)

    # Fast MAP inference（B=1 的特例，简化自原始 CDPruner 实现）
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

    # 去重并排序，映射回原来的 media 索引
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
    """简化版 SURGE：只在时间维度上建模惊讶度，假设每个 token 对应一个时间步的全局特征。

    该实现遵循官方 SURGE `compute_surge` 的核心思路：
    - 使用两帧差分建模短期趋势，并进行方差归一化；
    - 使用分位数阈值控制保留比例（rho）。
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

    # 平滑惊讶度曲线（便于诊断；当前选择逻辑只依赖原始 scores）
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
    """SURGE 风格的 video token 削减。

    实现要点：
    - 仅对 `type=="video"` 的多媒体 token 生效，image/audio 行为与默认策略一致；
    - 对同一路视频（相同文件路径）的多个切片，抽取代表帧并用 CLIP 得到时间序列特征；
    - 在每路视频内部按时间顺序应用简化版 SURGE，得到需要保留的时间步；
    - 若整体保留数仍超过 media_limit，则退化为“保留最后 media_limit 个 video token”；
    - 被丢弃的视频 token 用空文本占位，从而不再向下游模型发送该片段。
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

    # 仅对 video token 做 SURGE，其他 media 沿用数量限制策略
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

    # 1) 为每个 video token 抽取一帧图像，并计算 CLIP 特征
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

    # 2) 按视频路径分组，内部按出现顺序形成时间序列，并应用 SURGE
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
        # 所有帧都被判为低惊讶度时，至少保留最后一个 video token，避免彻底丢失视频
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
            f"无法探测视频时长（ffprobe 失败，probe 返回空）: {video_path!s} — "
            "请确认文件存在、为有效视频，且本机可执行 `ffprobe`（随 ffmpeg 安装）。"
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

    trim_video(
        video_path=video_path,
        trim_path=str(output),
        start_time=seconds_to_time(start),
        end_time=seconds_to_time(end),
        temp_dir=str(Path(cache_dir) / "moviepy_tmp"),
        fps=fps,
    )
    return str(output.resolve())
