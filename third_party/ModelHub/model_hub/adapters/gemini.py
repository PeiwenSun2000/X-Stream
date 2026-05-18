from textwrap import indent
from ..model_hub import ModelClient, register_adapter
from ..utils import to_base64
from typing import List, Dict, Any

# Gemini 单次请求最多 10 个视频，超过时均匀采样 10 个
GEMINI_MAX_VIDEOS = 10


def _is_video_part(part: Dict[str, Any]) -> bool:
    if part.get("video_metadata") is not None:
        return True
    inline = part.get("inline_data") or {}
    return (inline.get("mime_type") or "").startswith("video/")


def _uniform_sample_video_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """若视频 part 超过 GEMINI_MAX_VIDEOS，则均匀取 GEMINI_MAX_VIDEOS 个，保留非视频 part 不变。"""
    video_indices = [i for i, p in enumerate(parts) if _is_video_part(p)]
    n = len(video_indices)
    if n <= GEMINI_MAX_VIDEOS:
        return parts
    # 在 [0, n-1] 上均匀取 GEMINI_MAX_VIDEOS 个下标（均匀采样，不是取前 10 个）
    if GEMINI_MAX_VIDEOS <= 1:
        selected_positions = [0] if GEMINI_MAX_VIDEOS else []
    else:
        selected_positions = [round(i * (n - 1) / (GEMINI_MAX_VIDEOS - 1)) for i in range(GEMINI_MAX_VIDEOS)]
    keep_index_set = {video_indices[p] for p in selected_positions}
    return [p for i, p in enumerate(parts) if not _is_video_part(p) or i in keep_index_set]


def _drop_first_n_video_parts(contents: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """从整次请求的开头丢弃前 n 个视频 part（按顺序），保留后面的内容，用于 body too large 时逐帧裁剪重试。"""
    if n <= 0:
        return contents
    video_locations = []
    for ci, content in enumerate(contents):
        for pi, part in enumerate(content.get("parts", [])):
            if _is_video_part(part):
                video_locations.append((ci, pi))
    drop_set = set(video_locations[:n])
    result = []
    for ci, content in enumerate(contents):
        new_parts = [
            part for pi, part in enumerate(content.get("parts", []))
            if not _is_video_part(part) or (ci, pi) not in drop_set
        ]
        if new_parts:
            result.append({**content, "parts": new_parts})
    return result


def _limit_payload_videos_to_max(contents: List[Dict[str, Any]], max_videos: int = GEMINI_MAX_VIDEOS) -> List[Dict[str, Any]]:
    """整次请求中视频 part 总数不超过 max_videos，超过则在全请求内均匀采样 max_videos 个。"""
    # 收集所有 (content_idx, part_idx) 且该 part 为视频
    video_locations = []
    for ci, content in enumerate(contents):
        for pi, part in enumerate(content.get("parts", [])):
            if _is_video_part(part):
                video_locations.append((ci, pi))
    n = len(video_locations)
    if n <= max_videos:
        return contents
    # 均匀取 max_videos 个
    if max_videos <= 1:
        keep_locations = set(video_locations[:1]) if max_videos else set()
    else:
        selected_positions = [round(i * (n - 1) / (max_videos - 1)) for i in range(max_videos)]
        keep_locations = {video_locations[p] for p in selected_positions}
    # 每个 content 只保留被选中的视频 part 及全部非视频 part；若某 content 筛后为空则跳过，避免把“整段原 content”发出去导致仍超 10 个视频
    result = []
    for ci, content in enumerate(contents):
        new_parts = [
            part for pi, part in enumerate(content.get("parts", []))
            if not _is_video_part(part) or (ci, pi) in keep_locations
        ]
        if new_parts:
            result.append({**content, "parts": new_parts})
        # else: 不追加原 content，否则会再次带入全部视频，总视频数仍可超过 10
    return result


@register_adapter("gemini")
class GeminiAdapter(ModelClient):
    def format_messages(self, context: List[Dict[str, Any]]):
        system_instruction = None
        contents = []

        for msg in context:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                if isinstance(content, str):
                    text = content
                else:
                    text = " ".join(
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    )
                if system_instruction is None:
                    system_instruction = text
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            items = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for item in items:
                typ = item.get("type")
                if typ == "video":
                    video_path = item["video"]
                    b64 = to_base64(video_path, self.max_video_size_bytes)
                    if b64:
                        parts.append({
                            "inline_data": {"mime_type": "video/mp4", "data": b64},
                            "video_metadata": {"fps": 1}
                        })
                elif typ == "image":
                    img_path = item["image"]
                    b64 = to_base64(img_path, self.max_video_size_bytes)
                    if b64:
                        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
                elif typ == "text":
                    text = item.get("text", "").strip()
                    # Gemini 要求 part 的 data oneof 必须有一个已初始化字段，空字符串视为未初始化，会报错
                    if text:
                        parts.append({"text": text})

            parts = _uniform_sample_video_parts(parts)
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return {"contents": contents, "system_instruction": system_instruction}

    def build_payload(self, messages, request_params: Dict[str, Any]) -> Dict[str, Any]:
        # messages 在这里是 format_messages 返回的格式化结果
        formatted = messages if isinstance(messages, dict) else {"contents": messages, "system_instruction": None}

        # 验证 contents 不为空
        if not formatted.get("contents"):
            raise ValueError("Contents cannot be empty")

        # body too large 时从开头丢弃前 N 个视频，再限制总数
        drop_n = request_params.get("_drop_first_n_videos", 0)
        contents = _drop_first_n_video_parts(formatted["contents"], drop_n)
        if not contents:
            raise ValueError("Contents empty after dropping video parts")
        contents = _limit_payload_videos_to_max(contents, GEMINI_MAX_VIDEOS)
        video_part_count = sum(1 for c in contents for p in c.get("parts", []) if _is_video_part(p))

        payload = {"contents": contents}
        if formatted.get("system_instruction"):
            payload["system_instruction"] = {"parts": [{"text": formatted["system_instruction"]}]}

        # 直接添加 request_params 到 payload（跳过内部键如 _drop_first_n_videos）
        for k, v in request_params.items():
            if k in ("contents", "system_instruction") or (k.startswith("_")):
                continue
            payload[k] = v

        # 构建请求头：gemini 使用 x-goog-api-key
        headers = {}
        if self.api_key and "{api_key}" not in self.endpoint:
            headers["x-goog-api-key"] = self.api_key

        return {
            "headers": headers,
            "payload": payload,
            "video_part_count": video_part_count,
        }

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        candidate = response_json.get("candidates", [{}])[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason == "SAFETY":
            raise RuntimeError("Response blocked by safety filters.")

        # 获取最后一个 type=text 的 part
        parts = candidate.get("content", {}).get("parts", [])
        content = ""
        for part in reversed(parts):
            if part.get("text") is not None:
                content = part.get("text", "").strip()
                break

        usage_json = response_json.get("usageMetadata", {})
        usage = {
            "input_tokens": usage_json.get("promptTokenCount", 0),
            "output_tokens": usage_json.get("candidatesTokenCount", 0),
            "total_tokens": usage_json.get("totalTokenCount", 0),
        }
        return {
            "content": content,
            "usage": usage,
            "raw_response": response_json,
        }
