from textwrap import indent
from ..model_hub import ModelClient, register_adapter
from ..utils import to_base64
from typing import List, Dict, Any

# Gemini supports at most 10 videos per request; uniformly sample 10 if there are more
GEMINI_MAX_VIDEOS = 10


def _is_video_part(part: Dict[str, Any]) -> bool:
    if part.get("video_metadata") is not None:
        return True
    inline = part.get("inline_data") or {}
    return (inline.get("mime_type") or "").startswith("video/")


def _uniform_sample_video_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If video parts exceed GEMINI_MAX_VIDEOS, uniformly keep GEMINI_MAX_VIDEOS of them and leave non-video parts unchanged."""
    video_indices = [i for i, p in enumerate(parts) if _is_video_part(p)]
    n = len(video_indices)
    if n <= GEMINI_MAX_VIDEOS:
        return parts
    # Uniformly select GEMINI_MAX_VIDEOS indices from [0, n-1] (uniform sampling, not the first 10)
    if GEMINI_MAX_VIDEOS <= 1:
        selected_positions = [0] if GEMINI_MAX_VIDEOS else []
    else:
        selected_positions = [round(i * (n - 1) / (GEMINI_MAX_VIDEOS - 1)) for i in range(GEMINI_MAX_VIDEOS)]
    keep_index_set = {video_indices[p] for p in selected_positions}
    return [p for i, p in enumerate(parts) if not _is_video_part(p) or i in keep_index_set]


def _drop_first_n_video_parts(contents: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Drop the first n video parts from the beginning of the full request in order, keeping the later content; used for frame-by-frame trimming retries when the body is too large."""
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
    """Ensure the total number of video parts in the full request does not exceed max_videos; uniformly sample max_videos across the full request when it does."""
    # Collect all (content_idx, part_idx) pairs where the part is a video
    video_locations = []
    for ci, content in enumerate(contents):
        for pi, part in enumerate(content.get("parts", [])):
            if _is_video_part(part):
                video_locations.append((ci, pi))
    n = len(video_locations)
    if n <= max_videos:
        return contents
    # Uniformly select max_videos items
    if max_videos <= 1:
        keep_locations = set(video_locations[:1]) if max_videos else set()
    else:
        selected_positions = [round(i * (n - 1) / (max_videos - 1)) for i in range(max_videos)]
        keep_locations = {video_locations[p] for p in selected_positions}
    # For each content item, keep only selected video parts and all non-video parts; skip any content item that becomes empty to avoid sending the full original content and still exceeding 10 videos
    result = []
    for ci, content in enumerate(contents):
        new_parts = [
            part for pi, part in enumerate(content.get("parts", []))
            if not _is_video_part(part) or (ci, pi) in keep_locations
        ]
        if new_parts:
            result.append({**content, "parts": new_parts})
        # else: Do not append the original content; otherwise it would bring back all videos and the total video count could still exceed 10
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
                    # Gemini requires the part data oneof to contain an initialized field; an empty string is treated as uninitialized and causes an error
                    if text:
                        parts.append({"text": text})

            parts = _uniform_sample_video_parts(parts)
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return {"contents": contents, "system_instruction": system_instruction}

    def build_payload(self, messages, request_params: Dict[str, Any]) -> Dict[str, Any]:
        # messages here is the formatted result returned by format_messages
        formatted = messages if isinstance(messages, dict) else {"contents": messages, "system_instruction": None}

        # Validate that contents is not empty
        if not formatted.get("contents"):
            raise ValueError("Contents cannot be empty")

        # When the body is too large, drop the first N videos from the beginning, then limit the total count
        drop_n = request_params.get("_drop_first_n_videos", 0)
        contents = _drop_first_n_video_parts(formatted["contents"], drop_n)
        if not contents:
            raise ValueError("Contents empty after dropping video parts")
        contents = _limit_payload_videos_to_max(contents, GEMINI_MAX_VIDEOS)
        video_part_count = sum(1 for c in contents for p in c.get("parts", []) if _is_video_part(p))

        payload = {"contents": contents}
        if formatted.get("system_instruction"):
            payload["system_instruction"] = {"parts": [{"text": formatted["system_instruction"]}]}

        # Add request_params directly to the payload (skip internal keys such as _drop_first_n_videos)
        for k, v in request_params.items():
            if k in ("contents", "system_instruction") or (k.startswith("_")):
                continue
            payload[k] = v

        # Build request headers: Gemini uses x-goog-api-key
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

        # Get the last part with type=text
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
