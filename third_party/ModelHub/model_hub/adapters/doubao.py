from ..model_hub import ModelClient, register_adapter
from ..utils import to_base64
from typing import List, Dict, Any

@register_adapter("doubao")
class DoubaoAdapter(ModelClient):
    def format_messages(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = []
        for msg in context:
            role = msg["role"]
            content = msg["content"]

            items = [{"type": "text", "text": content}] if isinstance(content, str) else content
            new_content = []

            for item in items:
                typ = item.get("type")
                if typ == "video":
                    video_path = item["video"]
                    b64 = to_base64(video_path, self.max_video_size_bytes)
                    new_content.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}", "fps": 1}
                    })
                elif typ == "image":
                    img_path = item["image"]
                    b64 = to_base64(img_path, self.max_video_size_bytes)
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                elif typ == "text":
                    text = item.get("text", "").strip()
                    new_content.append({"type": "text", "text": text})

            if new_content:
                messages.append({"role": role, "content": new_content})

        # Inject default system message if missing
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            })
        return messages

    def build_payload(self, messages, request_params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = dict(request_params or {})
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        raw_mm_kwargs = request_params.pop("mm_processor_kwargs", {}) or {}
        mm_kwargs = dict(raw_mm_kwargs) if isinstance(raw_mm_kwargs, dict) else {}
        # X-Stream patch-level pruner hand-off: when the upstream MLLMFlow runs
        # ``cdpruner_token`` / ``surge_token`` it embeds an ``_xstream_pruner``
        # struct in ``request_params``. The plugin running inside the local
        # vLLM worker reads ``mm_processor_kwargs.xstream_instruction``, so we
        # forward it here and consume the marker before generic param copy.
        xstream_info = request_params.pop("_xstream_pruner", None)
        if isinstance(xstream_info, dict):
            instruction = (xstream_info.get("instruction") or "").strip()
            if instruction:
                mm_kwargs["xstream_instruction"] = instruction
        if mm_kwargs:
            payload["mm_processor_kwargs"] = mm_kwargs
        # Add all request_params directly to the payload (after stripping the
        # internal marker above to keep the wire payload clean).
        for k, v in request_params.items():
            if k not in ("model", "messages") and not str(k).startswith("_"):
                payload[k] = v

        # Build request headers using Authorization Bearer
        headers = {}
        if self.api_key and "{api_key}" not in self.endpoint:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return {
            "headers": headers,
            "payload": payload
        }

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        choice = response_json.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        usage_json = response_json.get("usage", {})
        usage = {
            "input_tokens": usage_json.get("prompt_tokens", 0),
            "output_tokens": usage_json.get("completion_tokens", 0),
            "total_tokens": usage_json.get("total_tokens", 0),
        }
        return {
            "content": content,
            "usage": usage,
            "raw_response": response_json,
        }
