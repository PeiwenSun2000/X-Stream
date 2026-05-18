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
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        # 直接添加所有 request_params 到 payload
        for k, v in request_params.items():
            if k not in ("model", "messages"):
                payload[k] = v

        # 构建请求头：使用 Authorization Bearer
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
