from ..model_hub import ModelClient, register_adapter
from ..utils import to_base64
from typing import List, Dict, Any

@register_adapter("celie")
class Celie(ModelClient):
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

            if new_content and role == "user":
                # 多个连续 video 时拆成多对 user / assistant <|silent|>，最后一组 <v>+<text> 后不插 assistant
                segments = []
                for item in new_content:
                    if item.get("type") == "video_url":
                        segments.append([item])
                    else:
                        if segments:
                            segments[-1].append(item)
                        else:
                            segments.append([item])
                for i, seg in enumerate(segments):
                    messages.append({"role": "user", "content": seg})
                    if i < len(segments) - 1:
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": "<|silent|>"}]})

            elif new_content:
                messages.append({"role": role, "content": new_content})

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            })

        ## remove pairs like: [user: "", assistant: "silent"]
        n = len(messages)
        is_silent = lambda x: len(x)==1 and x[0]["type"]=="text" and x[0]["text"] in ["", "<|silent|>"]
        delete_ids = set()
        for i in range(1, n):
            cur = messages[i-1]
            nxt = messages[i]
            if cur["role"]=="user" and nxt["role"]=="assistant" and is_silent(cur["content"]) and is_silent(nxt["content"]):
                delete_ids.add(i-1)
                delete_ids.add(i)
        messages = [msg for i, msg in enumerate(messages) if i not in delete_ids]

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
