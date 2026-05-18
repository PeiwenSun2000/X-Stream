from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any

@register_adapter("openai")
class OpenAIAdapter(ModelClient):
    def format_messages(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # OpenAI adapter 不支持图片和视频，需要转换为文本
        messages = []
        for msg in context:
            role = msg["role"]
            content = msg["content"]

            # 如果是字符串，直接使用
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            else:
                # 如果是列表，处理每个 item
                text_parts = []
                for item in content:
                    typ = item.get("type")
                    if typ == "text":
                        text = item.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif typ == "image":
                        # 将图片转换为文本描述
                        img_path = item.get("image", "")
                        text_parts.append(f"[IMAGE: {img_path}]")
                    elif typ == "video":
                        # 将视频转换为文本描述
                        video_path = item.get("video", "")
                        text_parts.append(f"[VIDEO: {video_path}]")

                # 合并所有文本部分
                messages.append({"role": role, "content": " ".join(text_parts)})

        return messages

    def build_payload(self, messages, request_params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
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
