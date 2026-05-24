from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any

@register_adapter("openai")
class OpenAIAdapter(ModelClient):
    def format_messages(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # The OpenAI adapter does not support images and videos, so they must be converted to text
        messages = []
        for msg in context:
            role = msg["role"]
            content = msg["content"]

            # If it is a string, use it directly
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            else:
                # If it is a list, process each item
                text_parts = []
                for item in content:
                    typ = item.get("type")
                    if typ == "text":
                        text = item.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif typ == "image":
                        # Convert the image to a text description
                        img_path = item.get("image", "")
                        text_parts.append(f"[IMAGE: {img_path}]")
                    elif typ == "video":
                        # Convert the video to a text description
                        video_path = item.get("video", "")
                        text_parts.append(f"[VIDEO: {video_path}]")

                # Merge all text parts
                messages.append({"role": role, "content": " ".join(text_parts)})

        return messages

    def build_payload(self, messages, request_params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        # Add all request_params directly to the payload
        for k, v in request_params.items():
            if k not in ("model", "messages"):
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
