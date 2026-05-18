from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any, Optional

@register_adapter("seedream")
class SeedreamAdapter(ModelClient):
    """Seedream 适配器：用于图像生成 API"""

    def format_messages(self, context: List[Dict[str, Any]]) -> str:
        """从消息中提取 prompt（图像生成提示词）

        提取最后一个用户消息的文本内容作为 prompt
        """
        # 从后往前查找最后一个用户消息
        for msg in reversed(context):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # 如果是字符串，直接返回
                if isinstance(content, str):
                    return content.strip()
                # 如果是列表，提取所有文本内容
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                    if text_parts:
                        return " ".join(text_parts)

        # 如果没有找到用户消息，返回空字符串
        return ""

    def build_payload(self, prompt: str, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """构建图像生成请求负载

        参数：
        - prompt: 图像生成提示词
        - request_params: 请求参数（可能包含 size, sequential_image_generation, stream, response_format, watermark 等）
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt
        }

        # 从 request_params 中提取图像生成相关参数
        # 保留默认值或从 request_params 中获取
        payload["size"] = request_params.get("size", "1920x1080")
        payload["sequential_image_generation"] = request_params.get("sequential_image_generation", "disabled")
        payload["stream"] = request_params.get("stream", False)
        payload["response_format"] = request_params.get("response_format", "url")
        payload["watermark"] = request_params.get("watermark", False)

        # 添加其他可能的参数
        for k, v in request_params.items():
            if k not in ("model", "prompt", "size", "sequential_image_generation", "stream", "response_format", "watermark"):
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
        """解析图像生成响应

        从响应中提取图像 URL 或图像数据
        """
        # 根据 response_format 解析响应
        # 如果 response_format 是 "url"，响应可能包含 "data" 字段，其中包含图像 URL
        # 如果 response_format 是 "b64_json"，响应可能包含 base64 编码的图像

        # 尝试多种可能的响应格式
        image_url = None

        # 格式1: {"data": [{"url": "..."}]}
        if "data" in response_json and isinstance(response_json["data"], list):
            if len(response_json["data"]) > 0:
                first_item = response_json["data"][0]
                image_url = first_item.get("url") or first_item.get("b64_json")

        # 格式2: {"url": "..."}
        elif "url" in response_json:
            image_url = response_json["url"]

        # 格式3: {"image_url": "..."}
        elif "image_url" in response_json:
            image_url = response_json["image_url"]

        # 格式4: {"b64_json": "..."}
        elif "b64_json" in response_json:
            image_url = f"data:image/png;base64,{response_json['b64_json']}"

        # 如果找到了图像 URL，返回标准格式
        if image_url:
            return {
                "content": image_url,
                "usage": {
                    "input_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": response_json.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response_json.get("usage", {}).get("total_tokens", 0),
                },
                "raw_response": response_json,
            }

        # 如果没有找到图像 URL，返回错误信息
        return {
            "content": f"无法解析响应: {response_json}",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "error": "无法从响应中提取图像 URL",
            "raw_response": response_json,
        }
