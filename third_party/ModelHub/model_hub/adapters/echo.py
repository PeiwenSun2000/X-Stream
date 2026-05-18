from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any, Optional

@register_adapter("echo")
class EchoAdapter(ModelClient):
    """Echo 适配器"""

    def call(self, messages: List[Dict[str, Any]], request_params: Optional[Dict[str, Any]] = None, request_id = None) -> Dict[str, Any]:
        """重写 call 方法，从 messages 的最后一个 text 类型读取内容"""
        request_params = {**self.default_request_params, **(request_params or {})}

        # 从后往前查找最后一个包含 text 的消息
        content = ""
        for msg in reversed(messages):
            msg_content = msg.get("content", "")

            # 如果是字符串，直接使用
            if isinstance(msg_content, str):
                content = msg_content.strip()
                if content:
                    break
            # 如果是列表，查找 text 类型
            elif isinstance(msg_content, list):
                for item in reversed(msg_content):
                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            content = text
                            break
                if content:
                    break

        # 返回标准格式的响应
        return {
            "content": content,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            "raw_response": content
        }

    def format_messages(self, context: List[Dict[str, Any]]) -> Any:
        """不需要实现，因为不会发送请求"""
        pass

    def build_payload(self, messages: Any, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """不需要实现，因为不会发送请求"""
        pass

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """不需要实现，因为不会发送请求"""
        pass
