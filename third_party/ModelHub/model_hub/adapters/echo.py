from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any, Optional

@register_adapter("echo")
class EchoAdapter(ModelClient):
    """Echo adapter"""

    def call(self, messages: List[Dict[str, Any]], request_params: Optional[Dict[str, Any]] = None, request_id = None) -> Dict[str, Any]:
        """Override the call method and read content from the last text item in messages"""
        request_params = {**self.default_request_params, **(request_params or {})}

        # Search backward for the last message containing text
        content = ""
        for msg in reversed(messages):
            msg_content = msg.get("content", "")

            # If it is a string, use it directly
            if isinstance(msg_content, str):
                content = msg_content.strip()
                if content:
                    break
            # If it is a list, look for text items
            elif isinstance(msg_content, list):
                for item in reversed(msg_content):
                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            content = text
                            break
                if content:
                    break

        # Return a standard-format response
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
        """No implementation needed because no request is sent"""
        pass

    def build_payload(self, messages: Any, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """No implementation needed because no request is sent"""
        pass

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """No implementation needed because no request is sent"""
        pass
