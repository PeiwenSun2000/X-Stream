from ..model_hub import ModelClient, register_adapter
from typing import List, Dict, Any, Optional

@register_adapter("seedream")
class SeedreamAdapter(ModelClient):
    """Seedream adapter for image generation APIs"""

    def format_messages(self, context: List[Dict[str, Any]]) -> str:
        """Extract the prompt from messages (image generation prompt)

        Extract text from the last user message as the prompt
        """
        # Search backward for the last user message
        for msg in reversed(context):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # If it is a string, return it directly
                if isinstance(content, str):
                    return content.strip()
                # If it is a list, extract all text content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                    if text_parts:
                        return " ".join(text_parts)

        # If no user message is found, return an empty string
        return ""

    def build_payload(self, prompt: str, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build the image generation request payload

        Parameters:
        - prompt: Image generation prompt
        - request_params: Request parameters (may include size, sequential_image_generation, stream, response_format, watermark, etc.)
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt
        }

        # Extract image generation parameters from request_params
        # Keep default values or read them from request_params
        payload["size"] = request_params.get("size", "1920x1080")
        payload["sequential_image_generation"] = request_params.get("sequential_image_generation", "disabled")
        payload["stream"] = request_params.get("stream", False)
        payload["response_format"] = request_params.get("response_format", "url")
        payload["watermark"] = request_params.get("watermark", False)

        # Add other possible parameters
        for k, v in request_params.items():
            if k not in ("model", "prompt", "size", "sequential_image_generation", "stream", "response_format", "watermark"):
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
        """Parse the image generation response

        Extract the image URL or image data from the response
        """
        # Parse the response according to response_format
        # If response_format is "url", the response may contain a "data" field with the image URL
        # If response_format is "b64_json", the response may contain a base64-encoded image

        # Try multiple possible response formats
        image_url = None

        # Format 1: {"data": [{"url": "..."}]}
        if "data" in response_json and isinstance(response_json["data"], list):
            if len(response_json["data"]) > 0:
                first_item = response_json["data"][0]
                image_url = first_item.get("url") or first_item.get("b64_json")

        # Format 2: {"url": "..."}
        elif "url" in response_json:
            image_url = response_json["url"]

        # Format 3: {"image_url": "..."}
        elif "image_url" in response_json:
            image_url = response_json["image_url"]

        # Format 4: {"b64_json": "..."}
        elif "b64_json" in response_json:
            image_url = f"data:image/png;base64,{response_json['b64_json']}"

        # If an image URL is found, return the standard format
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

        # If no image URL is found, return an error message
        return {
            "content": f"Unable to parse response: {response_json}",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "error": "Unable to extract image URL from response",
            "raw_response": response_json,
        }
