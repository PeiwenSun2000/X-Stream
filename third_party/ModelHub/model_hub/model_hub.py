import json
import os
import random
import time
from typing import Dict, List, Any, Optional, Tuple
import requests
from .utils import load_json, write_log

# Adapter registry
_ADAPTERS: Dict[str, type] = {}


def register_adapter(name: str):
    """Adapter registration decorator"""
    def decorator(cls):
        _ADAPTERS[name] = cls
        return cls
    return decorator


class ModelClient:
    """Base model client class"""

    def __init__(self, config: Dict[str, Any]):
        # System configuration fields
        self.model_name = os.path.expandvars(str(config.get("model_name", "")))
        self.endpoint = os.path.expandvars(str(config.get("endpoint", "")))
        self.api_key = os.path.expandvars(str(config.get("api_key", "")))
        # Whether this is a local vLLM deployment such as Qwen.
        # For these models, 400/429 usually indicates a request-content issue (for example, an unparseable video stream), so long unlimited retries are inappropriate.
        # Controlled by the is_vllm_local field in models.json; defaults to False.
        self.is_vllm_local = bool(config.get("is_vllm_local", False))
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 600)
        # For 400/429 responses, keep retrying until success or the limit is reached (backoff retry)
        self.retry_4xx_429 = config.get("retry_4xx_429", True)
        self.max_retries_4xx_429 = config.get("max_retries_4xx_429", 0)  # 0 means unlimited attempts
        self.max_retry_seconds_4xx_429 = config.get("max_retry_seconds_4xx_429", 3600)  # Maximum total retry duration (seconds)
        self.retry_backoff_initial = config.get("retry_backoff_initial", 5)
        self.retry_backoff_max = config.get("retry_backoff_max", 300)

        # Video size limit (MB to bytes)
        max_video_size_mb = config.get("max_video_size_mb", 100)
        self.max_video_size_bytes = max_video_size_mb * 1024 * 1024

        # Default request parameters (read from the request_params field)
        self.default_request_params = config.get("request_params", {})

    def format_messages(self, context: List[Dict[str, Any]]) -> Any:
        """Format messages (implemented by subclasses)"""
        pass

    def build_payload(self, messages: Any, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build the request payload and headers (implemented by subclasses)

        Return format:
        {
            "headers": Dict[str, str],  # Headers dictionary
            "payload": Dict[str, Any]   # Request payload
        }
        """
        pass

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response (implemented by subclasses)"""
        pass

    def call(self, messages: List[Dict[str, Any]], request_params: Optional[Dict[str, Any]] = None, request_id = None) -> Dict[str, Any]:
        """Call the API"""
        request_params = {**self.default_request_params, **(request_params or {})}

        # Build the URL once (independent of payload)
        url = (
            self.endpoint
            .replace("{model_name}", self.model_name)
            .replace("{api_key}", self.api_key)
        )

        drop_first_n = 0
        while True:
            # When the body is too large, use _drop_first_n_videos to drop videos from the beginning and rebuild the payload
            request_params_cur = {**request_params, "_drop_first_n_videos": drop_first_n}
            formatted_messages = self.format_messages(messages)
            result = self.build_payload(formatted_messages, request_params_cur)
            headers = result.get("headers", {})
            payload = result.get("payload", {})
            video_part_count = result.get("video_part_count", -1)

            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"

            log_data = {
                "payload": payload,
                "attempts_history": [],
                "response": None,
                "request_details": {
                    "model_name": self.model_name,
                    "endpoint": url,
                    "endpoint_template": self.endpoint,
                    "request_id": request_id,
                    "max_retries": self.max_retries,
                    "timeout": self.timeout
                }
            }

            body_too_large_retry = False
            # Send the request with retries. For 400/429, use separate backoff retries until success or the limit is reached.
            for attempt in range(self.max_retries):
                retry_4xx_count = 0
                start_time_4xx = time.time()
                while True:
                    try:
                        start_time = time.time()
                        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                        latency = time.time() - start_time
                        response.raise_for_status()

                        response_json = load_json(response.text)
                        log_data["response"] = response_json
                        log_data["latency_sec"] = latency
                        write_log(log_data, request_id)

                        return self.parse_response(response_json)

                    except requests.exceptions.HTTPError as e:
                        status_code = getattr(e.response, "status_code", None)
                        error_msg = e.response.text[:500] if hasattr(e.response, "text") else str(e)
                        error_str = f"{e.response.status_code}: {error_msg}"
                        error_type = "HTTPError"
                        is_body_too_large = "body too large" in (e.response.text or "").lower() or "request body too large" in (e.response.text or "").lower()
                        # request body too large: drop one video from the beginning, rebuild the payload, and retry until success or no videos remain
                        if status_code in (400, 429) and is_body_too_large and self.retry_4xx_429:
                            if video_part_count is not None and video_part_count >= 0:
                                if video_part_count == 0:
                                    log_data["attempts_history"].append({
                                        "attempt": "body_too_large_no_videos_left",
                                        "error": error_str,
                                        "error_type": error_type,
                                    })
                                    write_log(log_data, request_id)
                                    return {
                                        "content": f"@BodyTooLarge: {error_type} {error_str}",
                                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                                        "raw_response": {"detail": error_str},
                                        "error": error_str,
                                    }
                                drop_first_n += 1
                                print(f"[ModelHub] request body too large，retrying after dropping one video from the beginning (drop_first_n={drop_first_n}) ...")
                                log_data["attempts_history"].append({
                                    "attempt": f"body_too_large_drop_{drop_first_n}",
                                    "error": error_str,
                                    "error_type": error_type,
                                })
                                write_log(log_data, request_id)
                                body_too_large_retry = True
                                break
                        # For 400/429 responses, keep using backoff retries until success or the limit is reached
                        if status_code in (400, 429) and self.retry_4xx_429 and not body_too_large_retry:
                            # Local vLLM models such as Qwen: 4xx usually means the current request content has an issue (such as a video stream that cannot be opened),
                            # so it should not use long unlimited backoff retries; instead, use the normal outer max_retries logic (skip after 3 attempts by default).
                            if self.is_vllm_local:
                                break
                            retry_4xx_count += 1
                            elapsed_4xx = time.time() - start_time_4xx
                            if self.max_retries_4xx_429 and retry_4xx_count >= self.max_retries_4xx_429:
                                log_data["attempts_history"].append({
                                    "attempt": f"4xx_retry_{retry_4xx_count}",
                                    "error": error_str,
                                    "error_type": error_type,
                                })
                                write_log(log_data, request_id)
                                return {
                                    "content": f"@4xx_retry_limit: {error_type} {error_str}",
                                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                                    "raw_response": {"detail": error_str},
                                    "error": error_str,
                                }
                            if elapsed_4xx >= self.max_retry_seconds_4xx_429:
                                log_data["attempts_history"].append({
                                    "attempt": f"4xx_timeout_{elapsed_4xx:.0f}s",
                                    "error": error_str,
                                    "error_type": error_type,
                                })
                                write_log(log_data, request_id)
                                return {
                                    "content": f"@4xx_retry_timeout: {error_type} {error_str}",
                                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                                    "raw_response": {"detail": error_str},
                                    "error": error_str,
                                }
                            print(
                                f"[ModelHub] 400/429 retry attempt {retry_4xx_count}  (retrying immediately): {error_msg[:150]}..."
                            )
                            log_data["attempts_history"].append({
                                "attempt": f"4xx_retry_{retry_4xx_count}",
                                "error": error_str,
                                "error_type": error_type,
                            })
                            write_log(log_data, request_id)
                            continue
                        # For non-400/429 responses or when retry_4xx_429 is disabled, use normal retry handling
                        break
                    except requests.exceptions.Timeout:
                        error_str = f"Timeout: {self.timeout}s"
                        error_type = "Timeout"
                        break
                    except Exception as e:
                        error_str = str(e)
                        error_type = type(e).__name__
                        break

                if body_too_large_retry:
                    break
                # Non-400/429 failure: record it and possibly proceed to the next attempt
                log_data["attempts_history"].append({
                    "attempt": attempt + 1,
                    "error": error_str,
                    "error_type": error_type,
                })
                write_log(log_data, request_id)

                if attempt == self.max_retries - 1:
                    return {
                        "content": f"@Attempt-{attempt+1}: {error_type} {error_str}",
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "raw_response": {"detail": error_str},
                        "error": error_str,
                    }

            if body_too_large_retry:
                continue
            break


class ModelHub:
    """Model hub that manages multiple model configurations"""

    def __init__(self, config_path: str):
        """Load models.json configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.models_config = json.load(f)

    def _check_health(self, health_check_endpoint: str) -> bool:
        """Check health status

        Args:
            health_check_endpoint: Health-check endpoint URL

        Returns:
            bool: Return True if healthy; otherwise return False
        """
        try:
            response = requests.get(health_check_endpoint, timeout=10)
            # Check whether the HTTP status code is 200
            if response.status_code != 200:
                return False

            # Try parsing JSON
            try:
                health_data = load_json(response.text)
                # Check whether the status field is healthy
                if isinstance(health_data, dict) and health_data.get("status") == "healthy":
                    return True
            except:
                # If it is not JSON or parsing fails but the status code is 200, still treat it as healthy
                pass

            # If the status code is 200, treat it as healthy
            return True
        except Exception:
            # Treat any exception as unhealthy
            return False

    def _is_config_healthy(self, config: Dict[str, Any]) -> bool:
        """Check whether a configuration is healthy

        Args:
            config: Configuration dictionary

        Returns:
            bool: Return True if healthy; otherwise return False
        """
        health_check_endpoint = config.get("health_check_endpoint")
        if health_check_endpoint:
            return self._check_health(health_check_endpoint)
        # If there is no health_check_endpoint, treat it as healthy (no check needed)
        return True

    def _select_config(self, model_name: str, excluded_indices: Optional[List[int]] = None) -> Tuple[Dict[str, Any], int]:
        """Select a configuration by weight and check health status

        Logic:
        1. First select all healthy configs (excluding indices already tried)
        2. If the healthy list is empty, wait and keep checking until a non-empty healthy list appears or timeout is reached (600 seconds)
        3. If the healthy list is non-empty, select one healthy config by weight (probability)

        Args:
            model_name: Model name
            excluded_indices: List of configuration indices to exclude

        Returns:
            tuple[Dict[str, Any], int]: Return a tuple of (configuration dictionary, configuration index)
        """
        excluded_indices = excluded_indices or []
        configs = self.models_config[model_name]
        max_wait_time = 6000  # Wait at most 6000 seconds
        check_interval = 5  # Check every 5 seconds
        start_time = time.time()
        total_configs = len(configs)

        while time.time() - start_time < max_wait_time:
            # Filter all healthy configurations and their indices (excluding those already tried)
            healthy_configs_with_index = [(c, i) for i, c in enumerate(configs)
                                         if i not in excluded_indices and self._is_config_healthy(c)]
            healthy_count = len(healthy_configs_with_index)
            elapsed_time = int(time.time() - start_time)

            # If the healthy list is non-empty, select by weight
            if healthy_configs_with_index:
                print(f"[ModelHub] Model {model_name}: found {healthy_count}/{total_configs} healthy configurations, wait time {elapsed_time}s")
                healthy_configs = [c for c, _ in healthy_configs_with_index]
                weights = [c.get("weight", 1.0) for c in healthy_configs]
                selected_config = random.choices(healthy_configs_with_index, weights=weights, k=1)[0]
                return selected_config  # Return (config, index)

            # If the healthy list is empty, show a waiting message
            print(f"[ModelHub] Model {model_name}: currently has 0/{total_configs}，healthy configurations, waiting... (waited  {elapsed_time}s/{max_wait_time}s)")

            # Wait and retry
            time.sleep(check_interval)

        # If no healthy configuration is found before the maximum wait time, return the first non-excluded configuration
        available_configs = [(c, i) for i, c in enumerate(configs) if i not in excluded_indices]
        if available_configs:
            print(f"[ModelHub] Warning: model {model_name} within {max_wait_time}s did not find a healthy configuration; returning the first available configuration")
            return available_configs[0]
        # If all configurations have been excluded, return the first configuration and let the caller handle the error
        print(f"[ModelHub] Warning: model {model_name} all configurations have been tried; returning the first configuration")
        return (configs[0], 0)

    def call(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        request_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call the model API and automatically retry other configurations on failure"""
        tried_indices = []
        configs = self.models_config[model_name]

        assert len(configs) > 0

        while len(tried_indices) < len(configs):
            selected_config, config_index = self._select_config(model_name, tried_indices)
            adapter_class = _ADAPTERS[selected_config["adapter"]]
            client = adapter_class(selected_config)

            current_request_id = f"{request_id}_cfg{config_index}" if request_id else None
            result = client.call(messages, request_params or {}, request_id=current_request_id)

            # If the result does not contain an error field, return the successful result
            if "error" not in result:
                return result

            # If it contains error, record the tried index and continue trying other configurations
            tried_indices.append(config_index)
            # NOTE: Previously the full error was printed (including video/image numpy arrays returned by vLLM),
            # which produced excessive output. Truncate it to the first 300 characters here; comment out the two lines below for complete silence.
            _err = str(result.get("error"))
            if len(_err) > 300:
                _err = _err[:300] + f"... <truncated, total {len(_err)} chars>"
            print(f"[ModelHub] Configuration {config_index} request failed, trying another configuration: {_err}")
            # print(f"[ModelHub] Configuration {config_index} request failed, trying another configuration: {result.get('error')}")

        # All configurations have been tried; return the last error result
        return result


if __name__ == "__main__":
    hub = ModelHub("models.json")
    response = hub.call(
        model_name="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        request_params={"temperature": 0.7},
        request_id="req-123"
    )
    print(response)
