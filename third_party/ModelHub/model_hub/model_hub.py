import json
import random
import time
from typing import Dict, List, Any, Optional, Tuple
import requests
from .utils import load_json, write_log

# 适配器注册表
_ADAPTERS: Dict[str, type] = {}


def register_adapter(name: str):
    """注册适配器装饰器"""
    def decorator(cls):
        _ADAPTERS[name] = cls
        return cls
    return decorator


class ModelClient:
    """模型客户端基类"""

    def __init__(self, config: Dict[str, Any]):
        # 系统配置字段
        self.model_name = config.get("model_name", "")
        self.endpoint = config.get("endpoint", "")
        self.api_key = config.get("api_key", "")
        # 是否为本地 vLLM 部署的 Qwen 等模型。
        # 这类模型的 400/429 多为请求内容本身问题（例如视频流无法解析），不适合长时间无限重试。
        # 由 models.json 中的 is_vllm_local 字段控制，默认 False。
        self.is_vllm_local = bool(config.get("is_vllm_local", False))
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 600)
        # 400/429 时持续重试直到成功或达到上限（退避重试）
        self.retry_4xx_429 = config.get("retry_4xx_429", True)
        self.max_retries_4xx_429 = config.get("max_retries_4xx_429", 0)  # 0 表示不限制次数
        self.max_retry_seconds_4xx_429 = config.get("max_retry_seconds_4xx_429", 3600)  # 最多重试总时长（秒）
        self.retry_backoff_initial = config.get("retry_backoff_initial", 5)
        self.retry_backoff_max = config.get("retry_backoff_max", 300)

        # 视频大小限制（MB转字节）
        max_video_size_mb = config.get("max_video_size_mb", 100)
        self.max_video_size_bytes = max_video_size_mb * 1024 * 1024

        # 默认请求参数（从 request_params 字段读取）
        self.default_request_params = config.get("request_params", {})

    def format_messages(self, context: List[Dict[str, Any]]) -> Any:
        """格式化消息（由子类实现）"""
        pass

    def build_payload(self, messages: Any, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """构建请求负载和请求头（由子类实现）

        返回格式:
        {
            "headers": Dict[str, str],  # 请求头字典
            "payload": Dict[str, Any]   # 请求负载
        }
        """
        pass

    def parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """解析响应（由子类实现）"""
        pass

    def call(self, messages: List[Dict[str, Any]], request_params: Optional[Dict[str, Any]] = None, request_id = None) -> Dict[str, Any]:
        """调用 API"""
        request_params = {**self.default_request_params, **(request_params or {})}

        # 构建 URL（与 payload 无关，只建一次）
        url = self.endpoint.format(
            model_name=self.model_name,
            api_key=self.api_key
        )

        drop_first_n = 0
        while True:
            # body too large 时用 _drop_first_n_videos 从开头丢弃视频后重建 payload
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
            # 发送请求（带重试）。400/429 时单独做退避重试直到成功或达到上限。
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
                        # request body too large：从开头丢弃 1 个视频后重建 payload 再试，直到成功或已无视频可减
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
                                print(f"[ModelHub] request body too large，丢弃开头 1 个视频后重试 (drop_first_n={drop_first_n}) ...")
                                log_data["attempts_history"].append({
                                    "attempt": f"body_too_large_drop_{drop_first_n}",
                                    "error": error_str,
                                    "error_type": error_type,
                                })
                                write_log(log_data, request_id)
                                body_too_large_retry = True
                                break
                        # 400/429 时持续退避重试直到成功或达到上限
                        if status_code in (400, 429) and self.retry_4xx_429 and not body_too_large_retry:
                            # 本地 vLLM Qwen 等模型：4xx 一般表示当前请求内容有问题（如视频流无法打开），
                            # 不应长时间无限退避重试，而是按普通错误走外层 max_retries 逻辑（默认 3 次后跳过）。
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
                                f"[ModelHub] 400/429 重试第 {retry_4xx_count} 次（立即重试）: {error_msg[:150]}..."
                            )
                            log_data["attempts_history"].append({
                                "attempt": f"4xx_retry_{retry_4xx_count}",
                                "error": error_str,
                                "error_type": error_type,
                            })
                            write_log(log_data, request_id)
                            continue
                        # 非 400/429 或未开启 retry_4xx_429，按普通重试处理
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
                # 非 400/429 的失败：记录并可能进行下一轮 attempt
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
    """模型中心，管理多个模型配置"""

    def __init__(self, config_path: str):
        """加载 models.json 配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.models_config = json.load(f)

    def _check_health(self, health_check_endpoint: str) -> bool:
        """检查健康状态

        Args:
            health_check_endpoint: 健康检查端点 URL

        Returns:
            bool: 如果健康返回 True，否则返回 False
        """
        try:
            response = requests.get(health_check_endpoint, timeout=10)
            # 检查 HTTP 状态码是否为 200
            if response.status_code != 200:
                return False

            # 尝试解析 JSON
            try:
                health_data = load_json(response.text)
                # 检查 status 字段是否为 healthy
                if isinstance(health_data, dict) and health_data.get("status") == "healthy":
                    return True
            except:
                # 如果不是 JSON 或解析失败，但状态码是 200，也认为健康
                pass

            # 如果状态码是 200，认为健康
            return True
        except Exception:
            # 任何异常都认为不健康
            return False

    def _is_config_healthy(self, config: Dict[str, Any]) -> bool:
        """检查配置是否健康

        Args:
            config: 配置字典

        Returns:
            bool: 如果健康返回 True，否则返回 False
        """
        health_check_endpoint = config.get("health_check_endpoint")
        if health_check_endpoint:
            return self._check_health(health_check_endpoint)
        # 如果没有 health_check_endpoint，认为健康（不需要检查）
        return True

    def _select_config(self, model_name: str, excluded_indices: Optional[List[int]] = None) -> Tuple[Dict[str, Any], int]:
        """根据权重选择配置，并检查健康状态

        逻辑：
        1. 首先选出所有健康的 config（排除已尝试的索引）
        2. 如果健康列表为空，等待并循环检查，直到出现非空的健康列表，或者超时（600秒）
        3. 如果健康列表非空，根据权重（概率）从健康的 config 中选择一个

        Args:
            model_name: 模型名称
            excluded_indices: 要排除的配置索引列表

        Returns:
            tuple[Dict[str, Any], int]: 返回 (配置字典, 配置索引) 的元组
        """
        excluded_indices = excluded_indices or []
        configs = self.models_config[model_name]
        max_wait_time = 6000  # 最多等待 6000 秒
        check_interval = 5  # 每 5 秒检查一次
        start_time = time.time()
        total_configs = len(configs)

        while time.time() - start_time < max_wait_time:
            # 筛选出所有健康的配置及其索引（排除已尝试的）
            healthy_configs_with_index = [(c, i) for i, c in enumerate(configs)
                                         if i not in excluded_indices and self._is_config_healthy(c)]
            healthy_count = len(healthy_configs_with_index)
            elapsed_time = int(time.time() - start_time)

            # 如果健康列表非空，根据权重选择
            if healthy_configs_with_index:
                print(f"[ModelHub] 模型 {model_name}: 找到 {healthy_count}/{total_configs} 个健康配置，等待时间 {elapsed_time}s")
                healthy_configs = [c for c, _ in healthy_configs_with_index]
                weights = [c.get("weight", 1.0) for c in healthy_configs]
                selected_config = random.choices(healthy_configs_with_index, weights=weights, k=1)[0]
                return selected_config  # 返回 (config, index)

            # 如果健康列表为空，显示等待提示
            print(f"[ModelHub] 模型 {model_name}: 当前健康配置为 0/{total_configs}，等待中... (已等待 {elapsed_time}s/{max_wait_time}s)")

            # 等待后重试
            time.sleep(check_interval)

        # 如果超过最大等待时间仍未找到健康的配置，返回第一个未排除的配置
        available_configs = [(c, i) for i, c in enumerate(configs) if i not in excluded_indices]
        if available_configs:
            print(f"[ModelHub] 警告: 模型 {model_name} 在 {max_wait_time}s 内未找到健康配置，返回第一个可用配置")
            return available_configs[0]
        # 如果所有配置都被排除了，返回第一个配置（让调用方处理错误）
        print(f"[ModelHub] 警告: 模型 {model_name} 所有配置都已尝试，返回第一个配置")
        return (configs[0], 0)

    def call(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        request_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """调用模型 API，失败时自动重试其他配置"""
        tried_indices = []
        configs = self.models_config[model_name]

        assert len(configs) > 0

        while len(tried_indices) < len(configs):
            selected_config, config_index = self._select_config(model_name, tried_indices)
            adapter_class = _ADAPTERS[selected_config["adapter"]]
            client = adapter_class(selected_config)

            current_request_id = f"{request_id}_cfg{config_index}" if request_id else None
            result = client.call(messages, request_params or {}, request_id=current_request_id)

            # 如果结果不包含 error 字段，返回成功结果
            if "error" not in result:
                return result

            # 如果包含 error，记录已尝试的索引，继续尝试其他配置
            tried_indices.append(config_index)
            # NOTE: 原本会把整段 error（含 vLLM 返回的视频/图像 numpy 数组）全量 print 出来，
            # 刷屏严重。这里截断到前 300 字符；如需彻底静默，把下面两行注释掉即可。
            _err = str(result.get("error"))
            if len(_err) > 300:
                _err = _err[:300] + f"... <truncated, total {len(_err)} chars>"
            print(f"[ModelHub] 配置 {config_index} 请求失败，尝试其他配置: {_err}")
            # print(f"[ModelHub] 配置 {config_index} 请求失败，尝试其他配置: {result.get('error')}")

        # 所有配置都尝试过了，返回最后一个错误结果
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
