import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from model_hub import ModelHub
from pydantic import BaseModel

from mllmflow.video_utils.extract_frame import extract_frame_ffmpeg
from mllmflow.video_utils.probe_video import probe_video
from mllmflow.video_utils.video_change import video_change_score
from mllmflow.time_utils.time_utils import seconds_to_time
from mllmflow.memory_bank import MemoryOrchestrator
from mllmflow.utils import (
    _parse_args_str,
    _merge_text_parts,
    _construct_conversation,
    _apply_media_limit,
    _apply_media_limit_cdpruner,
    _apply_media_limit_surge,
    _probe_video_duration,
    _trim_video_cached,
    get_file_fingerprint,
)


class MLLMFlow:
    """多模态智能对话流程构建工具

    支持通过 JSON 格式模板定义多轮对话，集成文本、图像、视频等多种输入，
    并调用大语言模型进行处理。

    Example:
        >>> flow = MLLMFlow("models.json")
        >>> template = {
        ...     "vars": {},
        ...     "rounds": [{
        ...         "round_id": "1",
        ...         "messages": [
        ...             {"role": "user", "content": "{{file:prompt.txt}}"},
        ...             {"role": "assistant", "content": "{{model:gpt-4o,as=answer}}"}
        ...         ]
        ...     }]
        ... }
        >>> result = flow.run(template)
    """

    def __init__(
        self,
        models_config: str,
        cache_dir: str = "media_dir",
        model_replacement: Dict[str, str] = None,
        multi_stream_mode: str = "pixel",
        memory_bank: bool = False,
        memory_bank_model: Optional[str] = None,
        memory_bank_log_dir: Optional[str] = None,
    ):
        """初始化 MLLMFlow

        Args:
            models_config: 模型配置文件路径（JSON 格式），支持字符串或 Path 对象
            cache_dir: 媒体文件缓存目录，用于存储处理后的视频和图片
            model_replacement: 模型名称替换映射，如 {"gpt-4o": "gemini-pro-3-preview"}
            multi_stream_mode: 多流模式。
                - "pixel" 或默认：非 multi-stream，单路/按模板顺序。
                - "time"：双流按时间片段交错，A1 B1 A2 B2 ...，每段带 Stream 1/2 标签。
                - "code"：每段比较两路变化量，只输入变化较大的一路视频，另一路用 "Stream N: Unchanged" 代替。
                - "code_adaptive"：依据每段两路视频的变化量自适应控制像素大小（通过 fps 缩放），
                  对变化更大的一路使用更高像素（最多 2.0 倍），变化更小的一路使用更低像素（可接近 0）。
                  若某一路在该段完全无变化，则该路输出 "Stream N: Unchanged"，另一条使用满像素（2.0 倍）。
        """
        self.hub = ModelHub(models_config)
        self.cache_dir = cache_dir
        self.model_replacement = model_replacement or {}
        self.memory_bank_enabled = bool(memory_bank)
        self.memory_bank_model = (memory_bank_model or "").strip() or None
        self._memory_orchestrator: Optional[MemoryOrchestrator] = None
        self._memory_runtime_model: Optional[str] = None
        self.memory_bank_log_dir = (memory_bank_log_dir or "").strip() or None
        self._memory_audit_path: Optional[Path] = None
        self._memory_audit_index: int = 0
        if self.memory_bank_enabled and self.memory_bank_log_dir:
            log_dir = Path(self.memory_bank_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._memory_audit_path = log_dir / (
                f"memory_bank_audit_{int(time.time() * 1000)}_{os.getpid()}.jsonl"
            )
        self.multi_stream_mode = (multi_stream_mode or "pixel").strip().lower()
        if self.multi_stream_mode not in ("pixel", "time", "code", "code_adaptive", "cdpruner", "surge"):
            self.multi_stream_mode = "pixel"
        self._placeholder_re = re.compile(r"\{\{(\w+):([^}]+)\}\}")

    TModel = TypeVar("TModel", bound=BaseModel)

    def _call_memory_llm(
        self, prompt: str, response_format: Optional[Type[TModel]] = None
    ) -> Union[str, TModel]:
        if not self._memory_runtime_model:
            raise RuntimeError("memory bank model is not initialized")
        response = self.hub.call(
            model_name=self._memory_runtime_model,
            messages=[{"role": "user", "content": prompt}],
            request_params={"temperature": 0},
            request_id=f"memory_bank_{int(time.time() * 1000)}",
        )
        content = response.get("content", str(response))
        if response_format is None:
            return content
        json_text = self._extract_json_block(content)
        return response_format.model_validate_json(json_text)

    @staticmethod
    def _extract_json_block(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and first <= last:
            return text[first : last + 1]
        return text

    @staticmethod
    def _extract_latest_user_text(context_parts: List[Dict[str, Any]]) -> str:
        blocks: List[Tuple[str, List[str]]] = []
        current_role: Optional[str] = None
        current_texts: List[str] = []
        for part in context_parts:
            if part.get("type") == "role":
                if current_role is not None:
                    blocks.append((current_role, current_texts))
                current_role = str(part.get("role", ""))
                current_texts = []
                continue
            if part.get("type") == "text" and current_role is not None:
                txt = str(part.get("text", "")).strip()
                if txt:
                    current_texts.append(txt)
        if current_role is not None:
            blocks.append((current_role, current_texts))
        for role, texts in reversed(blocks):
            if role == "user" and texts:
                return "\n".join(texts)
        return ""

    def _build_memory_context(self, query: str) -> str:
        if self._memory_orchestrator is None:
            return ""
        candidates = self._memory_orchestrator.retrieve_topk_events(query, top_k=6)
        distilled = self._memory_orchestrator.distill_events(query, candidates)
        events_text = "\n".join(
            f"- [{event.timestamp}] {event.event_description}" for event in distilled
        )
        if not events_text:
            events_text = "- (no distilled events)"
        narrative = self._memory_orchestrator.narrative_state()
        narrative_json = json.dumps(narrative, ensure_ascii=False, indent=2)
        memory_guidance = (
            "Memory is the broadcast's SHORT-TERM BRAIN — it keeps the clock\n"
            "walking, the scoreboard honest, and possession coherent across\n"
            "seconds so the next director decision is not lonely.\n"
            "\n"
            "How to use it (MANDATORY):\n"
            "1) NarrativeState.clock is the authoritative game clock the\n"
            "   memory has maintained. Echo it verbatim in `atomic_event.clock`\n"
            "   unless the on-screen scoreboard clearly shows a different\n"
            "   value this second. Do NOT invent a new clock from nothing.\n"
            "2) NarrativeState.period is the current period label — reuse as\n"
            "   `atomic_event.period`.\n"
            "3) NarrativeState.score is the cumulative team score maintained\n"
            "   by memory via `points_delta`. Set `score_home_after =\n"
            "   NarrativeState.score.home` and `score_away_after =\n"
            "   NarrativeState.score.away` for every second where the game\n"
            "   clock is running and no new point has been made.\n"
            "4) ONLY when you actually see a made basket this second, emit a\n"
            "   scoring_event: set `event_type=\"scoring_event\"`, a fresh\n"
            "   `event_id`, `result=\"made\"`, `team_id` for the scorer, and\n"
            "   `points_delta` in {1,2,3}. Memory will compute the NEW total\n"
            "   score for you — you do NOT need to pre-add; keep\n"
            "   `score_*_after` equal to NarrativeState.score (memory will\n"
            "   update on the next round).\n"
            "5) `overlay_text`: when memory has both `score` and `clock`,\n"
            "   format as `HOME <h>-<a> AWAY | <PERIOD> <MM:SS>`. Reuse the\n"
            "   same string across consecutive seconds while none of the\n"
            "   fields change. When `score` is still empty (early game) leave\n"
            "   `overlay_text` null — do not hallucinate 0-0.\n"
            "6) NarrativeState.possession_team (`home`/`away`) is the last\n"
            "   team in control. Prefer the stream that frames that team's\n"
            "   offense; reference them in `commentary_en` as `the home side`\n"
            "   / `the away side`, NOT by fabricated jersey color.\n"
            "7) NarrativeState.recent_scoring_events is the last 4 made\n"
            "   baskets. If a mandatory-replay event happened in the last 6\n"
            "   seconds AND no replay has aired yet, plan a `start_effect`\n"
            "   with `slow_motion factor=0.5` now (see MANDATORY REPLAY).\n"
            "8) For routine seconds, REUSE the same `event_id` from memory's\n"
            "   most recent atomic_event so memory performs UPDATE / NOOP\n"
            "   instead of ADD.\n"
            "9) For replay seconds (between start_effect and end_effect),\n"
            "   reuse the originating live event's `event_id` and do NOT\n"
            "   re-emit a new scoring delta.\n"
        )
        return (
            "[MemoryBank Context]\n"
            "Use these distilled memories as extra context.\n"
            f"{memory_guidance}\n"
            f"NarrativeState={narrative_json}\n"
            f"GlobalMemory={self._memory_orchestrator.global_memory.model_dump_json(indent=2)}\n"
            f"DistilledEvents=\n{events_text}"
        )

    def _append_memory_audit(
        self,
        request_id: str,
        latest_user_text: str,
        timestamp: str,
        update: Any,
    ) -> None:
        if self._memory_audit_path is None or self._memory_orchestrator is None:
            return
        self._memory_audit_index += 1
        payload = {
            "audit_index": self._memory_audit_index,
            "request_id": request_id,
            "event_timestamp": timestamp,
            "event_text": latest_user_text,
            "global_memory_update": (
                update.model_dump() if hasattr(update, "model_dump") else str(update)
            ),
            "snapshot": self._memory_orchestrator.snapshot(),
            "logged_at": int(time.time()),
        }
        try:
            with self._memory_audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # 审计日志失败不应影响主推理流程。
            pass

    def run(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """执行模板流程

        Args:
            template: JSON 格式模板，包含 vars 和 rounds

        Returns:
            包含以下键的字典：
            - vars: 最终变量字典
            - rounds: 每轮对话的完整消息列表

        Example:
            >>> template = {"vars": {}, "rounds": [{"round_id": "1", "messages": [...]}]}
            >>> result = flow.run(template)
            >>> print(result["vars"])  # 查看变量
            >>> print(result["rounds"])  # 查看对话轮次
        """
        variables = dict(template.get("vars", {}))
        rounds = template.get("rounds", [])
        rounds_out = []

        for round_data in rounds:
            round_id = str(round_data["round_id"])
            round_parts: List[Dict] = []

            for turn_id, msg in enumerate(round_data["messages"], 1):
                role = msg["role"]
                content = msg["content"]

                if isinstance(content, str):
                    resolved = self._resolve_content(content, variables, round_parts, request_id=f"{round_id}_{turn_id}")
                elif isinstance(content, list):
                    resolved = content
                else:
                    raise TypeError(f"message content should be str or list, but get {type(content)}: {content}")

                round_parts.append({"type": "role", "role": role})
                round_parts.extend(resolved)

            conv = _construct_conversation(round_parts)
            rounds_out.append({"round_id": round_id, "messages": conv})

        return {"vars": variables, "rounds": rounds_out}

    def _resolve_content(
        self,
        template: str,
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> List[Dict]:
        """解析内容中的占位符。multi_stream_mode 为 time/code 时，对多路 video(step) 做特殊处理。"""
        matches = list(self._placeholder_re.finditer(template))

        # time / code / code_adaptive / cdpruner / surge 模式：收集所有带 step 的 video 占位符，若有 >=2 个则走多流逻辑
        video_step_indices: List[int] = []
        video_specs: List[Tuple[str, Dict[str, str]]] = []
        if self.multi_stream_mode in ("time", "code", "code_adaptive", "cdpruner", "surge") and matches:
            for idx, m in enumerate(matches):
                tag_type = m.group(1)
                args_str = m.group(2)
                if tag_type != "video":
                    continue
                resource, kwargs = _parse_args_str(args_str)
                if not kwargs.get("step"):
                    continue
                video_step_indices.append(idx)
                video_specs.append((resource, kwargs))

        # cdpruner / surge 模式在多流展开阶段沿用 time 的交错逻辑，真正的 token 选择在对应的 media_limit 函数中完成
        if self.multi_stream_mode in ("time", "cdpruner", "surge") and len(video_specs) >= 2:
            return self._resolve_content_multi_stream(
                template, matches, video_step_indices, video_specs, variables, context_parts, request_id
            )
        if self.multi_stream_mode == "code" and len(video_specs) >= 2:
            return self._resolve_content_multi_stream_code(
                template, matches, video_step_indices, video_specs, variables, context_parts, request_id
            )
        if self.multi_stream_mode == "code_adaptive" and len(video_specs) >= 2:
            return self._resolve_content_multi_stream_code_adaptive(
                template, matches, video_step_indices, video_specs, variables, context_parts, request_id
            )

        # 原有逻辑：顺序解析每个占位符
        parts = []
        last_end = 0
        for match in matches:
            if match.start() > last_end:
                text = template[last_end : match.start()]
                if text:
                    parts.append({"type": "text", "text": text})

            tag_type = match.group(1)
            args_str = match.group(2)
            resource, kwargs = _parse_args_str(args_str)

            result = self._handle_placeholder(tag_type, resource, kwargs, variables, context_parts, request_id)

            if isinstance(result, list):
                parts.extend(result)
            elif isinstance(result, dict):
                parts.append(result)
            elif isinstance(result, str):
                parts.append({"type": "text", "text": result})

            last_end = match.end()

        if last_end < len(template):
            trailing = template[last_end:]
            if trailing:
                parts.append({"type": "text", "text": trailing})

        return _merge_text_parts(parts)

    def _resolve_content_multi_stream(
        self,
        template: str,
        matches: List[re.Match],
        video_step_indices: List[int],
        video_specs: List[Tuple[str, Dict[str, str]]],
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> List[Dict]:
        """time 模式：多个 video(step) 按片段交错输出，且每个视频前插入对应 stream 标签（Stream 1: / Stream 2:）。
        每段输出 2 个视频（A_i, B_i），总视频数 = 2 * n_segments；单视频时为 n_segments。"""
        parts: List[Dict] = []
        first_idx = video_step_indices[0]
        last_idx = video_step_indices[-1]

        # 从模板中提取每个 stream 的标签（占位符前的文本，如 "Stream 1: "、"\nStream 2: "）
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        # 计算片段数（以第一个 video 的 step 为准，要求各 stream 范围一致）
        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = float(kw0.get("end", _probe_video_duration(video_specs[0][0]) if video_specs else 0))
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        # 按片段交错：每段内对每个 stream 先输出该 stream 的标签文本，再输出视频片段 → Stream 1: [A1] Stream 2: [B1] Stream 1: [A2] Stream 2: [B2] ...
        for seg_i in range(n_segments):
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                    parts.append({"type": "text", "text": stream_labels[stream_idx]})
                s = float(kwargs.get("start", 0))
                e = float(kwargs.get("end", _probe_video_duration(resource)))
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                f = float(kwargs["fps"]) if kwargs.get("fps") else None
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                clip_path = _trim_video_cached(resource, int(current_start), int(current_end), c_dir, f)
                parts.append({"type": "video", "video": clip_path})

        # 仅保留最后一个 video 占位符之后的文本（如问题 "\nWhat is the man's..."）
        after_last = matches[last_idx].end()
        if after_last < len(template):
            trailing = template[after_last:]
            if trailing:
                parts.append({"type": "text", "text": trailing})

        return _merge_text_parts(parts)

    def _resolve_content_multi_stream_code(
        self,
        template: str,
        matches: List[re.Match],
        video_step_indices: List[int],
        video_specs: List[Tuple[str, Dict[str, str]]],
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> List[Dict]:
        """code 模式：每段比较两路变化量，只输入变化较大的一路视频，另一路用 "Stream N: Unchanged" 代替。"""
        parts: List[Dict] = []
        last_idx = video_step_indices[-1]

        # 从模板中提取每个 stream 的标签（如 "Stream 1: "、"\nStream 2: "）
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = float(kw0.get("end", _probe_video_duration(video_specs[0][0]) if video_specs else 0))
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        for seg_i in range(n_segments):
            clip_paths: List[str] = []
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                s = float(kwargs.get("start", 0))
                e = float(kwargs.get("end", _probe_video_duration(resource)))
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                f = float(kwargs["fps"]) if kwargs.get("fps") else None
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                clip_path = _trim_video_cached(resource, int(current_start), int(current_end), c_dir, f)
                clip_paths.append(clip_path)

            # 计算每路变化量，选择变化较大的一路输出视频，另一路输出 "Stream N: Unchanged"
            scores = [video_change_score(p) for p in clip_paths]
            n_streams = len(video_specs)
            chosen = 0
            for i in range(1, n_streams):
                if scores[i] > scores[chosen]:
                    chosen = i

            # 固定顺序：始终先 Stream 1 再 Stream 2（…），每路先出标签再出视频或 Unchanged（Unchanged 不再重复标签避免 "Stream 2: Stream 2: Unchanged"）
            for stream_idx in range(n_streams):
                if stream_idx == chosen:
                    if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                        parts.append({"type": "text", "text": stream_labels[stream_idx]})
                    parts.append({"type": "video", "video": clip_paths[stream_idx]})
                else:
                    # 只输出 "Stream N: Unchanged"，不再加该路 label（label 已隐含在 Unchanged 里）
                    parts.append({"type": "text", "text": f"Stream {stream_idx + 1}: Unchanged\n"})

        after_last = matches[last_idx].end()
        if after_last < len(template):
            trailing = template[after_last:]
            if trailing:
                parts.append({"type": "text", "text": trailing})

        return _merge_text_parts(parts)

    def _resolve_content_multi_stream_code_adaptive(
        self,
        template: str,
        matches: List[re.Match],
        video_step_indices: List[int],
        video_specs: List[Tuple[str, Dict[str, str]]],
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> List[Dict]:
        """code_adaptive 模式：
        - 依据每个时间片段两路视频的变化量（video_change_score）自适应控制像素大小；
        - 通过调节 fps，将像素大小控制在 0～2 倍之间；
        - 当一条完全无变化（score=0）且另一条有变化时，无变化的一路输出 "Stream N: Unchanged"，有变化的一路使用满像素（2 倍）。
        """
        parts: List[Dict] = []
        last_idx = video_step_indices[-1]

        # 从模板中提取每个 stream 的标签（如 "Stream 1: "、"\nStream 2: "）
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        # 以第一个 video 的 step 为准估算片段数
        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = float(kw0.get("end", _probe_video_duration(video_specs[0][0]) if video_specs else 0))
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        # 为了稳定性，未显式提供 fps 时使用一个基础 fps（例如 2.0）
        BASE_FPS_DEFAULT = 2.0
        MIN_SCALE = 0.0
        MAX_SCALE = 2.0

        for seg_i in range(n_segments):
            scores: List[float] = []
            # 先为每个 stream 计算这一时间片段的变化量
            segment_meta: List[Tuple[float, float, str, Dict[str, str]]] = []
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                s = float(kwargs.get("start", 0))
                e = float(kwargs.get("end", _probe_video_duration(resource)))
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                segment_meta.append((current_start, current_end, c_dir, kwargs))

                # 评分时不强制 fps，使用原始片段（或模板中的 fps），避免缩放本身影响“变化量”判断；
                # 若模板里 fps<=0，则视为未指定，交由底层自行决定。
                raw_fps_for_score = float(kwargs["fps"]) if kwargs.get("fps") else None
                if raw_fps_for_score is not None and raw_fps_for_score <= 0:
                    raw_fps_for_score = None
                score_clip_path = _trim_video_cached(
                    resource, int(current_start), int(current_end), c_dir, raw_fps_for_score
                )
                scores.append(video_change_score(score_clip_path))

            n_streams = len(video_specs)
            non_zero_scores = [s for s in scores if s > 0]

            # 若所有路在该时间片段都无变化，则全部输出 Unchanged
            if not non_zero_scores:
                for stream_idx in range(n_streams):
                    parts.append({"type": "text", "text": f"Stream {stream_idx + 1}: Unchanged\n"})
                continue

            # 若只有一路有变化：该路用满像素（2.0），其他路 Unchanged
            if len(non_zero_scores) == 1:
                changed_idx = scores.index(non_zero_scores[0])
                for stream_idx in range(n_streams):
                    if stream_idx == changed_idx:
                        if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                            parts.append({"type": "text", "text": stream_labels[stream_idx]})
                        resource, kwargs = video_specs[stream_idx]
                        current_start, current_end, c_dir, _ = segment_meta[stream_idx]
                        base_fps = float(kwargs["fps"]) if kwargs.get("fps") else BASE_FPS_DEFAULT
                        if base_fps <= 0:
                            base_fps = BASE_FPS_DEFAULT
                        fps = base_fps * MAX_SCALE
                        clip_path = _trim_video_cached(
                            resource, int(current_start), int(current_end), c_dir, fps
                        )
                        parts.append({"type": "video", "video": clip_path})
                    else:
                        parts.append({"type": "text", "text": f"Stream {stream_idx + 1}: Unchanged\n"})
                continue

            # 至少两路有变化：将非零 score 线性映射到 [0.0, 2.0] 区间
            min_non_zero = min(non_zero_scores)
            max_non_zero = max(non_zero_scores)
            scales: List[float] = []
            if max_non_zero == min_non_zero:
                # 所有有变化的路变化量相同，则统一用 1.0
                scales = [1.0 if s > 0 else 0.0 for s in scores]
            else:
                for s in scores:
                    if s <= 0:
                        scales.append(0.0)
                    else:
                        norm = (s - min_non_zero) / (max_non_zero - min_non_zero)
                        scale = MIN_SCALE + norm * (MAX_SCALE - MIN_SCALE)
                        scales.append(scale)

            for stream_idx in range(n_streams):
                score = scores[stream_idx]
                if score <= 0:
                    parts.append({"type": "text", "text": f"Stream {stream_idx + 1}: Unchanged\n"})
                    continue

                scale = max(MIN_SCALE, min(MAX_SCALE, scales[stream_idx]))
                if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                    parts.append({"type": "text", "text": stream_labels[stream_idx]})

                resource, kwargs = video_specs[stream_idx]
                current_start, current_end, c_dir, _ = segment_meta[stream_idx]
                base_fps = float(kwargs["fps"]) if kwargs.get("fps") else BASE_FPS_DEFAULT
                if base_fps <= 0:
                    base_fps = BASE_FPS_DEFAULT
                fps = base_fps * scale
                clip_path = _trim_video_cached(
                    resource, int(current_start), int(current_end), c_dir, fps
                )
                parts.append({"type": "video", "video": clip_path})

        after_last = matches[last_idx].end()
        if after_last < len(template):
            trailing = template[after_last:]
            if trailing:
                parts.append({"type": "text", "text": trailing})

        return _merge_text_parts(parts)

    def _handle_placeholder(
        self,
        tag_type: str,
        resource: str,
        kwargs: Dict[str, str],
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> Any:
        """处理各种占位符"""
        if tag_type == "var":
            return {"type": "text", "text": str(variables.get(resource, ""))}

        elif tag_type == "file":
            with open(resource, "r", encoding="utf-8") as f:
                return {"type": "text", "text": f.read()}

        elif tag_type == "image":
            time_sec = kwargs.get("time")
            img_path = resource
            if time_sec is not None:
                try:
                    t = float(time_sec)
                    cache_dir = kwargs.get("cache_dir", self.cache_dir)
                    out_path = Path(cache_dir) / f"{get_file_fingerprint(resource)}_{t:.1f}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if extract_frame_ffmpeg(resource, str(out_path), t):
                        img_path = str(out_path.resolve())
                except (ValueError, TypeError):
                    pass
            return {"type": "image", "image": img_path}

        elif tag_type == "video":
            start = float(kwargs.get("start", 0))
            end = float(kwargs.get("end", _probe_video_duration(resource)))
            step = float(kwargs["step"]) if kwargs.get("step") else None
            fps = float(kwargs["fps"]) if kwargs.get("fps") else None
            cache_dir = kwargs.get("cache_dir", self.cache_dir)

            # 如果指定了 step，将视频切分成多段
            if step:
                video_list = []
                current_start = start
                while current_start < end:
                    current_end = min(current_start + step, end)
                    clip_path = _trim_video_cached(resource, int(current_start), int(current_end), cache_dir, fps)
                    video_list.append({"type": "video", "video": clip_path})
                    current_start += step
                return video_list
            else:
                video_path = _trim_video_cached(resource, int(start), int(end), cache_dir, fps)
                return {"type": "video", "video": video_path}

        elif tag_type == "model":
            model_name = self.model_replacement.get(resource, resource)
            if self.memory_bank_enabled and self._memory_orchestrator is None:
                self._memory_runtime_model = self.memory_bank_model or model_name
                self._memory_orchestrator = MemoryOrchestrator(call_llm=self._call_memory_llm)

            local_context_parts = context_parts
            if self.memory_bank_enabled and self._memory_orchestrator is not None:
                latest_user_text = self._extract_latest_user_text(context_parts)
                if latest_user_text:
                    m = re.search(
                        r"(Q\d+\s+\d{1,2}:\d{2}|\d{1,2}:\d{2}|[0-9]+\.[0-9]+s?)",
                        latest_user_text,
                        flags=re.IGNORECASE,
                    )
                    timestamp = m.group(1) if m else request_id
                    update = self._memory_orchestrator.add_event(
                        timestamp=timestamp,
                        event_description=latest_user_text,
                        importance_score=0.7,
                    )
                    self._append_memory_audit(
                        request_id=request_id,
                        latest_user_text=latest_user_text,
                        timestamp=timestamp,
                        update=update,
                    )
                    memory_text = self._build_memory_context(latest_user_text)
                    if memory_text:
                        # 若首个 block 已是 system，则把 memory 合并进原 system 的文本区，
                        # 避免出现两个 system 消息（Qwen3.5 等 chat template 会严格要求
                        # "System message must be at the beginning." 只允许首位出现一次）。
                        if (
                            context_parts
                            and context_parts[0].get("type") == "role"
                            and context_parts[0].get("role") == "system"
                        ):
                            local_context_parts = (
                                [context_parts[0], {"type": "text", "text": memory_text + "\n\n"}]
                                + context_parts[1:]
                            )
                        else:
                            local_context_parts = [
                                {"type": "role", "role": "system"},
                                {"type": "text", "text": memory_text},
                            ] + context_parts

            media_limit = int(kwargs.pop("media_limit", 10000))
            as_name = kwargs.pop("as", None)
            return_flag = int(kwargs.pop("return", 1))
            request_id = kwargs.pop("request_id", request_id)

            # 根据 multi_stream_mode 选择不同的 media limit 策略：
            # - 默认：_apply_media_limit 按顺序保留最后 media_limit 个多媒体 token；
            # - cdpruner 模式：使用 CDPruner 风格的基于指令与多样性的选择策略（对所有模型生效，包括 API 模型）；
            # - surge 模式：使用 SURGE 风格的时间惊讶度策略对 video token 进行剪枝（仅作用于 video，image/audio 行为与默认策略一致）。
            if self.multi_stream_mode == "cdpruner":
                instruction_text = "".join(
                    p.get("text", "")
                    for p in local_context_parts
                    if p.get("type") == "text"
                ).strip()
                limited = _apply_media_limit_cdpruner(
                    local_context_parts,
                    media_limit,
                    instruction_text=instruction_text,
                )
            elif self.multi_stream_mode == "surge":
                limited = _apply_media_limit_surge(
                    local_context_parts,
                    media_limit,
                )
            else:
                limited = _apply_media_limit(local_context_parts, media_limit)
            limited = _merge_text_parts(limited)
            messages = _construct_conversation(limited)

            start_time = time.time()
            print(f"[{request_id}][{model_name}] Sending Request ...")
            response = self.hub.call(
                model_name=model_name,
                messages=messages,
                request_params={},
                request_id=request_id
            )
            latency = time.time() - start_time
            print(f"[{request_id}][{model_name}] latency: {latency:.2f}s\n")

            content = response.get("content", str(response)) if isinstance(response, dict) else str(response)

            if self.memory_bank_enabled and self._memory_orchestrator is not None:
                post_update = self._memory_orchestrator.add_model_output(
                    timestamp=request_id,
                    model_output=content,
                    importance_score=0.85,
                )
                self._append_memory_audit(
                    request_id=f"{request_id}_model_output",
                    latest_user_text=content,
                    timestamp=request_id,
                    update=post_update,
                )

            if as_name:
                variables[as_name] = content

            if return_flag:
                return {"type": "text", "text": content}
            else:
                return None

        else:
            return {"type": "text", "text": f"{{{{{tag_type}:{resource}}}}}"}
