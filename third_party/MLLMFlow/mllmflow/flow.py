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
    """Multimodal intelligent conversation workflow builder

    Supports defining multi-turn conversations with JSON templates and integrating text, image, video, and other inputs,
    then calling large language models for processing.

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
        """Initialize MLLMFlow

        Args:
            models_config: Path to the model configuration file (JSON format), supporting strings or Path objects
            cache_dir: Media cache directory for storing processed videos and images
            model_replacement: Model name replacement mapping, such as {"gpt-4o": "gemini-pro-3-preview"}
            multi_stream_mode: Multi-stream mode.
                - "pixel" or default: non-multi-stream, single stream / template order.
                - "time"：Dual streams interleaved by time segment, A1 B1 A2 B2 ..., with Stream 1/2 labels for each segment.
                - "code"：Compare the two streams' change magnitude for each segment, input only the stream with larger changes, and replace the other with "Stream N: Unchanged".
                - "code_adaptive"：Adaptively control pixel scale based on the change magnitude of the two video streams in each segment (via fps scaling),
                  using higher pixel scale for the stream with larger changes (up to 2.0x) and lower pixel scale for the stream with smaller changes (can approach 0).
                  If one stream has no changes in that segment, output "Stream N: Unchanged" for that stream and use full pixel scale (2.0x) for the other.
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
        # New modes ``surge_token`` and ``cdpruner_token`` route to the
        # xstream_vllm_pruner plugin running inside the local vLLM worker.
        # They are accepted here so the MLLMFlow runtime is consistent; the
        # actual constraint that the backend must be local vLLM is enforced
        # later, when a model name is resolved.
        if self.multi_stream_mode not in (
            "pixel",
            "time",
            "code",
            "code_adaptive",
            "cdpruner",
            "surge",
            "cdpruner_token",
            "surge_token",
        ):
            self.multi_stream_mode = "pixel"
        self._placeholder_re = re.compile(r"\{\{(\w+):([^}]+)\}\}")

    TModel = TypeVar("TModel", bound=BaseModel)

    def _require_local_vllm_backend(self, model_name: str) -> None:
        """Ensure every backend registered for ``model_name`` is local vLLM.

        Used by the ``cdpruner_token`` / ``surge_token`` multi-stream modes
        which rely on the xstream_vllm_pruner plugin running inside a local
        vLLM worker. Hosted APIs cannot honour these modes because we have no
        way to install the patch-level pruner there.
        """
        configs = self.hub.models_config.get(model_name)
        if not configs:
            raise ValueError(
                f"multi_stream_mode={self.multi_stream_mode!r} requires a local "
                f"vLLM backend, but model {model_name!r} is not registered in "
                f"the model hub."
            )
        bad = [
            cfg for cfg in configs if not bool(cfg.get("is_vllm_local", False))
        ]
        if bad:
            adapters = sorted({cfg.get("adapter", "<unknown>") for cfg in bad})
            raise ValueError(
                f"multi_stream_mode={self.multi_stream_mode!r} is only supported "
                f"when every backend for model {model_name!r} has "
                f"is_vllm_local=True (found adapters: {adapters}). Use one of "
                f"the existing modes (pixel/time/code/code_adaptive/cdpruner/"
                f"surge) for hosted API models."
            )

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
            # Audit logging failures should not affect the main inference flow.
            pass

    def run(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the template workflow

        Args:
            template: JSON template containing vars and rounds

        Returns:
            Dictionary containing the following keys:
            - vars: Final variables dictionary
            - rounds: Complete message list for each conversation round

        Example:
            >>> template = {"vars": {}, "rounds": [{"round_id": "1", "messages": [...]}]}
            >>> result = flow.run(template)
            >>> print(result["vars"])  # Inspect variables
            >>> print(result["rounds"])  # Inspect conversation rounds
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

    def warm_video_cache(self, template: Dict[str, Any]) -> Dict[str, int]:
        """Resolve video placeholders only, forcing segment cache creation without model calls."""
        variables = dict(template.get("vars", {}))
        rounds = template.get("rounds", [])
        video_messages = 0
        video_segments = 0

        for round_data in rounds:
            round_id = str(round_data.get("round_id", "round"))
            round_parts: List[Dict] = []

            for turn_id, msg in enumerate(round_data.get("messages", []), 1):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not isinstance(content, str) or "{{video:" not in content:
                    continue

                video_messages += 1
                content = re.sub(r"\{\{(?!video:)[^}]+\}\}", "", content)
                resolved = self._resolve_content(
                    content,
                    variables,
                    round_parts,
                    request_id=f"{round_id}_{turn_id}_warm",
                )
                video_segments += sum(1 for part in resolved if part.get("type") == "video")
                round_parts.append({"type": "role", "role": role})
                round_parts.extend(resolved)

        return {"video_messages": video_messages, "video_segments": video_segments}

    @staticmethod
    def _resolve_video_end(resource: str, kwargs: Dict[str, str]) -> float:
        """Return explicit end time when provided; probe duration only as a fallback."""
        end = kwargs.get("end")
        if end not in (None, ""):
            return float(end)
        return float(_probe_video_duration(resource))

    def _resolve_content(
        self,
        template: str,
        variables: Dict[str, Any],
        context_parts: List[Dict],
        request_id: str = "",
    ) -> List[Dict]:
        """Resolve placeholders in content. When multi_stream_mode is time/code, apply special handling for multiple video(step) placeholders."""
        matches = list(self._placeholder_re.finditer(template))

        # time / code / code_adaptive / cdpruner / surge / *_token modes:
        # collect all video placeholders with step; use multi-stream logic
        # when there are at least two
        video_step_indices: List[int] = []
        video_specs: List[Tuple[str, Dict[str, str]]] = []
        if (
            self.multi_stream_mode
            in (
                "time",
                "code",
                "code_adaptive",
                "cdpruner",
                "surge",
                "cdpruner_token",
                "surge_token",
            )
            and matches
        ):
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

        # cdpruner / surge / *_token modes reuse time-mode interleaving during
        # multi-stream expansion; the actual token selection is performed in
        # the corresponding media_limit function (or, for ``*_token`` modes,
        # inside the vLLM worker via the xstream_vllm_pruner plugin).
        if (
            self.multi_stream_mode
            in ("time", "cdpruner", "surge", "cdpruner_token", "surge_token")
            and len(video_specs) >= 2
        ):
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

        # Original logic: resolve each placeholder in order
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
        """time mode: output multiple video(step) placeholders interleaved by segment and insert the corresponding stream label before each video (Stream 1: / Stream 2:).
        Each segment outputs 2 videos (A_i, B_i), so total videos = 2 * n_segments; for one video, the count is n_segments."""
        parts: List[Dict] = []
        first_idx = video_step_indices[0]
        last_idx = video_step_indices[-1]

        # Extract each stream's label from the template (text before the placeholder, such as "Stream 1: " or "\nStream 2: ")
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        # Compute the number of segments (based on the first video's step; all stream ranges are expected to match)
        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = self._resolve_video_end(video_specs[0][0], kw0) if video_specs else 0.0
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        # Interleave by segment: within each segment, output the stream label text first and then the video clip for each stream -> Stream 1: [A1] Stream 2: [B1] Stream 1: [A2] Stream 2: [B2] ...
        for seg_i in range(n_segments):
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                    parts.append({"type": "text", "text": stream_labels[stream_idx]})
                s = float(kwargs.get("start", 0))
                e = self._resolve_video_end(resource, kwargs)
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                f = float(kwargs["fps"]) if kwargs.get("fps") else None
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                clip_path = _trim_video_cached(resource, int(current_start), int(current_end), c_dir, f)
                parts.append({"type": "video", "video": clip_path})

        # Keep only the text after the last video placeholder (for example, the question "\nWhat is the man's...")
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
        """code mode: Compare the two streams' change magnitude for each segment, input only the stream with larger changes, and replace the other with "Stream N: Unchanged"."""
        parts: List[Dict] = []
        last_idx = video_step_indices[-1]

        # Extract each stream's label from the template (for example, "Stream 1: " or "\nStream 2: ")
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = self._resolve_video_end(video_specs[0][0], kw0) if video_specs else 0.0
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        for seg_i in range(n_segments):
            clip_paths: List[str] = []
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                s = float(kwargs.get("start", 0))
                e = self._resolve_video_end(resource, kwargs)
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                f = float(kwargs["fps"]) if kwargs.get("fps") else None
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                clip_path = _trim_video_cached(resource, int(current_start), int(current_end), c_dir, f)
                clip_paths.append(clip_path)

            # Compute each stream's change magnitude, output the stream with larger changes, and output "Stream N: Unchanged" for the other stream
            scores = [video_change_score(p) for p in clip_paths]
            n_streams = len(video_specs)
            chosen = 0
            for i in range(1, n_streams):
                if scores[i] > scores[chosen]:
                    chosen = i

            # Fixed order: always output Stream 1 before Stream 2 (...); for each stream, output the label first and then the video or Unchanged (Unchanged does not repeat the label to avoid "Stream 2: Stream 2: Unchanged")
            for stream_idx in range(n_streams):
                if stream_idx == chosen:
                    if stream_idx < len(stream_labels) and stream_labels[stream_idx]:
                        parts.append({"type": "text", "text": stream_labels[stream_idx]})
                    parts.append({"type": "video", "video": clip_paths[stream_idx]})
                else:
                    # Output only "Stream N: Unchanged" without adding that stream's label again (the label is already implied in Unchanged)
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
        """code_adaptive mode:
        - Adaptively control pixel scale based on the change magnitude (video_change_score) of the two video streams in each time segment;
        - Control pixel scale between 0 and 2x by adjusting fps;
        - When one stream has no changes (score=0) and the other has changes, output "Stream N: Unchanged" for the unchanged stream and use full pixel scale (2x) for the changed stream.
        """
        parts: List[Dict] = []
        last_idx = video_step_indices[-1]

        # Extract each stream's label from the template (for example, "Stream 1: " or "\nStream 2: ")
        stream_labels: List[str] = []
        for k, idx in enumerate(video_step_indices):
            if k == 0:
                span = template[0 : matches[idx].start()]
            else:
                span = template[matches[video_step_indices[k - 1]].end() : matches[idx].start()]
            stream_labels.append(span)

        # Estimate the number of segments based on the first video's step
        _, kw0 = video_specs[0]
        start = float(kw0.get("start", 0))
        end = self._resolve_video_end(video_specs[0][0], kw0) if video_specs else 0.0
        step = float(kw0["step"])
        n_segments = max(0, int((end - start) / step) if step else 0)

        # For stability, use a base fps (for example, 2.0) when fps is not explicitly provided
        BASE_FPS_DEFAULT = 2.0
        MIN_SCALE = 0.0
        MAX_SCALE = 2.0

        for seg_i in range(n_segments):
            scores: List[float] = []
            # First compute the change magnitude for each stream in this time segment
            segment_meta: List[Tuple[float, float, str, Dict[str, str]]] = []
            for stream_idx, (resource, kwargs) in enumerate(video_specs):
                s = float(kwargs.get("start", 0))
                e = self._resolve_video_end(resource, kwargs)
                st = float(kwargs.get("step", 1))
                c_dir = kwargs.get("cache_dir", self.cache_dir)
                current_start = s + seg_i * st
                current_end = min(current_start + st, e)
                segment_meta.append((current_start, current_end, c_dir, kwargs))

                # Do not force fps during scoring; use the original segment (or the fps in the template) so scaling itself does not affect the change-magnitude decision;
                # If fps <= 0 in the template, treat it as unspecified and let the lower layer decide.
                raw_fps_for_score = float(kwargs["fps"]) if kwargs.get("fps") else None
                if raw_fps_for_score is not None and raw_fps_for_score <= 0:
                    raw_fps_for_score = None
                score_clip_path = _trim_video_cached(
                    resource, int(current_start), int(current_end), c_dir, raw_fps_for_score
                )
                scores.append(video_change_score(score_clip_path))

            n_streams = len(video_specs)
            non_zero_scores = [s for s in scores if s > 0]

            # If no stream changes in this time segment, output Unchanged for all streams
            if not non_zero_scores:
                for stream_idx in range(n_streams):
                    parts.append({"type": "text", "text": f"Stream {stream_idx + 1}: Unchanged\n"})
                continue

            # If only one stream changes, use full pixel scale (2.0) for that stream and Unchanged for the others
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

            # When at least two streams change, linearly map non-zero scores to the [0.0, 2.0] range
            min_non_zero = min(non_zero_scores)
            max_non_zero = max(non_zero_scores)
            scales: List[float] = []
            if max_non_zero == min_non_zero:
                # If all changed streams have the same change magnitude, use 1.0 for all of them
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
        """Handle all placeholder types"""
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
            end = self._resolve_video_end(resource, kwargs)
            step = float(kwargs["step"]) if kwargs.get("step") else None
            fps = float(kwargs["fps"]) if kwargs.get("fps") else None
            cache_dir = kwargs.get("cache_dir", self.cache_dir)

            # If step is specified, split the video into multiple segments
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
                        # If the first block is already system, merge memory into the original system text area,
                        # avoiding two system messages (chat templates such as Qwen3.5 strictly require
                        # "System message must be at the beginning." the system message to appear only once at the beginning).
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

            # Choose different media limit strategies based on multi_stream_mode:
            # - default: _apply_media_limit keeps the last media_limit multimedia tokens in order;
            # - cdpruner mode: use a CDPruner-style selection strategy based on instruction relevance and diversity (applies to all models, including API models);
            # - surge mode: use a SURGE-style temporal surprise strategy to prune video tokens (only affects video; image/audio behavior matches the default strategy);
            # - cdpruner_token / surge_token modes: client side does NOT prune.
            #   The actual patch-level pruning runs inside the vLLM worker via
            #   the xstream_vllm_pruner plugin; the client simply forwards the
            #   instruction text via mm_processor_kwargs.
            request_extra: Dict[str, Any] = {}
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
            elif self.multi_stream_mode in ("cdpruner_token", "surge_token"):
                # Token-level pruning is only supported on local vLLM backends.
                self._require_local_vllm_backend(model_name)
                limited = _apply_media_limit(local_context_parts, media_limit)
                if self.multi_stream_mode == "cdpruner_token":
                    instruction_text = "".join(
                        p.get("text", "")
                        for p in local_context_parts
                        if p.get("type") == "text"
                    ).strip()
                    request_extra["_xstream_pruner"] = {
                        "algo": "cdpruner_token",
                        "instruction": instruction_text,
                    }
                else:
                    request_extra["_xstream_pruner"] = {"algo": "surge_token"}
            else:
                limited = _apply_media_limit(local_context_parts, media_limit)
            limited = _merge_text_parts(limited)
            messages = _construct_conversation(limited)

            start_time = time.time()
            print(f"[{request_id}][{model_name}] Sending Request ...")
            response = self.hub.call(
                model_name=model_name,
                messages=messages,
                request_params=request_extra,
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
