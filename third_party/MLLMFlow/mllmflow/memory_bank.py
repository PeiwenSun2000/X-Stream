from __future__ import annotations

import json
import re
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import requests
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class TrackingStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    ENDED = "ended"
    UNKNOWN = "unknown"


class HighlightTimeRange(BaseModel):
    start: float
    end: float


class AtomicEvent(BaseModel):
    """
    场景泛化的最小闭环字段（尽量都可选）：
    不强绑定某单一运动，可容纳篮球/排球/棒球/CS/LOL 等。
    """

    game_id: Optional[str] = None
    event_id: Optional[str] = None
    period: Optional[str] = None
    clock: Optional[str] = None
    team_id: Optional[str] = None
    primary_player_id: Optional[str] = None
    event_type: Optional[str] = None
    sub_type: Optional[str] = None
    result: Optional[str] = None
    points_delta: Optional[int] = None
    secondary_player_id: Optional[str] = None
    lineup_home: Optional[List[str]] = None
    lineup_away: Optional[List[str]] = None
    score_home_after: Optional[int] = None
    score_away_after: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream_id: str
    content: str
    importance: float = 0.5
    timestamp: Optional[str] = None
    highlight_time_range: Optional[HighlightTimeRange] = None
    tracking_status: TrackingStatus = TrackingStatus.UNKNOWN
    metadata: Dict[str, Any] = Field(default_factory=dict)
    atomic_event: Optional[AtomicEvent] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))


class StreamEvent(BaseModel):
    timestamp: str
    event_description: str
    importance_score: float = 0.5
    stream_id: str = "global"
    memory_id: str = ""
    tracking_status: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    atomic_event: Optional[AtomicEvent] = None


class MemoryActionResult(BaseModel):
    operation: MemoryOperation
    memory_id: Optional[str] = None
    stream_id: Optional[str] = None
    notes: str = ""
    fused_global_updates: Dict[str, Any] = Field(default_factory=dict)


class GlobalMemory(BaseModel):
    score: Dict[str, int] = Field(default_factory=dict)
    player_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    shooting_percentage: Dict[str, float] = Field(default_factory=dict)
    hit_rate: Dict[str, float] = Field(default_factory=dict)
    kda: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    global_events: Dict[str, Any] = Field(default_factory=dict)
    confidence: Dict[str, float] = Field(default_factory=dict)
    # Narrative state — concrete, generally-useful fields maintained by
    # ``MemoryOrchestrator`` so the director can rely on a stable view of
    # scoreboard / clock / possession even when the current frame fails to
    # reveal them.
    period: Optional[str] = None
    clock: Optional[str] = None
    # Monotonic counter of per-second rounds the orchestrator has observed;
    # used to drive clock auto-advance when the model omits the clock.
    tick_count: int = 0
    possession_team: Optional[str] = None
    last_scoring_event: Optional[Dict[str, Any]] = None
    scoring_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated_ts: int = Field(default_factory=lambda: int(time.time()))


def _extract_json_block(text: str) -> str:
    if not text:
        return "{}"
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first <= last:
        return text[first : last + 1]
    return text.strip()


class OpenAICompatibleLLMCaller:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8901/v1",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 4,
        retry_sleep_seconds: float = 1.5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def call_llm(self, prompt: str, response_format: Optional[Type[T]] = None) -> Union[str, T]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if response_format is None:
                    return content
                return response_format.model_validate_json(_extract_json_block(content))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed without explicit exception")


class StreamMemoryManager:
    def __init__(self) -> None:
        self._items_by_stream: Dict[str, List[MemoryItem]] = {}

    def add_memory(self, item: MemoryItem) -> MemoryActionResult:
        self._items_by_stream.setdefault(item.stream_id, []).append(item)
        return MemoryActionResult(operation=MemoryOperation.ADD, memory_id=item.memory_id, stream_id=item.stream_id, notes="local memory added")

    def update_memory(self, memory_id: str, stream_id: str, **updates: Any) -> MemoryActionResult:
        for item in self._items_by_stream.get(stream_id, []):
            if item.memory_id != memory_id:
                continue
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            item.updated_at = int(time.time())
            return MemoryActionResult(operation=MemoryOperation.UPDATE, memory_id=memory_id, stream_id=stream_id, notes="local memory updated")
        return MemoryActionResult(operation=MemoryOperation.NOOP, memory_id=memory_id, stream_id=stream_id, notes="memory item not found")

    def delete_memory(self, memory_id: str, stream_id: str) -> MemoryActionResult:
        items = self._items_by_stream.get(stream_id, [])
        for idx, item in enumerate(items):
            if item.memory_id == memory_id:
                items.pop(idx)
                return MemoryActionResult(operation=MemoryOperation.DELETE, memory_id=memory_id, stream_id=stream_id, notes="local memory deleted")
        return MemoryActionResult(operation=MemoryOperation.NOOP, memory_id=memory_id, stream_id=stream_id, notes="memory item not found")

    def query_local(self, stream_id: Optional[str] = None, min_importance: float = 0.0, limit: Optional[int] = None) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        if stream_id is None:
            for per_stream in self._items_by_stream.values():
                items.extend(per_stream)
        else:
            items.extend(self._items_by_stream.get(stream_id, []))
        items = [x for x in items if x.importance >= min_importance]
        items.sort(key=lambda x: (x.importance, x.updated_at), reverse=True)
        return items[:limit] if limit is not None else items

    def select_memories(self, query: str = "", stream_ids: Optional[List[str]] = None, min_importance: float = 0.0, top_k: int = 6) -> List[MemoryItem]:
        stream_ids = stream_ids or list(self._items_by_stream.keys())
        all_items: List[MemoryItem] = []
        for sid in stream_ids:
            all_items.extend(self._items_by_stream.get(sid, []))
        filtered = [item for item in all_items if item.importance >= min_importance]
        query_terms = {x for x in re.split(r"\W+", query.lower()) if x}

        def _score(item: MemoryItem) -> float:
            item_terms = {x for x in re.split(r"\W+", item.content.lower()) if x}
            overlap = len(query_terms & item_terms) if query_terms else 0
            return overlap * 2.0 + item.importance + (item.updated_at / 1e12)

        return sorted(filtered, key=_score, reverse=True)[:top_k]

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {sid: [item.model_dump() for item in items] for sid, items in self._items_by_stream.items()}


class GlobalMemoryManager:
    def __init__(self) -> None:
        self.state = GlobalMemory()

    def _set_with_conflict_resolution(self, key: str, value: Any, confidence: float = 0.6) -> None:
        old_conf = float(self.state.confidence.get(key, 0.0))
        if key in self.state.global_events and old_conf > confidence:
            return
        self.state.global_events[key] = value
        self.state.confidence[key] = confidence

    def _merge_player_stats(self, player: str, delta: Dict[str, float]) -> None:
        cur = self.state.player_stats.setdefault(player, {})
        for k, v in delta.items():
            cur[k] = float(cur.get(k, 0.0)) + float(v)

    def fuse_to_global(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        updates: Dict[str, Any] = {"score": {}, "player_stats": {}, "kda": {}, "global_events": {}}
        for item in memories:
            md = item.metadata or {}
            ae = item.atomic_event
            score_update = md.get("score_update")
            if isinstance(score_update, dict):
                for team, delta in score_update.items():
                    self.state.score[team] = int(self.state.score.get(team, 0)) + int(delta)
                    updates["score"][team] = self.state.score[team]
            player_stats = md.get("player_stats")
            if isinstance(player_stats, dict):
                for player, delta in player_stats.items():
                    if isinstance(delta, dict):
                        self._merge_player_stats(player, delta)
                        updates["player_stats"][player] = self.state.player_stats[player]
            kda_update = md.get("kda")
            if isinstance(kda_update, dict):
                for player, kda in kda_update.items():
                    if isinstance(kda, dict):
                        self.state.kda[player] = {
                            "kills": float(kda.get("kills", 0)),
                            "deaths": float(kda.get("deaths", 0)),
                            "assists": float(kda.get("assists", 0)),
                        }
                        updates["kda"][player] = self.state.kda[player]
            global_state = md.get("global_state")
            if isinstance(global_state, dict):
                for k, v in global_state.items():
                    self._set_with_conflict_resolution(k, v, confidence=item.importance)
                    updates["global_events"][k] = self.state.global_events[k]
            if ae and ae.event_type:
                key = f"event_type::{ae.event_type}"
                self._set_with_conflict_resolution(key, {"last_event_id": ae.event_id, "result": ae.result}, confidence=item.importance)
                updates["global_events"][key] = self.state.global_events[key]
                # NOTE: GlobalMemory.score is maintained by MemoryOrchestrator
                # via `points_delta` (see `_apply_scoring_event`). We no longer
                # overwrite it from raw ``score_*_after`` here — those fields
                # are frequently hallucinated from wide-angle frames with no
                # scoreboard visible.
        self.state.last_updated_ts = int(time.time())
        return updates


class MemoryBank:
    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k
        self.stream_manager = StreamMemoryManager()
        self.global_manager = GlobalMemoryManager()
        self._entity_index: Dict[str, str] = {}
        self._memory_to_entity: Dict[str, str] = {}
        self._recent_fingerprint: Dict[str, str] = {}
        self._effect_entity_by_stream: Dict[str, str] = {}

    @staticmethod
    def _fingerprint(item: MemoryItem) -> str:
        gs = (item.metadata or {}).get("global_state", {})
        ae = item.atomic_event
        return json.dumps(
            {
                "stream_id": item.stream_id,
                "event_type": ae.event_type if ae else None,
                "sub_type": ae.sub_type if ae else None,
                "result": ae.result if ae else None,
                "program_stream": gs.get("program_stream"),
                "last_director_action": gs.get("last_director_action"),
                "score_home_after": ae.score_home_after if ae else None,
                "score_away_after": ae.score_away_after if ae else None,
                "period": ae.period if ae else None,
                "clock": ae.clock if ae else None,
                "primary_player_id": ae.primary_player_id if ae else None,
                "effect_state": gs.get("effect_state"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _entity_key(item: MemoryItem) -> str:
        gs = (item.metadata or {}).get("global_state", {})
        if gs.get("program_stream") is not None:
            return "entity::director_program"
        if gs.get("effect_state") is not None:
            return f"entity::effect::{item.stream_id}"
        ae = item.atomic_event
        if ae and ae.event_id:
            return f"event::{ae.event_id}"
        if ae and (ae.game_id or ae.event_type or ae.clock):
            return f"event::{ae.game_id or 'na'}::{ae.event_type or 'na'}::{ae.clock or 'na'}::{ae.primary_player_id or 'na'}"
        return f"entity::stream_note::{item.stream_id}"

    def _resolve_operation(self, item: MemoryItem) -> tuple[MemoryOperation, Optional[str], str]:
        fp = self._fingerprint(item)
        last_fp = self._recent_fingerprint.get(item.stream_id)
        if last_fp == fp:
            return MemoryOperation.NOOP, None, "duplicate observation, noop"
        ent_key = self._entity_key(item)
        existing_id = self._entity_index.get(ent_key)
        gs = (item.metadata or {}).get("global_state", {})
        if gs.get("effect_state") == "end_effect":
            eff_key = self._effect_entity_by_stream.get(item.stream_id)
            if eff_key and eff_key in self._entity_index:
                return MemoryOperation.DELETE, self._entity_index[eff_key], "effect ended, delete effect entity"
            return MemoryOperation.NOOP, None, "end_effect without active effect, noop"
        if existing_id:
            return MemoryOperation.UPDATE, existing_id, "entity exists, update"
        return MemoryOperation.ADD, None, "new entity, add"

    def _update_indices(self, op: MemoryOperation, item: MemoryItem, memory_id: str) -> None:
        ent_key = self._entity_key(item)
        self._recent_fingerprint[item.stream_id] = self._fingerprint(item)
        if op in {MemoryOperation.ADD, MemoryOperation.UPDATE}:
            self._entity_index[ent_key] = memory_id
            self._memory_to_entity[memory_id] = ent_key
            gs = (item.metadata or {}).get("global_state", {})
            if gs.get("effect_state") == "start_effect":
                self._effect_entity_by_stream[item.stream_id] = ent_key
        if op == MemoryOperation.DELETE:
            old_ent_key = self._memory_to_entity.pop(memory_id, None)
            if old_ent_key:
                self._entity_index.pop(old_ent_key, None)
                if self._effect_entity_by_stream.get(item.stream_id) == old_ent_key:
                    self._effect_entity_by_stream.pop(item.stream_id, None)

    def upsert_memory(self, item: MemoryItem) -> MemoryActionResult:
        op, target_memory_id, note = self._resolve_operation(item)
        if op == MemoryOperation.NOOP:
            return MemoryActionResult(operation=MemoryOperation.NOOP, stream_id=item.stream_id, notes=note)
        if op == MemoryOperation.DELETE and target_memory_id:
            res = self.stream_manager.delete_memory(memory_id=target_memory_id, stream_id=item.stream_id)
            self._update_indices(MemoryOperation.DELETE, item, target_memory_id)
            return MemoryActionResult(operation=MemoryOperation.DELETE, memory_id=target_memory_id, stream_id=item.stream_id, notes=note)
        if op == MemoryOperation.UPDATE and target_memory_id:
            res = self.stream_manager.update_memory(
                memory_id=target_memory_id,
                stream_id=item.stream_id,
                content=item.content,
                importance=item.importance,
                timestamp=item.timestamp,
                tracking_status=item.tracking_status,
                metadata=item.metadata,
                atomic_event=item.atomic_event,
            )
            if res.operation == MemoryOperation.UPDATE:
                refreshed = self.stream_manager.query_local(stream_id=item.stream_id, limit=200)
                target = [x for x in refreshed if x.memory_id == target_memory_id]
                fused = self.global_manager.fuse_to_global(target) if target else {}
                self._update_indices(MemoryOperation.UPDATE, item, target_memory_id)
                return MemoryActionResult(operation=MemoryOperation.UPDATE, memory_id=target_memory_id, stream_id=item.stream_id, notes=note, fused_global_updates=fused)
            return MemoryActionResult(operation=MemoryOperation.NOOP, stream_id=item.stream_id, notes="update target missing, noop")
        add_res = self.stream_manager.add_memory(item)
        fused = self.global_manager.fuse_to_global([item])
        self._update_indices(MemoryOperation.ADD, item, item.memory_id)
        add_res.fused_global_updates = fused
        add_res.notes = note
        return add_res

    def select_memories(self, query: str = "", stream_ids: Optional[List[str]] = None, min_importance: float = 0.0, top_k: Optional[int] = None) -> List[MemoryItem]:
        return self.stream_manager.select_memories(query=query, stream_ids=stream_ids, min_importance=min_importance, top_k=top_k or self.top_k)

    def query_local(self, stream_id: Optional[str] = None, min_importance: float = 0.0, limit: Optional[int] = None) -> List[MemoryItem]:
        return self.stream_manager.query_local(stream_id=stream_id, min_importance=min_importance, limit=limit)


class MemoryOrchestrator:
    """Manages local + global memory and maintains narrative state.

    Narrative state lives on ``self.global_memory``:
      - ``period`` / ``clock``: last-known game clock (auto-advanced each round).
      - ``score``: dict keyed by team_id (defaults: "home" / "away").
      - ``possession_team``: last team observed attacking / controlling.
      - ``last_scoring_event`` / ``scoring_timeline``: scoring event ledger.
      - ``tick_count``: monotonic counter so the clock can auto-advance even
        when the model omits clock for this second.

    Design principle: memory must "keep moving" with the live broadcast. The
    clock never freezes; scores only change when a grounded ``scoring_event``
    arrives (with ``points_delta``); possession flips on made baskets /
    turnovers / steals.
    """

    def __init__(self, call_llm: Callable[[str, Optional[Type[T]]], Union[str, T]], top_k: int = 6) -> None:
        self.call_llm = call_llm
        self.top_k = top_k
        self.memory_bank = MemoryBank(top_k=top_k)
        self.global_memory = self.memory_bank.global_manager.state
        self._default_game_id = "game_auto"

    @staticmethod
    def _infer_stream_id(text: str) -> str:
        m = re.search(r"stream[_\s]*(\d+)", text.lower())
        return f"stream_{m.group(1)}" if m else "global"

    @staticmethod
    def _extract_score_pair(text: str) -> Optional[tuple[int, int]]:
        """Pull a (home, away) score pair from text only when clearly score-like.

        We avoid clock formats like "Q3 03:42" or "02:11" by requiring the pair
        to be either explicitly tagged (HOME/AWAY/SCORE) or be the only ``-``
        separated number pair in the string and not look like a clock.
        """
        if not isinstance(text, str) or not text:
            return None
        # 显式 score 标签：HOME 76-74, AWAY 74, score 76:74
        tagged = re.search(
            r"(?i)(?:home|away|score)[^\d]{0,8}(\d{1,3})\s*[-:]\s*(\d{1,3})",
            text,
        )
        if tagged:
            return int(tagged.group(1)), int(tagged.group(2))
        # 仅当字符串中只有一对 N-N 且不是时钟（mm:ss）时才采用。
        dash_pairs = re.findall(r"(?<!\d)(\d{1,3})\s*-\s*(\d{1,3})(?!\d)", text)
        if len(dash_pairs) == 1:
            a, b = int(dash_pairs[0][0]), int(dash_pairs[0][1])
            return a, b
        return None

    def _build_atomic_event(self, content: str, timestamp: str, parsed_json: Optional[Dict[str, Any]] = None) -> AtomicEvent:
        parsed_json = parsed_json or {}
        lowered = content.lower()
        event_type = None
        sub_type = None
        result = None
        # 泛化事件类型识别：体育/电竞共享层，不绑定单一项目。
        if any(k in lowered for k in ("round_start", "round start", "freeze_end", "side_switch", "match_end")):
            event_type = "round_state"
        elif any(k in lowered for k in ("player_kill", "kill", "bomb_planted", "bomb_defused", "utility_use")):
            event_type = "combat_event"
        elif any(k in lowered for k in ("score", "basket", "three-pointer", "layup", "goal", "point")):
            event_type = "scoring_event"
        elif any(k in lowered for k in ("switch", "continue", "start_effect", "end_effect", "layout_change")):
            event_type = "director_action"

        model_ae: Dict[str, Any] = {}
        if parsed_json:
            event_type = parsed_json.get("action") or event_type
            effect = parsed_json.get("effect") or {}
            if isinstance(effect, dict):
                sub_type = effect.get("type")
            result = "ok"
            raw_ae = parsed_json.get("atomic_event")
            if isinstance(raw_ae, dict):
                model_ae = {k: v for k, v in raw_ae.items() if v not in (None, "", [], {})}

        score_pair = self._extract_score_pair(content)
        score_home = model_ae.get("score_home_after") if "score_home_after" in model_ae else (score_pair[0] if score_pair else None)
        score_away = model_ae.get("score_away_after") if "score_away_after" in model_ae else (score_pair[1] if score_pair else None)

        ae_event_type = model_ae.get("event_type") or event_type or "observation"
        ae_sub_type = model_ae.get("sub_type") or sub_type
        ae_period = model_ae.get("period")
        ae_clock = model_ae.get("clock") or timestamp
        ae_team = model_ae.get("team_id")
        ae_primary = model_ae.get("primary_player_id")
        ae_secondary = model_ae.get("secondary_player_id")
        ae_result = model_ae.get("result") or result
        ae_points_delta = model_ae.get("points_delta")
        ae_lineup_home = model_ae.get("lineup_home") if isinstance(model_ae.get("lineup_home"), list) else None
        ae_lineup_away = model_ae.get("lineup_away") if isinstance(model_ae.get("lineup_away"), list) else None
        # 让模型提供的 event_id 接管去重锚点；否则 fallback 自动生成。
        provided_event_id = model_ae.get("event_id")
        if provided_event_id:
            ae_event_id = str(provided_event_id)
        else:
            ae_event_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        return AtomicEvent(
            game_id=str(model_ae.get("game_id") or self._default_game_id),
            event_id=ae_event_id,
            period=str(ae_period) if ae_period is not None else None,
            clock=str(ae_clock) if ae_clock is not None else None,
            team_id=str(ae_team) if ae_team is not None else None,
            primary_player_id=str(ae_primary) if ae_primary is not None else None,
            event_type=ae_event_type,
            sub_type=str(ae_sub_type) if ae_sub_type is not None else None,
            result=str(ae_result) if ae_result is not None else None,
            points_delta=int(ae_points_delta) if isinstance(ae_points_delta, (int, float)) else None,
            secondary_player_id=str(ae_secondary) if ae_secondary is not None else None,
            lineup_home=ae_lineup_home,
            lineup_away=ae_lineup_away,
            score_home_after=int(score_home) if isinstance(score_home, (int, float)) else None,
            score_away_after=int(score_away) if isinstance(score_away, (int, float)) else None,
            extra={k: v for k, v in (model_ae.get("extra") or {}).items()} if isinstance(model_ae.get("extra"), dict) else {},
        )

    def _extract_metadata(self, content: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(_extract_json_block(content))
        except Exception:  # noqa: BLE001
            parsed = {}
        if parsed:
            action = str(parsed.get("action", "")).strip()
            stream_id = str(parsed.get("stream_id", "")).strip()
            if action or stream_id:
                metadata["global_state"] = {"last_director_action": action, "program_stream": stream_id}
            if action in {"start_effect", "end_effect"}:
                metadata["global_state"] = {**metadata.get("global_state", {}), "effect_state": action}
            metadata["decision"] = parsed
            reason = str(parsed.get("reason", "")).strip()
            if reason:
                metadata["commentary"] = {"latest_reason": reason}
            overlay_text = parsed.get("overlay_text")
            if isinstance(overlay_text, str) and overlay_text.strip():
                metadata["commentary"] = {
                    **metadata.get("commentary", {}),
                    "overlay_text": overlay_text.strip(),
                }
                # 仅在 overlay_text 中出现明确 score 标签时，才把比分纳入 global state。
                ot_score = self._extract_score_pair(overlay_text)
                if ot_score:
                    metadata["global_state"] = {
                        **metadata.get("global_state", {}),
                        "score_home_after": ot_score[0],
                        "score_away_after": ot_score[1],
                    }
            ae = parsed.get("atomic_event")
            if isinstance(ae, dict):
                ae_clean = {k: v for k, v in ae.items() if v not in (None, "", [], {})}
                if ae_clean:
                    metadata["atomic_event_raw"] = ae_clean
                gs_patch: Dict[str, Any] = {}
                if isinstance(ae_clean.get("score_home_after"), (int, float)):
                    gs_patch["score_home_after"] = int(ae_clean["score_home_after"])
                if isinstance(ae_clean.get("score_away_after"), (int, float)):
                    gs_patch["score_away_after"] = int(ae_clean["score_away_after"])
                if isinstance(ae_clean.get("period"), str):
                    gs_patch["period"] = ae_clean["period"]
                if isinstance(ae_clean.get("clock"), str):
                    gs_patch["clock"] = ae_clean["clock"]
                if gs_patch:
                    metadata["global_state"] = {**metadata.get("global_state", {}), **gs_patch}
                primary = ae_clean.get("primary_player_id")
                points_delta = ae_clean.get("points_delta")
                if isinstance(primary, str) and primary and isinstance(points_delta, (int, float)):
                    metadata["player_stats"] = {primary: {"points": float(points_delta)}}
        return metadata, parsed

    def _upsert(self, stream_id: str, content: str, importance: float, timestamp: str, tracking_status: TrackingStatus, metadata: Optional[Dict[str, Any]] = None, atomic_event: Optional[AtomicEvent] = None) -> MemoryActionResult:
        item = MemoryItem(
            stream_id=stream_id,
            content=content,
            importance=importance,
            timestamp=timestamp,
            tracking_status=tracking_status,
            metadata=metadata or {},
            atomic_event=atomic_event,
        )
        return self.memory_bank.upsert_memory(item)

    _CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

    @staticmethod
    def _parse_clock(clock: Optional[str]) -> Optional[tuple[int, int]]:
        if not isinstance(clock, str):
            return None
        m = MemoryOrchestrator._CLOCK_RE.match(clock.strip())
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _format_clock(mm: int, ss: int) -> str:
        mm = max(0, mm)
        ss = max(0, min(59, ss))
        return f"{mm:02d}:{ss:02d}"

    def _auto_tick_clock(self, observed_clock: Optional[str]) -> None:
        """Advance narrative clock by 1 second per round.

        Most live sports (basketball/hockey/football) run a DESCENDING clock,
        so on each per-second round we tick the stored clock down by 1s. We
        only RESYNC to ``observed_clock`` when the model's reading differs
        from our auto-ticked expectation by more than 2 seconds — assuming
        anything closer is the model lazily echoing memory rather than
        actually re-reading the scoreboard. Non-MM:SS strings (e.g. esports
        round timers) are stored verbatim without auto-ticking.
        """
        gm = self.global_memory
        gm.tick_count += 1
        current = self._parse_clock(gm.clock)
        observed = self._parse_clock(observed_clock)

        if current is None:
            # No baseline yet — accept the very first reading verbatim.
            if observed is not None:
                gm.clock = self._format_clock(*observed)
            elif isinstance(observed_clock, str) and observed_clock.strip():
                gm.clock = observed_clock.strip()
            return

        # Auto-tick down by 1 second.
        mm, ss = current
        total = max(0, mm * 60 + ss - 1)
        ticked_mm, ticked_ss = total // 60, total % 60
        ticked = self._format_clock(ticked_mm, ticked_ss)

        if observed is None:
            # Model didn't supply a clock — trust auto-tick.
            gm.clock = ticked
            return

        # Resync only on >5s drift (likely a real new scoreboard read).
        # We trust auto-tick over the model's per-second readout because the
        # model frequently echoes a stale clock value lifted from memory; a
        # 5s tolerance covers normal lag/mis-reads while still catching
        # period changes or actual scoreboard jumps.
        observed_total = observed[0] * 60 + observed[1]
        drift = abs(observed_total - total)
        if drift > 5:
            gm.clock = self._format_clock(*observed)
        else:
            gm.clock = ticked

    def _apply_scoring_event(self, atomic: AtomicEvent) -> None:
        """Update score via ``points_delta`` when a valid scoring event is seen.

        We trust ``points_delta`` more than raw ``score_*_after`` because the
        latter is frequently hallucinated from a wide-angle frame with no
        scoreboard. Delta updates compose cleanly with memory history.
        """
        if atomic is None:
            return
        gm = self.global_memory
        team = (atomic.team_id or "").strip()
        if atomic.event_type == "scoring_event" and atomic.result == "made":
            delta = atomic.points_delta
            if isinstance(delta, int) and delta > 0 and team:
                prior = int(gm.score.get(team, 0))
                gm.score[team] = prior + delta
                record = {
                    "event_id": atomic.event_id,
                    "team_id": team,
                    "player_id": atomic.primary_player_id,
                    "points_delta": delta,
                    "period": atomic.period or gm.period,
                    "clock": atomic.clock or gm.clock,
                    "score_after": dict(gm.score),
                    "sub_type": atomic.sub_type,
                }
                gm.last_scoring_event = record
                gm.scoring_timeline.append(record)
                if len(gm.scoring_timeline) > 32:
                    gm.scoring_timeline = gm.scoring_timeline[-32:]
                # scorer retains possession only for inbound — flip it to let
                # director anticipate the next attack.
                gm.possession_team = "away" if team == "home" else ("home" if team == "away" else team)

    def _apply_possession_hints(self, atomic: AtomicEvent) -> None:
        if atomic is None:
            return
        gm = self.global_memory
        team = (atomic.team_id or "").strip()
        et = atomic.event_type or ""
        sub = atomic.sub_type or ""
        if et == "turnover" and team:
            gm.possession_team = "away" if team == "home" else ("home" if team == "away" else team)
        elif sub == "steal" and team:
            gm.possession_team = team
        elif et == "rebound" and team:
            gm.possession_team = team
        elif not gm.possession_team and team:
            gm.possession_team = team

    def _update_narrative_state(self, atomic: AtomicEvent) -> None:
        gm = self.global_memory
        if atomic is not None:
            if atomic.period:
                gm.period = atomic.period
            self._auto_tick_clock(atomic.clock)
            self._apply_scoring_event(atomic)
            self._apply_possession_hints(atomic)
        else:
            self._auto_tick_clock(None)

    def add_event(self, timestamp: str, event_description: str, importance_score: float = 0.5) -> MemoryActionResult:
        stream_id = self._infer_stream_id(event_description)
        atomic_event = self._build_atomic_event(event_description, timestamp, None)
        return self._upsert(stream_id=stream_id, content=event_description, importance=importance_score, timestamp=timestamp, tracking_status=TrackingStatus.UNKNOWN, atomic_event=atomic_event)

    def add_model_output(self, timestamp: str, model_output: str, importance_score: float = 0.8) -> MemoryActionResult:
        metadata, parsed = self._extract_metadata(model_output)
        # 模型决策是跨流节目状态，统一放到 global，便于稳定 UPDATE。
        stream_id = "global"
        atomic_event = self._build_atomic_event(model_output, timestamp, parsed)
        self._update_narrative_state(atomic_event)
        return self._upsert(stream_id=stream_id, content=model_output, importance=importance_score, timestamp=timestamp, tracking_status=TrackingStatus.ACTIVE, metadata=metadata, atomic_event=atomic_event)

    def narrative_state(self) -> Dict[str, Any]:
        gm = self.global_memory
        recent_scores = gm.scoring_timeline[-4:] if gm.scoring_timeline else []
        return {
            "period": gm.period,
            "clock": gm.clock,
            "tick_count": gm.tick_count,
            "score": dict(gm.score),
            "possession_team": gm.possession_team,
            "last_scoring_event": gm.last_scoring_event,
            "recent_scoring_events": recent_scores,
        }

    def query_local_events(self) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for item in self.memory_bank.query_local():
            events.append(
                StreamEvent(
                    timestamp=item.timestamp or "",
                    event_description=item.content,
                    importance_score=item.importance,
                    stream_id=item.stream_id,
                    memory_id=item.memory_id,
                    tracking_status=item.tracking_status.value,
                    metadata=item.metadata,
                    atomic_event=item.atomic_event,
                )
            )
        return events

    def retrieve_topk_events(self, query: str, top_k: Optional[int] = None) -> List[StreamEvent]:
        candidates = self.query_local_events()
        query_terms = {x for x in re.split(r"\W+", query.lower()) if x}

        def _score(idx_event: tuple[int, StreamEvent]) -> float:
            idx, ev = idx_event
            event_terms = {x for x in re.split(r"\W+", ev.event_description.lower()) if x}
            overlap = len(query_terms & event_terms)
            return overlap * 2.0 + ev.importance_score + (idx + 1) / max(len(candidates), 1)

        ranked = sorted(enumerate(candidates), key=_score, reverse=True)
        return [x[1] for x in ranked[: (top_k or self.top_k)]]

    def distill_events(self, query: str, candidates: List[StreamEvent]) -> List[StreamEvent]:
        if not candidates:
            return []
        query_terms = {x for x in re.split(r"\W+", query.lower()) if x}
        scored: List[tuple[float, StreamEvent]] = []
        for event in candidates:
            terms = {x for x in re.split(r"\W+", event.event_description.lower()) if x}
            overlap = len(query_terms & terms)
            scored.append((overlap * 2.0 + event.importance_score, event))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: min(6, len(scored))]]

    def answer(self, query: str, top_k: Optional[int] = None) -> str:
        distilled = self.distill_events(query, self.retrieve_topk_events(query, top_k=top_k))
        payload = {
            "query": query,
            "global_memory": self.global_memory.model_dump(),
            "distilled_events": [e.model_dump() for e in distilled],
        }
        return json.dumps(payload, ensure_ascii=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "global_memory": self.global_memory.model_dump(),
            "stream_memory": self.memory_bank.stream_manager.to_dict(),
            "updated_at": int(time.time()),
        }
