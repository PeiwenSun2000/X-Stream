#!/usr/bin/env python3
"""Runnable demo for two-level memory module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MEMORY_MODULE_DIR = CURRENT_DIR.parent / "mllmflow"
sys.path.insert(0, str(MEMORY_MODULE_DIR))

from memory_bank import MemoryOrchestrator  # type: ignore


def main() -> int:
    orchestrator = MemoryOrchestrator(call_llm=lambda p, r=None: "{}", top_k=6)
    r1 = orchestrator.add_event("Q3_10:11", "Stream 1: Home 76-74. Guard drives and dishes for corner three.")
    r2 = orchestrator.add_model_output("Q3_10:11", '{"action":"start_effect","stream_id":"stream_1","effect":{"type":"tracking"},"reason":"Priority 4"}')
    r3 = orchestrator.add_model_output("Q3_10:12", '{"action":"switch","stream_id":"stream_2","reason":"Priority 1"}')
    r4 = orchestrator.add_model_output("Q3_10:13", '{"action":"switch","stream_id":"stream_2","reason":"Priority 1"}')  # likely NOOP
    r5 = orchestrator.add_model_output("Q3_10:14", '{"action":"end_effect","stream_id":"stream_1","reason":"Priority 7"}')  # DELETE

    selected = orchestrator.memory_bank.select_memories(query="score tracking switch", top_k=5)
    local_stream_1 = orchestrator.memory_bank.query_local(stream_id="stream_1")
    global_state = orchestrator.global_memory.model_dump()

    print("=== Operation Results ===")
    for result in [r1, r2, r3, r4, r5]:
        print(result.model_dump_json())

    print("\n=== Selected Local Memories (top-k) ===")
    print(json.dumps([x.model_dump() for x in selected], ensure_ascii=False, indent=2))

    print("\n=== Stream-1 Local Memories ===")
    print(json.dumps([x.model_dump() for x in local_stream_1], ensure_ascii=False, indent=2))

    print("\n=== Global Memory ===")
    print(json.dumps(global_state, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
