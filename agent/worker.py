"""Background cognition loop (thread target) for the Hybrid ToM agent.

HARD RULE: This worker must NOT do any USD writes. It only updates Python memory.

Input queue items (dict):
  {
    "frames": [np_rgb_uint8, ...],
    "user_text": str,
    "trigger": {"type": "gt_change"|"user_input"|"periodic", ...},
    "gt_state": {"ground_truth_objects": {...}, "timestamp": float},
    "ts": float,
    "run_id": int,
  }

Output queue items:
  {"status":"Done", "reply":str, "action":dict, "belief_update":dict,
   "ltm_snippets":[...], "gt_state":dict, "trigger":dict, "ts":float}
  {"status":"Error", "error":"...", "ts":float}
"""
from __future__ import annotations

import queue
import time
from typing import Any, Dict, Iterable, Optional

from .graph import AgentGraph


def cognitive_worker(
    belief_manager,
    short_memory,
    long_memory,
    input_queue,
    output_queue,
) -> None:
    """Background cognition loop (thread target)."""

    agent = AgentGraph(
        short_memory=short_memory,
        long_memory=long_memory,
        logger=lambda *_: None,
    )

    while True:
        item = input_queue.get()  # blocking
        try:
            if item is None:
                return

            if isinstance(item, tuple) and len(item) == 2:
                frames, trigger = item
                item = {"frames": frames, "trigger": trigger, "user_text": "", "ts": time.time()}

            if not isinstance(item, dict):
                raise RuntimeError(f"Bad worker input type: {type(item)}")

            frames = item.get("frames") or []
            if isinstance(frames, Iterable) and not isinstance(frames, (bytes, str)):
                frames = list(frames)
            else:
                frames = []

            state = {
                "frames": frames,
                "user_text": str(item.get("user_text") or ""),
                "trigger": item.get("trigger") or {},
                "gt_state": item.get("gt_state") or {},
                "ts": float(item.get("ts") or time.time()),
                # Inject belief snapshot as AgentState single source of truth (master plan §B).
                # This makes AgentState the authoritative belief carrier through the graph.
                "belief_state": belief_manager.get_snapshot() if belief_manager is not None else {},
            }
            run_id = int(item.get("run_id", -1))
            t0 = time.time()
            trigger = state.get("trigger") if isinstance(state.get("trigger"), dict) else {}
            print(
                f"[WORKER] start run_id={run_id} trigger={trigger.get('type')} frames={len(state.get('frames') or [])}",
                flush=True,
            )

            out = agent.run(state)

            # Update short-term belief stores (thread-safe).
            if isinstance(out.belief_update, dict) and out.belief_update:
                belief_manager.update_belief(out.belief_update)
                snap = belief_manager.get_snapshot()
                if isinstance(snap, dict):
                    short_memory.update_objects_from_belief(snap, now=state["ts"])

            # Record dialogue for explicit user turns.
            if out.reply and str(state.get("user_text") or "").strip():
                short_memory.set_dialogue(user_text=state["user_text"], model_reply=out.reply)

            # Reflect & store to LTM (GT events + user dialogue).
            # Re-inject model_output so reflect_store can access it.
            state["model_output"] = out.raw
            agent._reflect_store(state)

            output_queue.put(
                {
                    "status": "Done",
                    "run_id": run_id,
                    "reply": out.reply,
                    "action": out.action,
                    "belief_update": out.belief_update,
                    "stm_observation": out.stm_observation,
                    "ltm_snippets": out.ltm_snippets,
                    "gt_state": out.gt_state,
                    "trigger": out.trigger,
                    "raw": out.raw,
                    "ts": time.time(),
                }
            )
            print(
                f"[WORKER] done run_id={run_id} in {time.time()-t0:.2f}s | reply_len={len(out.reply or '')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WORKER] error run_id={int(item.get('run_id', -1)) if isinstance(item, dict) else -1}: {exc}", flush=True)
            output_queue.put(
                {
                    "status": "Error",
                    "run_id": int(item.get("run_id", -1)) if isinstance(item, dict) else -1,
                    "error": str(exc),
                    "ts": time.time(),
                }
            )
        finally:
            if hasattr(input_queue, "task_done"):
                try:
                    input_queue.task_done()
                except Exception:
                    pass
