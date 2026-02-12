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

from rc_agent_graph import AgentGraph
from rc_reason2_agent import reason2_decide


def cognitive_worker(
    belief_manager,
    short_memory,
    long_memory,
    input_queue,
    output_queue,
) -> None:
    """Background cognition loop (thread target)."""

    def _reason2_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        frames = state.get("frames") or []
        user_text = str(state.get("user_text") or "")
        stm_summary = str(state.get("stm_summary") or "")
        ltm_snips = state.get("ltm_snippets") or []
        if not isinstance(ltm_snips, list):
            ltm_snips = []
        prev_belief = belief_manager.get_snapshot() if belief_manager is not None else {}

        # Augment user_text with GT context if available.
        gt_context = str(state.get("gt_context") or "")
        augmented_text = user_text
        if gt_context:
            augmented_text = f"{user_text}\n{gt_context}" if user_text else gt_context

        try:
            return reason2_decide(
                frames=frames,
                user_text=augmented_text,
                short_memory_summary=stm_summary,
                long_memory_snippets=[str(s) for s in ltm_snips],
                prev_belief_json=prev_belief,
            )
        except Exception as exc:
            return {
                "reply": f"model_error: {exc}",
                "action": {"type": "noop", "args": {}},
                "belief_update": {},
                "meta": {"num_images": len(frames), "error": str(exc)},
            }

    agent = AgentGraph(
        short_memory=short_memory,
        long_memory=long_memory,
        reason2_fn=_reason2_fn,
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
            }
            run_id = int(item.get("run_id", -1))

            out = agent.run(state)

            # Update short-term belief stores (thread-safe).
            if isinstance(out.belief_update, dict) and out.belief_update:
                belief_manager.update_belief(out.belief_update)
                snap = belief_manager.get_snapshot()
                canonical_objects = {}
                if isinstance(snap, dict) and isinstance(snap.get("objects"), dict):
                    canonical_objects = snap.get("objects") or {}
                short_memory.update_objects_from_belief({"objects": canonical_objects}, now=state["ts"])

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
                    "ltm_snippets": out.ltm_snippets,
                    "gt_state": out.gt_state,
                    "trigger": out.trigger,
                    "raw": out.raw,
                    "ts": time.time(),
                }
            )
        except Exception as exc:
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
