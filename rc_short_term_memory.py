from __future__ import annotations

import copy
import json
import re
import threading
import time
from typing import Any, Dict, Optional


class ShortTermMemory:
    """Short-term memory for the agent (thread-safe, JSON-serializable).

    This is *not* ground-truth. It stores the agent's belief state and dialogue/task context.
    """

    def __init__(self, *, ttl_sec: float = 5.0, logger=print) -> None:
        self._log = logger
        self._lock = threading.Lock()
        self._ttl = float(ttl_sec)
        self._state: Dict[str, Any] = {
            "objects": {},
            "task": {"phase": "idle", "goal_stack": [], "last_action": None, "action_status": "idle"},
            "dialogue": {"last_user_utterance": "", "last_model_reply": "", "rolling_summary": ""},
            "meta": {"updated_ts": time.time()},
        }

    def update_objects_from_belief(self, belief_update: Dict[str, Any], *, now: Optional[float] = None) -> None:
        """Merge model belief output into memory with stale-on-unknown semantics."""
        if not isinstance(belief_update, dict):
            return
        now = float(now if now is not None else time.time())

        with self._lock:
            incoming_objects = self._extract_objects_block(belief_update)
            if not incoming_objects:
                return

            for name, incoming in incoming_objects.items():
                if not isinstance(incoming, dict):
                    continue
                cur = self._state["objects"].get(name, {})
                if not isinstance(cur, dict):
                    cur = {}

                merged = copy.deepcopy(cur)
                for k, v in incoming.items():
                    # If incoming says unknown, keep last known but mark stale.
                    if k in ("belief_status", "location") and v == "unknown":
                        prev = merged.get(k)
                        if prev is not None and prev != "unknown":
                            merged["stale"] = True
                            continue
                    merged[k] = copy.deepcopy(v)

                # Update last_seen/confidence if it is visible/contained.
                if merged.get("belief_status") in ("visible", "contained", "moving"):
                    merged["last_seen_time"] = now
                    merged["stale"] = bool(merged.get("stale", False))
                    # If confidence is missing, set a reasonable default.
                    if "confidence" not in merged:
                        merged["confidence"] = 0.7
                else:
                    merged.setdefault("stale", True)
                    merged.setdefault("confidence", 0.5)

                self._state["objects"][name] = merged

            self._state["meta"]["updated_ts"] = now

    def _extract_objects_block(self, belief_update: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        def _coerce(v: Any) -> Dict[str, Any] | None:
            if isinstance(v, dict):
                return copy.deepcopy(v)
            if isinstance(v, str):
                t = v.strip()
                if t.startswith("{") and t.endswith("}"):
                    try:
                        p = json.loads(t)
                    except Exception:
                        p = None
                    if isinstance(p, dict):
                        return p
                # Loose parse for malformed inline JSON snippets.
                status = re.search(r'"?belief_status"?\s*:\s*"([^"]+)"', t)
                temporal = re.search(r'"?temporal_change"?\s*:\s*"([^"]+)"', t)
                container = re.search(r'"?inferred_container"?\s*:\s*"([^"]*)"', t)
                confidence = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', t)
                visible = re.search(r'"?visible"?\s*:\s*(true|false)', t, re.IGNORECASE)
                stale = re.search(r'"?stale"?\s*:\s*(true|false)', t, re.IGNORECASE)
                if status or temporal or container or confidence or visible or stale:
                    parsed: Dict[str, Any] = {}
                    if status:
                        parsed["belief_status"] = status.group(1).strip().lower()
                    if temporal:
                        parsed["temporal_change"] = temporal.group(1).strip()
                    if container:
                        parsed["inferred_container"] = container.group(1).strip()
                    if confidence:
                        try:
                            parsed["confidence"] = float(confidence.group(1))
                        except Exception:
                            pass
                    if visible:
                        parsed["visible"] = visible.group(1).lower() == "true"
                    if stale:
                        parsed["stale"] = stale.group(1).lower() == "true"
                    return parsed if parsed else None
            return None

        incoming_objects: Dict[str, Dict[str, Any]] = {}
        objects = belief_update.get("objects")
        if isinstance(objects, dict):
            for name, payload in objects.items():
                parsed = _coerce(payload)
                if parsed is not None:
                    incoming_objects[str(name)] = parsed

        # Accept object-like entries directly under belief_update as a robust fallback.
        reserved = {"objects", "scene", "meta", "task"}
        for key, value in belief_update.items():
            if key in reserved:
                continue
            parsed = _coerce(value)
            if parsed is None:
                continue
            if any(
                k in parsed
                for k in (
                    "belief_status",
                    "visible",
                    "confidence",
                    "inferred_container",
                    "temporal_change",
                    "stale",
                    "location",
                )
            ):
                incoming_objects[str(key)] = parsed
        return incoming_objects

    def decay(self, *, now: Optional[float] = None) -> None:
        """Apply TTL-based decay to object confidences."""
        now = float(now if now is not None else time.time())
        with self._lock:
            for name, obj in list(self._state["objects"].items()):
                if not isinstance(obj, dict):
                    continue
                last_seen = obj.get("last_seen_time")
                if last_seen is None:
                    continue
                age = now - float(last_seen)
                if age <= self._ttl:
                    continue
                # Decay confidence and mark unknown when too old.
                conf = float(obj.get("confidence", 0.5))
                conf = max(0.0, conf * 0.8)
                obj["confidence"] = conf
                obj["stale"] = True
                if age >= 2.0 * self._ttl:
                    obj["belief_status"] = "unknown"
                self._state["objects"][name] = obj
            self._state["meta"]["updated_ts"] = now

    def set_dialogue(self, *, user_text: str, model_reply: str) -> None:
        with self._lock:
            self._state["dialogue"]["last_user_utterance"] = str(user_text)
            self._state["dialogue"]["last_model_reply"] = str(model_reply)
            # Cheap rolling summary: keep the last ~800 chars of alternating turns.
            prev = str(self._state["dialogue"].get("rolling_summary", ""))
            addition = f"USER: {user_text}\nASSISTANT: {model_reply}\n"
            combined = (prev + addition)[-800:]
            self._state["dialogue"]["rolling_summary"] = combined
            self._state["meta"]["updated_ts"] = time.time()

    def set_last_action(self, action: Dict[str, Any], status: str) -> None:
        with self._lock:
            self._state["task"]["last_action"] = copy.deepcopy(action)
            self._state["task"]["action_status"] = str(status)
            self._state["meta"]["updated_ts"] = time.time()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._state["task"]["phase"] = str(phase)
            self._state["meta"]["updated_ts"] = time.time()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def to_json(self) -> str:
        return json.dumps(self.get_snapshot(), ensure_ascii=True)

    def compact_summary(self) -> str:
        """Return a short text summary suitable for prompts."""
        snap = self.get_snapshot()
        objs = snap.get("objects", {})
        parts = []
        if isinstance(objs, dict) and objs:
            for name, obj in objs.items():
                if not isinstance(obj, dict):
                    continue
                status = obj.get("belief_status", "unknown")
                stale = obj.get("stale", False)
                conf = obj.get("confidence", None)
                container = obj.get("inferred_container", "")
                s = f"{name}: {status}"
                if container:
                    s += f" in {container}"
                if stale:
                    s += " (stale)"
                if conf is not None:
                    try:
                        s += f" conf={float(conf):.2f}"
                    except Exception:
                        pass
                parts.append(s)
        else:
            parts.append("no objects tracked")

        task = snap.get("task", {})
        if isinstance(task, dict):
            parts.append(f"phase={task.get('phase', 'idle')}")
            if task.get("action_status"):
                parts.append(f"action_status={task.get('action_status')}")
            last_action = task.get("last_action")
            if isinstance(last_action, dict) and last_action.get("type"):
                parts.append(f"last_action={last_action.get('type')}")

        dlg = snap.get("dialogue", {})
        if isinstance(dlg, dict):
            last_user = str(dlg.get("last_user_utterance") or "").strip()
            if last_user:
                parts.append(f"user=\"{last_user[:80]}\"")

        return " | ".join(parts)[:600]
