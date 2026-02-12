from __future__ import annotations

import copy
import json
import re
import threading
from typing import Any, Dict


class BeliefManager:
    """Thread-safe belief store for robot memory."""

    def __init__(self, initial_state: Dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = copy.deepcopy(initial_state) if initial_state else {}

    def update_belief(self, new_data: Dict[str, Any]) -> None:
        """Merge VLM output into master state with stale-on-unknown logic."""
        with self._lock:
            if not isinstance(new_data, dict):
                return
            new_data = self._normalize_update(new_data)
            merged = self._merge_state(self._state, new_data)
            self._state = self._canonicalize_state(merged)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of current state for safe reading."""
        with self._lock:
            # Defensive canonicalization on read so UI/logs never show split object keys
            # (e.g. both state["objects"]["orange"] and state["orange"]).
            return self._canonicalize_state(copy.deepcopy(self._state))

    def reset(self, initial_state: Dict[str, Any] | None = None) -> None:
        """Reset the full belief store (thread-safe)."""
        with self._lock:
            self._state = copy.deepcopy(initial_state) if initial_state is not None else {"objects": {}}

    @staticmethod
    def _coerce_object_like(value: Any) -> Dict[str, Any] | None:
        """Best-effort parse for object entries from dict or malformed JSON-ish strings."""
        if isinstance(value, dict):
            return copy.deepcopy(value)
        if not isinstance(value, str):
            return None

        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed

        # Loose parse for partially malformed model strings.
        status = re.search(r'"?belief_status"?\s*:\s*"([^"]+)"', text)
        temporal = re.search(r'"?temporal_change"?\s*:\s*"([^"]+)"', text)
        container = re.search(r'"?inferred_container"?\s*:\s*"([^"]*)"', text)
        confidence = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', text)
        visible = re.search(r'"?visible"?\s*:\s*(true|false)', text, re.IGNORECASE)
        stale = re.search(r'"?stale"?\s*:\s*(true|false)', text, re.IGNORECASE)
        if not any((status, temporal, container, confidence, visible, stale)):
            return None

        out: Dict[str, Any] = {}
        if status:
            out["belief_status"] = status.group(1).strip().lower()
        if temporal:
            out["temporal_change"] = temporal.group(1).strip()
        if container:
            out["inferred_container"] = container.group(1).strip()
        if confidence:
            try:
                out["confidence"] = float(confidence.group(1))
            except Exception:
                pass
        if visible:
            out["visible"] = visible.group(1).lower() == "true"
        if stale:
            out["stale"] = stale.group(1).lower() == "true"
        return out if out else None

    def _merge_state(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, val in incoming.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_object_dict(merged[key], val)
            else:
                merged[key] = copy.deepcopy(val)
        return merged

    def _normalize_update(self, incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Canonicalize belief updates to {"objects": {...}} shape."""
        out = copy.deepcopy(incoming)
        objects = out.get("objects")
        if isinstance(objects, str):
            parsed_objects = self._coerce_object_like(objects)
            if isinstance(parsed_objects, dict):
                objects = parsed_objects
        normalized_objects: Dict[str, Any] = {}
        lifted_keys = []
        if isinstance(objects, dict):
            # Normalize object payloads in the canonical block.
            for k, v in list(objects.items()):
                parsed = self._coerce_object_like(v)
                if parsed is not None:
                    normalized_objects[str(k)] = parsed
                # Some model outputs leak full object dictionaries as a key string.
                key_text = str(k).strip()
                if key_text.startswith("{") and key_text.endswith("}"):
                    key_obj = self._coerce_object_like(key_text)
                    if isinstance(key_obj, dict):
                        for kk, vv in key_obj.items():
                            vv_parsed = self._coerce_object_like(vv)
                            if vv_parsed is not None:
                                normalized_objects[str(kk)] = vv_parsed

        for key, value in list(out.items()):
            if key in {"objects", "scene", "meta", "task"}:
                continue
            parsed = self._coerce_object_like(value)
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
                normalized_objects[key] = parsed
                lifted_keys.append(key)

        out["objects"] = normalized_objects
        # Remove object-like keys lifted to the canonical objects block.
        for key in lifted_keys:
            out.pop(key, None)
        return out

    def _canonicalize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure state keeps object tracks only under state['objects']."""
        out = copy.deepcopy(state)
        objects = out.get("objects")
        if isinstance(objects, str):
            parsed_objects = self._coerce_object_like(objects)
            if isinstance(parsed_objects, dict):
                objects = parsed_objects
        if not isinstance(objects, dict):
            objects = {}
        else:
            for k, v in list(objects.items()):
                parsed = self._coerce_object_like(v)
                if parsed is not None:
                    objects[str(k)] = parsed
                key_text = str(k).strip()
                if key_text.startswith("{") and key_text.endswith("}"):
                    key_obj = self._coerce_object_like(key_text)
                    if isinstance(key_obj, dict):
                        for kk, vv in key_obj.items():
                            vv_parsed = self._coerce_object_like(vv)
                            if vv_parsed is not None:
                                objects[str(kk)] = vv_parsed

        lifted = []
        for key, value in out.items():
            if key in {"objects", "scene", "meta", "task"}:
                continue
            parsed = self._coerce_object_like(value)
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
                existing = objects.get(key)
                if isinstance(existing, dict):
                    merged = copy.deepcopy(existing)
                    merged.update(parsed)
                    objects[key] = merged
                else:
                    objects[key] = parsed
                lifted.append(key)

        out["objects"] = objects
        for key in lifted:
            out.pop(key, None)
        return out

    def _merge_object_dict(self, base_obj: Dict[str, Any], incoming_obj: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base_obj)
        for k, v in incoming_obj.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = self._merge_object_dict(merged[k], v)
            else:
                # Special-case "unknown" handling: preserve last known value and mark stale.
                if k in ("location", "belief_status") and v == "unknown":
                    prev = merged.get(k)
                    if prev is not None and prev != "unknown":
                        merged["stale"] = True
                        continue
                merged[k] = copy.deepcopy(v)
        return merged
