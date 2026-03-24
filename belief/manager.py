from __future__ import annotations

import copy
import json
import re
import threading
from typing import Any, Dict

_MOTION_WORDS = (
    "move",
    "moving",
    "motion",
    "fall",
    "falling",
    "drop",
    "dropping",
    "roll",
    "rolling",
    "slide",
    "sliding",
    "swing",
    "rotate",
    "spin",
    "tilt",
    "shift",
    "displace",
    "changed",
)


class BeliefManager:
    """Thread-safe belief store for robot memory."""

    def __init__(self, initial_state: Dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        seed = copy.deepcopy(initial_state) if isinstance(initial_state, dict) else {}
        if "objects" not in seed:
            seed["objects"] = {}
        if "static_context" not in seed:
            seed["static_context"] = {}
        self._state: Dict[str, Any] = self._canonicalize_state(seed)

    def update_belief(self, new_data: Dict[str, Any]) -> None:
        """Merge VLM output into master state with stale-on-unknown logic."""
        with self._lock:
            if not isinstance(new_data, dict):
                return
            base = self._canonicalize_state(copy.deepcopy(self._state))
            incoming = self._normalize_update(new_data)

            base_objects = base.get("objects", {})
            base_static = base.get("static_context", {})
            if not isinstance(base_objects, dict):
                base_objects = {}
            if not isinstance(base_static, dict):
                base_static = {}

            inc_objects = incoming.get("objects", {})
            inc_static = incoming.get("static_context", {})
            if not isinstance(inc_objects, dict):
                inc_objects = {}
            if not isinstance(inc_static, dict):
                inc_static = {}

            for name, payload in inc_objects.items():
                if not isinstance(payload, dict):
                    continue
                merged_obj = self._merge_object_dict(base_objects.get(name, {}), payload)
                base_objects[name] = merged_obj
                if self._is_dynamic_entry(merged_obj):
                    # Wake-up promotion: a previously static/background entity is now dynamic.
                    base_static.pop(name, None)

            for name, payload in inc_static.items():
                if not isinstance(payload, dict):
                    continue
                payload_is_dynamic = self._is_dynamic_entry(payload)
                normalized_static = self._normalize_static_entry(payload)
                if payload_is_dynamic or self._is_dynamic_entry(normalized_static):
                    # Wake-up promotion from static_context -> active objects.
                    merged_obj = self._merge_object_dict(base_objects.get(name, {}), normalized_static)
                    base_objects[name] = merged_obj
                    base_static.pop(name, None)
                    continue
                merged_static = self._merge_object_dict(base_static.get(name, {}), normalized_static)
                base_static[name] = self._normalize_static_entry(merged_static)

            merged: Dict[str, Any] = {}
            for k, v in base.items():
                if k in {"objects", "static_context"}:
                    continue
                merged[k] = copy.deepcopy(v)
            for k, v in incoming.items():
                if k in {"objects", "static_context"}:
                    continue
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = self._merge_object_dict(merged[k], v)
                else:
                    merged[k] = copy.deepcopy(v)
            merged["objects"] = base_objects
            merged["static_context"] = base_static
            self._state = self._canonicalize_state(merged)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of current state for safe reading."""
        with self._lock:
            # Defensive canonicalization on read so UI/logs never show split object keys
            # (e.g. both state["objects"]["<name>"] and state["<name>"]).
            return self._canonicalize_state(copy.deepcopy(self._state))

    def reset(self, initial_state: Dict[str, Any] | None = None) -> None:
        """Reset the full belief store (thread-safe)."""
        with self._lock:
            if isinstance(initial_state, dict):
                seed = copy.deepcopy(initial_state)
            else:
                seed = {"objects": {}, "static_context": {}}
            seed.setdefault("objects", {})
            seed.setdefault("static_context", {})
            self._state = self._canonicalize_state(seed)

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
        """Canonicalize belief updates to {"objects": {...}, "static_context": {...}} shape."""
        out = copy.deepcopy(incoming)
        objects = out.get("objects")
        static_context = out.get("static_context")
        if isinstance(objects, str):
            parsed_objects = self._coerce_object_like(objects)
            if isinstance(parsed_objects, dict):
                objects = parsed_objects
        if isinstance(static_context, str):
            parsed_static = self._coerce_object_like(static_context)
            if isinstance(parsed_static, dict):
                static_context = parsed_static
        normalized_objects: Dict[str, Any] = {}
        normalized_static: Dict[str, Any] = {}
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
        if isinstance(static_context, dict):
            for k, v in list(static_context.items()):
                parsed = self._coerce_object_like(v)
                if parsed is not None:
                    if self._is_dynamic_entry(parsed):
                        normalized_objects[str(k)] = parsed
                    else:
                        normalized_static[str(k)] = self._normalize_static_entry(parsed)
                key_text = str(k).strip()
                if key_text.startswith("{") and key_text.endswith("}"):
                    key_obj = self._coerce_object_like(key_text)
                    if isinstance(key_obj, dict):
                        for kk, vv in key_obj.items():
                            vv_parsed = self._coerce_object_like(vv)
                            if vv_parsed is not None:
                                if self._is_dynamic_entry(vv_parsed):
                                    normalized_objects[str(kk)] = vv_parsed
                                else:
                                    normalized_static[str(kk)] = self._normalize_static_entry(vv_parsed)
        elif isinstance(static_context, list):
            for item in static_context:
                name = str(item or "").strip()
                if not name:
                    continue
                normalized_static[name] = self._normalize_static_entry({})

        for key, value in list(out.items()):
            if key in {"objects", "static_context", "scene", "meta", "task"}:
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
        out["static_context"] = normalized_static
        # Remove object-like keys lifted to the canonical objects block.
        for key in lifted_keys:
            out.pop(key, None)
        return out

    def _canonicalize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure state keeps object tracks only under state['objects']."""
        out = copy.deepcopy(state)
        objects = out.get("objects")
        static_context = out.get("static_context")
        if isinstance(objects, str):
            parsed_objects = self._coerce_object_like(objects)
            if isinstance(parsed_objects, dict):
                objects = parsed_objects
        if isinstance(static_context, str):
            parsed_static = self._coerce_object_like(static_context)
            if isinstance(parsed_static, dict):
                static_context = parsed_static
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
        if not isinstance(static_context, dict):
            static_context = {}
        else:
            for k, v in list(static_context.items()):
                parsed = self._coerce_object_like(v)
                if parsed is not None:
                    static_context[str(k)] = self._normalize_static_entry(parsed)
                key_text = str(k).strip()
                if key_text.startswith("{") and key_text.endswith("}"):
                    key_obj = self._coerce_object_like(key_text)
                    if isinstance(key_obj, dict):
                        for kk, vv in key_obj.items():
                            vv_parsed = self._coerce_object_like(vv)
                            if vv_parsed is not None:
                                static_context[str(kk)] = self._normalize_static_entry(vv_parsed)

        lifted = []
        for key, value in out.items():
            if key in {"objects", "static_context", "scene", "meta", "task"}:
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
                target_map = objects
                existing = target_map.get(key)
                if isinstance(existing, dict):
                    merged = copy.deepcopy(existing)
                    merged.update(parsed)
                    target_map[key] = merged
                else:
                    target_map[key] = parsed
                lifted.append(key)

        # Wake-up: dynamic updates remove entities from background context.
        for name, obj in list(objects.items()):
            if isinstance(obj, dict) and self._is_dynamic_entry(obj):
                static_context.pop(name, None)

        out["objects"] = objects
        out["static_context"] = static_context
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

    @staticmethod
    def _is_dynamic_entry(obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        status = str(obj.get("belief_status", "")).lower().strip()
        if status in {"moving", "contained"}:
            return True
        temporal = str(obj.get("temporal_change", "")).lower()
        if any(w in temporal for w in _MOTION_WORDS):
            return True
        for key in ("displacement", "max_displacement", "delta"):
            try:
                if float(obj.get(key, 0.0)) > 1e-4:
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _normalize_static_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(obj) if isinstance(obj, dict) else {}
        status = str(out.get("belief_status", "static") or "static").lower().strip()
        if status in {"moving", "contained"}:
            # Caller handles promotion. Keep static payload canonical.
            status = "static"
        out["belief_status"] = status if status else "static"
        out["visible"] = bool(out.get("visible", True))
        try:
            out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.7))))
        except Exception:
            out["confidence"] = 0.7
        temporal = str(out.get("temporal_change", "") or "").strip()
        out["temporal_change"] = temporal if temporal else "static in sequence"
        out["stale"] = bool(out.get("stale", False))
        return out
