from __future__ import annotations

import base64
import copy
import json
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Sequence, Tuple

from rc_config import (
    COSMOS_API_KEY,
    FORCE_JSON_RESPONSE,
    COSMOS_HOST,
    COSMOS_MODEL,
    COSMOS_PORT,
    COSMOS_URL,
    IMAGE_FORMAT,
    IMAGE_MIME,
    IMAGE_QUALITY,
    RESOLUTION,
    TIMEOUT_SEC,
)
from rc_parser import parse_json_response

# Sequence-level motion gates tuned for small moving objects in 640x480.
# Keep these permissive enough to catch subtle temporal changes.
MOTION_DETECTED_THRESHOLD = 0.00025
MOTION_STRONG_THRESHOLD = 0.00080


def cosmos_endpoint() -> str:
    if COSMOS_URL:
        return COSMOS_URL
    return f"http://{COSMOS_HOST}:{COSMOS_PORT}/v1/chat/completions"


def _frame_to_b64(frame: Any) -> str:
    """Return base64 for a frame (numpy RGB array) or an on-disk image path."""
    if isinstance(frame, str):
        with open(frame, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow (PIL) not available: {e}")

    img = Image.fromarray(frame)
    # Keep multimodal payload light enough for stable vLLM inference.
    try:
        max_w = int(RESOLUTION[0]) if isinstance(RESOLUTION, (tuple, list)) else 640
        max_h = int(RESOLUTION[1]) if isinstance(RESOLUTION, (tuple, list)) else 480
    except Exception:
        max_w, max_h = 640, 480
    max_w = max(192, min(max_w, 448))
    max_h = max(192, min(max_h, 448))
    if img.width > max_w or img.height > max_h:
        img = img.resize((max_w, max_h))
    buf = BytesIO()
    img.save(buf, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_to_parts(frames: Sequence[Any]) -> List[dict]:
    parts: List[dict] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        b64 = _frame_to_b64(frame)
        if total > 1:
            if i == 0:
                tag = "oldest"
            elif i == total - 1:
                tag = "newest"
            else:
                tag = "middle"
            parts.append({"type": "text", "text": f"Frame {i + 1}/{total} ({tag}, ordered oldest->newest)"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME};base64,{b64}"}})
    return parts


def _frame_to_rgb_array(frame: Any) -> Any:
    """Best-effort conversion to a uint8 RGB array."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy required for motion estimation: {exc}")

    if isinstance(frame, str):
        from PIL import Image

        with Image.open(frame) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _estimate_motion_features(frames: Sequence[Any]) -> Dict[str, float]:
    """Estimate temporal change features from the whole sequence.

    Returns:
    - mean_diff: endpoint normalized mean absolute pixel diff in [0, 1]
    - changed_ratio: endpoint ratio of changed pixels in [0, 1]
    - motion_score: robust motion gate from endpoint + pairwise stats in [0, 1]
    - pairwise_max: max consecutive-frame changed ratio in [0, 1]
    - pairwise_mean: mean consecutive-frame changed ratio in [0, 1]
    """
    out = {
        "mean_diff": 0.0,
        "changed_ratio": 0.0,
        "motion_score": 0.0,
        "pairwise_max": 0.0,
        "pairwise_mean": 0.0,
        "num_frames": float(len(frames)),
    }
    if len(frames) < 2:
        return out
    try:
        import numpy as np

        arrays = [_frame_to_rgb_array(f) for f in frames]

        # Align all frames to the first frame shape.
        base_h, base_w = arrays[0].shape[0], arrays[0].shape[1]
        if base_w > 320 or base_h > 240:
            base_w, base_h = 320, 240
        if arrays[0].shape[1] != base_w or arrays[0].shape[0] != base_h:
            from PIL import Image

            arrays[0] = np.asarray(Image.fromarray(arrays[0]).resize((base_w, base_h)), dtype=np.uint8)

        for i in range(1, len(arrays)):
            if arrays[i].shape[1] != base_w or arrays[i].shape[0] != base_h:
                from PIL import Image

                arrays[i] = np.asarray(Image.fromarray(arrays[i]).resize((base_w, base_h)), dtype=np.uint8)

        def _diff_stats(x: Any, y: Any) -> tuple[float, float]:
            delta = np.abs(x.astype(np.int16) - y.astype(np.int16))
            gray_delta = np.mean(delta, axis=-1)
            mean_diff_v = float(np.mean(gray_delta) / 255.0)
            # Slightly lower threshold to catch smaller object motion.
            changed_ratio_v = float(np.mean(gray_delta > 5.0))
            return mean_diff_v, changed_ratio_v

        # Endpoint difference (oldest -> newest).
        mean_diff, changed_ratio = _diff_stats(arrays[0], arrays[-1])

        # Consecutive differences: catches transient motion that endpoint-only can miss.
        pairwise_changed: List[float] = []
        for i in range(1, len(arrays)):
            _m, c = _diff_stats(arrays[i - 1], arrays[i])
            pairwise_changed.append(c)

        pairwise_max = float(max(pairwise_changed)) if pairwise_changed else 0.0
        pairwise_mean = float(sum(pairwise_changed) / len(pairwise_changed)) if pairwise_changed else 0.0
        motion_score = float(max(mean_diff, changed_ratio, pairwise_max, pairwise_mean))
        out["mean_diff"] = max(0.0, min(1.0, mean_diff))
        out["changed_ratio"] = max(0.0, min(1.0, changed_ratio))
        out["pairwise_max"] = max(0.0, min(1.0, pairwise_max))
        out["pairwise_mean"] = max(0.0, min(1.0, pairwise_mean))
        out["motion_score"] = max(0.0, min(1.0, motion_score))
    except Exception:
        pass
    return out


def build_reason2_messages(
    *,
    frames: Sequence[Any],
    user_text: str,
    short_memory_summary: str,
    long_memory_snippets: Sequence[str],
    prev_belief_json: Dict[str, Any],
    motion_features: Dict[str, float] | None = None,
) -> List[dict]:
    num_images = len(frames)

    def _clip(s: str, n: int = 220) -> str:
        s = str(s or "").strip().replace("\n", " ")
        return s if len(s) <= n else (s[: n - 3] + "...")

    def _compact_belief(prev: Dict[str, Any], max_objs: int = 6) -> Dict[str, Any]:
        if not isinstance(prev, dict):
            return {"objects": {}}
        objs = prev.get("objects")
        if not isinstance(objs, dict):
            return {"objects": {}}
        compact: Dict[str, Any] = {}
        for i, (name, payload) in enumerate(objs.items()):
            if i >= int(max_objs):
                break
            if not isinstance(payload, dict):
                continue
            compact[str(name)] = {
                "belief_status": str(payload.get("belief_status", "unknown")),
                "visible": bool(payload.get("visible", False)),
                "confidence": float(payload.get("confidence", 0.0)),
                "inferred_container": str(payload.get("inferred_container", "")),
                "temporal_change": str(payload.get("temporal_change", "")),
                "stale": bool(payload.get("stale", False)),
            }
        return {"objects": compact}

    ltm_items = [f"- {_clip(s, 140)}" for s in (long_memory_snippets or [])[:2]]
    ltm_text = "\n".join(ltm_items) if ltm_items else "(none)"
    prev_compact = _compact_belief(prev_belief_json)

    motion_features = motion_features or {}
    motion_score = float(motion_features.get("motion_score", 0.0))
    mean_diff = float(motion_features.get("mean_diff", 0.0))
    changed_ratio = float(motion_features.get("changed_ratio", 0.0))
    pairwise_max = float(motion_features.get("pairwise_max", 0.0))
    pairwise_mean = float(motion_features.get("pairwise_mean", 0.0))

    # Detect initialization mode
    is_init_mode = "INITIALIZATION MODE" in user_text
    has_existing_belief = prev_compact.get("objects") and len(prev_compact.get("objects", {})) > 0

    system = (
        "Robot observing tabletop scene. Images are ordered sequence (oldest→newest).\n"
        "Compare first vs last frame. Track objects with stable semantic names (not object_1).\n"
        "Return JSON only. Keys: reply, action, belief_update, meta.\n"
        "action.type: noop|home|inspect|move_ee_pose|open_gripper|close_gripper\n"
        "belief_update.objects: {name: {belief_status, confidence, temporal_change, visible}}\n"
        "belief_status: visible|moving|contained|occluded|unknown\n"
        "reply: short scene + temporal summary (<100 words). NEVER empty.\n"
        "Focus on changes between frames.\n"
    )

    json_shape_hint = {
        "reply": "An orange is on the table near a basket. It rolled slightly between frames.",
        "action": {"type": "noop", "args": {}},
        "belief_update": {
            "objects": {
                "orange": {
                    "belief_status": "moving",
                    "visible": True,
                    "confidence": 0.75,
                    "inferred_container": "",
                    "temporal_change": "moved between frame 1 and frame N",
                },
                "basket": {
                    "belief_status": "visible",
                    "visible": True,
                    "confidence": 0.85,
                    "inferred_container": "",
                    "temporal_change": "static in sequence",
                },
                "table": {
                    "belief_status": "visible",
                    "visible": True,
                    "confidence": 0.90,
                    "inferred_container": "",
                    "temporal_change": "static in sequence",
                },
            }
        },
        "meta": {"num_images": num_images, "temporal_summary": "orange moved slightly on table"},
    }

    if is_init_mode:
        # INITIALIZATION MODE: VLM creates initial belief state from static frame
        # COMPACT VERSION to save tokens
        user_msg = (
            f"{user_text}\n\n"
            f"Images: {num_images} (static)\n"
            "INIT MODE: Identify all objects. Use semantic names (orange, grey_basket, table). NO object_1.\n"
            "Each: belief_status='visible', confidence=0.7-0.95, visible=true, temporal_change='initial_state'\n"
            f"JSON format:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
        )
    elif has_existing_belief:
        # RUNTIME MODE: VLM updates existing belief state based on changes
        # COMPACT VERSION to save tokens
        user_msg = (
            f"{user_text or 'Update'}\n\n"
            f"Current belief:\n{json.dumps(prev_compact, ensure_ascii=True)}\n"
            f"STM: {short_memory_summary[:100]}\n"
            f"Images: {num_images} | motion={motion_score:.3f}\n\n"
            "UPDATE MODE: Preserve object names. Detect changes.\n"
            "Rules: Keep names. Update belief_status (visible/moving/contained/occluded). Describe temporal_change.\n"
            f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
        )
    else:
        # FALLBACK: No existing belief, treat as first observation (legacy behavior)
        user_msg = (
            f"User request: {user_text or '(none)'}\n\n"
            f"Previous belief JSON:\n{json.dumps(prev_compact, ensure_ascii=True)}\n\n"
            f"Short-term memory summary:\n{short_memory_summary}\n\n"
            f"Long-term memory snippets:\n{ltm_text}\n\n"
            f"Images provided: {num_images}. Treat them as one short clip.\n"
            f"Sequence diagnostics from frame differencing: motion_score={motion_score:.5f}, endpoint_mean_diff={mean_diff:.5f}, endpoint_changed_ratio={changed_ratio:.5f}, pairwise_max_changed_ratio={pairwise_max:.5f}, pairwise_mean_changed_ratio={pairwise_mean:.5f}\n"
            "Analyze the full sequence as one event. Compare first and last frame, then cross-check intermediates.\n"
            "Your reply must emphasize temporal change rather than a single-frame snapshot.\n"
            "First provide a short scene summary, then describe what changed over time.\n"
            "Explicitly mention any object state transitions detected from the ordered frames.\n"
            "If there is a trajectory or displacement, report it explicitly in temporal_summary and object temporal_change.\n"
            "Always include salient objects and their updated state in belief_update.objects.\n"
            "IMPORTANT: Use descriptive object names as dictionary keys (e.g., 'orange', 'basket', 'table', 'cup').\n"
            "NEVER use generic names like 'object_1', 'object_2'. Name objects by what they are.\n"
            "If ground-truth object names are provided, use those exact names.\n"
            "Do not return an empty objects dictionary unless the scene is completely unreadable.\n"
            "If uncertain, keep object entries and set belief_status='unknown' with lower confidence.\n"
            f"Output format example:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
            "Do not wrap JSON in markdown fences.\n"
        )

    parts = [{"type": "text", "text": user_msg}]
    parts.extend(frames_to_parts(frames))

    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


_RESERVED_BELIEF_KEYS = {"objects", "scene", "meta"}
_MOTION_WORDS = (
    "move",
    "moving",
    "motion",
    "fall",
    "falling",
    "drop",
    "dropping",
    "roll",
    "rolled",
    "swing",
    "shift",
    "changed",
)


def _contains_any(text: str, words: Tuple[str, ...]) -> bool:
    for w in words:
        if w in text:
            return True
    return False


_OBJECT_TOKEN_TO_CANONICAL = {
    "ball": "ball",
    "orange": "orange",
    "fruit": "fruit",
    "table": "table",
    "desk": "table",
    "counter": "table",
    "bucket": "bucket",
    "buckets": "bucket",
    "basket": "basket",
    "baskets": "basket",
    "container": "container",
    "containers": "container",
    "bin": "container",
    "robot": "robot_arm",
    "arm": "robot_arm",
    "gripper": "robot_arm",
    "window": "window",
    "floor": "floor",
    "wall": "wall",
    "ceiling": "ceiling",
    "room": "room",
}

_GENERIC_OBJECT_STOPWORDS = {
    "image",
    "scene",
    "frame",
    "frames",
    "sequence",
    "time",
    "change",
    "changes",
    "movement",
    "motion",
    "position",
    "background",
    "foreground",
    "left",
    "right",
    "center",
    "middle",
    "top",
    "bottom",
    "oldest",
    "newest",
    "ordered",
    "robot",
}


def _extract_scene_objects_from_text(reply: str) -> List[str]:
    """Extract stable object names from free-form model reply."""
    text = str(reply or "").lower()
    tokens = re.findall(r"[a-z_][a-z0-9_]+", text)
    seen = set()
    names: List[str] = []
    for tok in tokens:
        name = _OBJECT_TOKEN_TO_CANONICAL.get(tok)
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    # Fallback: extract noun-like tokens after articles from free-form text.
    if not names:
        for cand in re.findall(r"\b(?:a|an|the)\s+([a-z][a-z0-9_-]{2,})\b", text):
            if cand in _GENERIC_OBJECT_STOPWORDS:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            names.append(cand)
            if len(names) >= 8:
                break
    return names


def _looks_like_object_entry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {
        "belief_status",
        "visible",
        "confidence",
        "inferred_container",
        "temporal_change",
        "stale",
        "location",
        "pos",
        "position",
    }
    return any(k in value for k in expected)


def _coerce_object_entry(value: Any) -> Dict[str, Any] | None:
    """Best-effort object-entry coercion from dict or JSON-encoded string."""
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                # Fall through to loose parsing for partially malformed strings.
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        # Loose extraction for malformed object snippets emitted by VLMs.
        status = re.search(r'"?belief_status"?\s*:\s*"([^"]+)"', text)
        temporal = re.search(r'"?temporal_change"?\s*:\s*"([^"]+)"', text)
        container = re.search(r'"?inferred_container"?\s*:\s*"([^"]*)"', text)
        conf = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', text)
        visible = re.search(r'"?visible"?\s*:\s*(true|false)', text, re.IGNORECASE)
        stale = re.search(r'"?stale"?\s*:\s*(true|false)', text, re.IGNORECASE)
        if status or temporal or container or conf or visible or stale:
            out: Dict[str, Any] = {}
            if status:
                out["belief_status"] = status.group(1).strip().lower()
            if temporal:
                out["temporal_change"] = temporal.group(1).strip()
            if container:
                out["inferred_container"] = container.group(1).strip()
            if conf:
                try:
                    out["confidence"] = float(conf.group(1))
                except Exception:
                    pass
            if visible:
                out["visible"] = visible.group(1).lower() == "true"
            if stale:
                out["stale"] = stale.group(1).lower() == "true"
            return out if out else None
    return None


def _infer_objects_from_reply(reply: str, *, motion_score: float = 0.0) -> Dict[str, Dict[str, Any]]:
    text = str(reply or "").lower()
    inferred: Dict[str, Dict[str, Any]] = {}

    has_motion = _contains_any(text, _MOTION_WORDS) or float(motion_score) >= MOTION_DETECTED_THRESHOLD
    has_contained = bool(re.search(r"\b(in|inside|into)\b.*\b(bucket|basket|container|bin)\b", text))
    object_names = _extract_scene_objects_from_text(reply)

    primary_dynamic = object_names[0] if object_names else "moving_object_1"
    if has_motion and primary_dynamic in {"room", "window", "wall", "floor", "ceiling"} and len(object_names) > 1:
        primary_dynamic = object_names[1]

    for name in object_names:
        status = "visible"
        temporal = "static in sequence"
        confidence = 0.58
        container = ""

        if has_motion and name == primary_dynamic:
            status = "moving"
            temporal = "position changed across ordered frames"
            confidence = 0.68
        elif has_motion:
            temporal = "context changed in sequence"

        if has_contained and name == primary_dynamic:
            status = "contained"
            temporal = "moved into a container across the sequence"
            if "bucket" in text:
                container = "bucket"
            elif "basket" in text:
                container = "basket"
            elif "container" in text:
                container = "container"

        if name in {"room", "window", "floor", "wall", "ceiling"} and has_motion:
            temporal = "background mostly static while foreground motion changed"

        inferred[name] = {
            "belief_status": status,
            "visible": True,
            "confidence": confidence,
            "inferred_container": container,
            "temporal_change": temporal,
            "stale": False,
        }

    if not inferred:
        base_temporal = "temporal change detected from sequence" if has_motion else "static in sequence"
        inferred["object_1"] = {
            "belief_status": "moving" if has_motion else "visible",
            "visible": True,
            "confidence": 0.40 if has_motion else 0.35,
            "inferred_container": "",
            "temporal_change": base_temporal,
            "stale": False,
        }
        inferred["object_2"] = {
            "belief_status": "visible",
            "visible": True,
            "confidence": 0.30,
            "inferred_container": "",
            "temporal_change": "static in sequence",
            "stale": False,
        }
        inferred["object_3"] = {
            "belief_status": "unknown" if has_motion else "visible",
            "visible": True,
            "confidence": 0.28,
            "inferred_container": "",
            "temporal_change": base_temporal if has_motion else "static in sequence",
            "stale": False,
        }

    return inferred


def _ensure_minimum_tracks(
    tracks: Dict[str, Dict[str, Any]],
    *,
    reply: str,
    motion_score: float,
    min_tracks: int = 3,
) -> Dict[str, Dict[str, Any]]:
    out = copy.deepcopy(tracks if isinstance(tracks, dict) else {})
    text = str(reply or "").lower()
    has_motion = float(motion_score) >= MOTION_DETECTED_THRESHOLD or _contains_any(text, _MOTION_WORDS)

    scene_candidates = []
    extracted = _extract_scene_objects_from_text(reply)
    for name in extracted:
        scene_candidates.append((name, (name,)))

    structural = {"room", "window", "wall", "floor", "ceiling"}
    dynamic_hint = ""
    for name in extracted:
        if name not in structural:
            dynamic_hint = name
            break

    for name, kws in scene_candidates:
        if len(out) >= int(min_tracks):
            break
        if name in out:
            continue
        if any(kw in text for kw in kws):
            out[name] = {
                "belief_status": "moving" if (has_motion and name == dynamic_hint) else "visible",
                "visible": True,
                "confidence": 0.35,
                "inferred_container": "",
                "temporal_change": "position changed across ordered frames" if has_motion else "static in sequence",
                "stale": False,
            }

    i = 1
    while len(out) < int(min_tracks):
        key = f"scene_object_{i}"
        i += 1
        if key in out:
            continue
        out[key] = {
            "belief_status": "moving" if has_motion and len(out) == 0 else "visible",
            "visible": True,
            "confidence": 0.25,
            "inferred_container": "",
            "temporal_change": "temporal change detected from sequence" if has_motion else "static in sequence",
            "stale": False,
        }
    return out


def _promote_dynamic_track(
    tracks: Dict[str, Dict[str, Any]], *, motion_score: float, temporal_text: str
) -> Dict[str, Dict[str, Any]]:
    """Guarantee at least one dynamic object when diagnostics indicate motion."""
    out = copy.deepcopy(tracks if isinstance(tracks, dict) else {})
    if float(motion_score) < MOTION_DETECTED_THRESHOLD:
        return out

    # Prefer a non-structural object as dynamic anchor.
    structural = {"room", "window", "wall", "floor", "ceiling", "table", "desk", "counter"}
    candidate = ""
    for name in out.keys():
        n = str(name).lower()
        if n not in structural:
            candidate = name
            break
    if not candidate and out:
        candidate = next(iter(out.keys()))
    if not candidate:
        candidate = "moving_object_1"

    cur = out.get(candidate, {})
    if not isinstance(cur, dict):
        cur = {}
    cur["belief_status"] = "moving"
    cur["visible"] = True
    cur["stale"] = False
    cur["confidence"] = max(float(cur.get("confidence", 0.0) or 0.0), 0.55)
    cur["temporal_change"] = temporal_text or "position changed across ordered frames"
    out[candidate] = cur
    return out


def _normalize_model_output(
    raw: Dict[str, Any], *, prev_belief_json: Dict[str, Any], num_images: int, motion_score: float = 0.0
) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    reply = str(out.get("reply") or "").strip()

    # If reply is empty, generate one from belief_update for better UX
    if not reply:
        belief_update = out.get("belief_update", {})
        if isinstance(belief_update, dict):
            objects = belief_update.get("objects", {})
            if isinstance(objects, dict) and objects:
                obj_descriptions = []
                for name, obj in list(objects.items())[:3]:  # Describe top 3 objects
                    if isinstance(obj, dict):
                        status = obj.get("belief_status", "unknown")
                        obj_descriptions.append(f"{name}: {status}")
                if obj_descriptions:
                    reply = "Observing: " + ", ".join(obj_descriptions)
        if not reply:
            reply = f"Processing {num_images} frames" if num_images > 0 else "Scene observed"

    action = out.get("action")
    if not isinstance(action, dict):
        action = {"type": "noop", "args": {}}
    action.setdefault("type", "noop")
    if not isinstance(action.get("args"), dict):
        action["args"] = {}

    belief_update = out.get("belief_update")
    if isinstance(belief_update, str):
        try:
            belief_update = json.loads(belief_update)
        except Exception:
            belief_update = {}
    if not isinstance(belief_update, dict):
        belief_update = {}

    prev_objects = {}
    if isinstance(prev_belief_json, dict):
        p = prev_belief_json.get("objects")
        if isinstance(p, dict):
            prev_objects = copy.deepcopy(p)

    incoming_objects: Dict[str, Dict[str, Any]] = {}

    objects_block = belief_update.get("objects")
    if isinstance(objects_block, str):
        try:
            objects_block = json.loads(objects_block)
        except Exception:
            objects_block = {}
    if isinstance(objects_block, dict):
        for name, value in objects_block.items():
            parsed = _coerce_object_entry(value)
            if parsed is not None:
                incoming_objects[str(name)] = parsed

    for key, value in belief_update.items():
        if key in _RESERVED_BELIEF_KEYS:
            continue
        parsed = _coerce_object_entry(value)
        if parsed is not None and _looks_like_object_entry(parsed):
            incoming_objects[str(key)] = parsed

    for key, value in out.items():
        if key in {"reply", "action", "belief_update", "meta"}:
            continue
        parsed = _coerce_object_entry(value)
        if parsed is not None and _looks_like_object_entry(parsed):
            incoming_objects[str(key)] = parsed

    if not incoming_objects and reply:
        incoming_objects = _infer_objects_from_reply(reply, motion_score=float(motion_score))

    normalized_objects: Dict[str, Dict[str, Any]] = {}

    # Keep previously tracked objects unless the new update explicitly overwrites them.
    for name, prev in prev_objects.items():
        if isinstance(prev, dict):
            normalized_objects[str(name)] = copy.deepcopy(prev)

    for name, obj in incoming_objects.items():
        if not isinstance(obj, dict):
            continue
        cur = normalized_objects.get(name, {})
        if not isinstance(cur, dict):
            cur = {}
        merged = copy.deepcopy(cur)
        for k, v in obj.items():
            merged[k] = copy.deepcopy(v)
        if "belief_status" not in merged:
            merged["belief_status"] = "unknown"
        status = str(merged.get("belief_status") or "unknown").strip().lower()
        if status in {"falling", "dropping"}:
            status = "moving"
        if status == "unknown" and float(motion_score) >= MOTION_DETECTED_THRESHOLD and _contains_any(
            reply.lower(), _MOTION_WORDS
        ):
            status = "moving"
            if not str(merged.get("temporal_change") or "").strip():
                merged["temporal_change"] = "temporal change detected from sequence"
        merged["belief_status"] = status
        merged["visible"] = bool(merged.get("visible", status in ("visible", "moving", "contained")))
        merged["stale"] = bool(merged.get("stale", status == "unknown"))
        if "confidence" not in merged:
            merged["confidence"] = 0.5 if status == "unknown" else 0.65
        try:
            merged["confidence"] = max(0.0, min(1.0, float(merged["confidence"])))
        except Exception:
            merged["confidence"] = 0.5 if status == "unknown" else 0.65
        if "temporal_change" not in merged:
            merged["temporal_change"] = "unknown"
        normalized_objects[name] = merged

    if not normalized_objects:
        normalized_objects = _infer_objects_from_reply(reply, motion_score=float(motion_score))

    normalized_objects = _ensure_minimum_tracks(
        normalized_objects, reply=reply, motion_score=float(motion_score), min_tracks=3
    )

    # If diagnostics indicate motion, force a dynamic anchor even when model output is weak.
    temporal_hint = ""
    low_reply = str(reply or "").lower()
    if "fall" in low_reply or "drop" in low_reply:
        temporal_hint = "fell/dropped across sequence"
    normalized_objects = _promote_dynamic_track(
        normalized_objects, motion_score=float(motion_score), temporal_text=temporal_hint
    )

    # If sequence motion is detected but all tracks still look static/unknown, force one motion anchor.
    if float(motion_score) >= MOTION_DETECTED_THRESHOLD:
        has_dynamic_track = False
        for _name, obj in normalized_objects.items():
            if not isinstance(obj, dict):
                continue
            s = str(obj.get("belief_status", "")).lower()
            t = str(obj.get("temporal_change", "")).lower()
            if s in {"moving", "contained"} or (
                t and t not in {"none", "unknown", "no change", "static", "static in sequence"}
            ):
                has_dynamic_track = True
                break
        if not has_dynamic_track:
            normalized_objects["moving_object_1"] = {
                "belief_status": "moving",
                "visible": True,
                "confidence": 0.45,
                "inferred_container": "",
                "temporal_change": "temporal change detected from frame sequence",
                "stale": False,
            }

    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["num_images"] = int(num_images)
    temporal_changed = False
    for name, cur_obj in normalized_objects.items():
        prev_obj = prev_objects.get(name) if isinstance(prev_objects, dict) else None
        if isinstance(cur_obj, dict):
            status = str(cur_obj.get("belief_status", "unknown")).lower()
            t_change = str(cur_obj.get("temporal_change", "")).lower()
            if status in {"moving", "contained"}:
                temporal_changed = True
            if t_change and t_change not in {"none", "unknown", "no change", "static"}:
                temporal_changed = True
            if isinstance(prev_obj, dict) and prev_obj.get("belief_status") != cur_obj.get("belief_status"):
                temporal_changed = True

    existing_temporal = str(meta.get("temporal_summary", "")).strip().lower()
    if (not temporal_changed) and num_images >= 2 and float(motion_score) >= MOTION_DETECTED_THRESHOLD:
        temporal_changed = True

    if temporal_changed:
        if float(motion_score) >= MOTION_STRONG_THRESHOLD:
            meta["temporal_summary"] = "clear temporal change observed"
        else:
            meta["temporal_summary"] = "temporal change observed"
    elif existing_temporal:
        meta["temporal_summary"] = existing_temporal
    elif normalized_objects:
        if num_images >= 2 and float(motion_score) >= MOTION_DETECTED_THRESHOLD:
            meta["temporal_summary"] = "possible temporal change detected from frame deltas"
        else:
            meta["temporal_summary"] = "no temporal change observed"
    else:
        meta["temporal_summary"] = "unknown"

    if "scene_summary" not in meta:
        head = reply[:160].strip()
        meta["scene_summary"] = head if head else "unknown"

    if float(motion_score) >= MOTION_DETECTED_THRESHOLD:
        low = str(reply or "").lower()
        if "no temporal change" in low or "no visible changes" in low or "scene remains static" in low:
            reply = (reply.rstrip() + " Temporal change is detected across the ordered frame sequence.").strip()

    return {
        "reply": reply or "<empty response>",
        "action": action,
        "belief_update": {"objects": normalized_objects},
        "meta": meta,
    }


def call_reason2(messages: List[dict], *, max_tokens: int = 512, temperature: float = 0.2) -> str:
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        raise RuntimeError("COSMOS_MODEL is not set.")

    endpoint = cosmos_endpoint()

    # CRITICAL: Cosmos Reason2-8B has a 2048 token context limit
    # We need to dynamically adjust max_tokens based on input length
    MODEL_MAX_CONTEXT = 2048
    SAFETY_MARGIN = 20  # Reserve tokens for tokenization variance
    TOKENS_PER_IMAGE = 350  # VLM image tokens (NOT related to base64 byte size)

    # Estimate input tokens: count text chars (exclude base64 image data) + per-image fixed cost
    num_images = 0
    text_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_chars += len(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        num_images += 1
    estimated_text_tokens = text_chars // 5  # ~5 chars per token (calibrated: 930→744 matches actual)
    estimated_input_tokens = estimated_text_tokens + (num_images * TOKENS_PER_IMAGE)

    available_tokens = MODEL_MAX_CONTEXT - estimated_input_tokens - SAFETY_MARGIN
    adjusted_max_tokens = max(128, min(max_tokens, available_tokens))

    if adjusted_max_tokens < max_tokens:
        print(f"[VLM] Adjusted max_tokens: {max_tokens} → {adjusted_max_tokens} (est_input={estimated_input_tokens}, text={estimated_text_tokens}, imgs={num_images}x{TOKENS_PER_IMAGE}, available={available_tokens})", flush=True)

    payload = {"model": COSMOS_MODEL, "messages": messages, "max_tokens": adjusted_max_tokens, "temperature": temperature}
    if FORCE_JSON_RESPONSE:
        payload["response_format"] = {"type": "json_object"}

    # Log payload size for debugging (not content - too large with images)
    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    print(f"[VLM] POST {endpoint} | model={COSMOS_MODEL} | images={num_images} | payload={payload_bytes/1024:.1f}KB | max_tokens={adjusted_max_tokens}", flush=True)

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if COSMOS_API_KEY:
        headers["Authorization"] = f"Bearer {COSMOS_API_KEY}"

    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(exc)
        print(f"[VLM] HTTP ERROR {exc.code} after {time.time()-t0:.1f}s: {body[:500]}", flush=True)
        raise RuntimeError(f"HTTP {exc.code}: {body}")
    except Exception as exc:
        print(f"[VLM] CONNECTION ERROR after {time.time()-t0:.1f}s: {exc}", flush=True)
        raise

    elapsed = time.time() - t0
    res = json.loads(raw)
    content = str(res["choices"][0]["message"]["content"]).strip()
    # Log raw response for debugging
    usage = res.get("usage", {})
    print(f"[VLM] Response in {elapsed:.1f}s | tokens={usage} | content_len={len(content)}", flush=True)
    print(f"[VLM] Raw content: {content[:500]}", flush=True)
    return content


def reason2_decide(
    *,
    frames: Sequence[Any],
    user_text: str,
    short_memory_summary: str,
    long_memory_snippets: Sequence[str],
    prev_belief_json: Dict[str, Any],
) -> Dict[str, Any]:
    """High-level call used by the agent graph.

    Returns parsed JSON dict (reply/action/belief_update/meta).
    """
    motion_features = _estimate_motion_features(frames)
    motion_score = float(motion_features.get("motion_score", 0.0))

    def _fallback_from_text(raw_text: str, parse_error: str) -> Dict[str, Any]:
        reply_text = (raw_text or "").strip() or "<empty response>"
        print(f"[VLM] FALLBACK: parse_error={parse_error} | reply_preview={reply_text[:200]}", flush=True)
        fallback = {
            "reply": reply_text,
            "action": {"type": "noop", "args": {}},
            "belief_update": {},
            "meta": {
                "num_images": int(len(frames)),
                "temporal_summary": "fallback_text_mode",
                "parse_error": str(parse_error),
                "motion_score": float(motion_score),
            },
            "_timing_sec": round(time.time() - t0, 3),
        }
        return _normalize_model_output(
            fallback, prev_belief_json=prev_belief_json, num_images=int(len(frames)), motion_score=float(motion_score)
        )

    t0 = time.time()
    try:
        # First attempt: multimodal (frames included).
        messages = build_reason2_messages(
            frames=frames,
            user_text=user_text,
            short_memory_summary=short_memory_summary,
            long_memory_snippets=long_memory_snippets,
            prev_belief_json=prev_belief_json,
            motion_features=motion_features,
        )
        try:
            content = call_reason2(messages, max_tokens=512, temperature=0.1)
        except Exception as exc:
            # Fallback: retry text-only if the endpoint rejects images.
            msg = str(exc)
            if frames and ("image" in msg.lower()) and ("at most 0" in msg.lower() or "may be provided" in msg.lower()):
                messages = build_reason2_messages(
                    frames=[],
                    user_text=user_text,
                    short_memory_summary=short_memory_summary,
                    long_memory_snippets=long_memory_snippets,
                    prev_belief_json=prev_belief_json,
                    motion_features=motion_features,
                )
                content = call_reason2(messages, max_tokens=512, temperature=0.1)
                try:
                    out = parse_json_response(content)
                except Exception as parse_exc:
                    out = _fallback_from_text(content, f"text-only fallback parse failed: {parse_exc}")
                    out.setdefault("meta", {})
                    if isinstance(out["meta"], dict):
                        out["meta"]["images_dropped"] = True
                        out["meta"]["num_images"] = 0
                    return out
                normalized = _normalize_model_output(
                    out, prev_belief_json=prev_belief_json, num_images=0, motion_score=float(motion_score)
                )
                normalized.setdefault("meta", {})
                if isinstance(normalized["meta"], dict):
                    normalized["meta"]["images_dropped"] = True
                    normalized["meta"]["num_images"] = 0
                    normalized["meta"]["motion_score"] = float(motion_score)
                normalized["_timing_sec"] = round(time.time() - t0, 3)
                return normalized
            raise
        try:
            out = parse_json_response(content)
        except Exception as parse_exc:
            return _fallback_from_text(content, str(parse_exc))
        if not isinstance(out, dict):
            return _fallback_from_text(str(content), "non-dict JSON output")

        normalized = _normalize_model_output(
            out, prev_belief_json=prev_belief_json, num_images=int(len(frames)), motion_score=float(motion_score)
        )
        normalized.setdefault("meta", {})
        if isinstance(normalized["meta"], dict):
            normalized["meta"]["motion_score"] = float(motion_score)
        normalized["_timing_sec"] = round(time.time() - t0, 3)
        return normalized
    except Exception as exc:
        # Never raise to worker loop on model parsing/formatting issues.
        import traceback
        print(f"[VLM] EXCEPTION in reason2_decide: {exc}", flush=True)
        traceback.print_exc()
        return _fallback_from_text("", f"reason2_decide exception: {exc}")
