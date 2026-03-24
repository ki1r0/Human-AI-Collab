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
    COSMOS_MAX_MODEL_LEN,
    COSMOS_MODEL,
    IMAGE_FORMAT,
    IMAGE_MIME,
    IMAGE_QUALITY,
    RESOLUTION,
    TIMEOUT_SEC,
    cosmos_chat_completions_url,
    cosmos_is_configured,
)
from .parser import parse_json_response

# Sequence-level motion gates tuned for small moving objects in 640x480.
# Keep these permissive enough to catch subtle temporal changes.
MOTION_DETECTED_THRESHOLD = 0.00025
MOTION_STRONG_THRESHOLD = 0.00080
MIN_COMPLETION_TOKENS = 96


def cosmos_endpoint() -> str:
    return cosmos_chat_completions_url()


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
    prompt_compact_level: int = 0,
    trigger: Dict[str, Any] | None = None,
) -> List[dict]:
    num_images = len(frames)
    user_text_clean = str(user_text or "").strip()
    user_lower = user_text_clean.lower()
    is_user_question = ("?" in user_text_clean) and ("initialization mode" not in user_lower)
    trigger_dict = trigger if isinstance(trigger, dict) else {}
    trigger_type = str(trigger_dict.get("type") or "").strip().lower()
    is_auto_gt = trigger_type == "gt_change"
    no_visual_guard = ""
    if num_images <= 0:
        no_visual_guard = (
            "No images attached. Do not invent visual facts.\n"
            "Answer from CURRENT BELIEF if available.\n"
        )
    memory_priority_guard = ""
    if is_user_question:
        memory_priority_guard = (
            "User asked a question. Answer based on BELIEF, not just images.\n"
            "If belief says object is contained/located, state that directly in reply.\n"
        )

    compact_level = max(0, int(prompt_compact_level))
    if is_auto_gt and num_images >= 3 and compact_level < 3:
        compact_level = 3
    elif num_images >= 4 and compact_level < 1:
        compact_level = 1

    def _clip(s: str, n: int = 220) -> str:
        s = str(s or "").strip().replace("\n", " ")
        return s if len(s) <= n else (s[: n - 3] + "...")

    if compact_level <= 0:
        belief_max_objs = 6
        static_max_objs = 10
        stm_max_len = 320
        ltm_max_items = 2
        ltm_item_len = 180
        hint_max_objs = 5
    elif compact_level == 1:
        belief_max_objs = 4
        static_max_objs = 6
        stm_max_len = 180
        ltm_max_items = 1
        ltm_item_len = 120
        hint_max_objs = 4
    elif compact_level == 2:
        belief_max_objs = 3
        static_max_objs = 4
        stm_max_len = 120
        ltm_max_items = 0
        ltm_item_len = 80
        hint_max_objs = 3
    elif compact_level == 3:
        belief_max_objs = 2
        static_max_objs = 3
        stm_max_len = 80
        ltm_max_items = 0
        ltm_item_len = 60
        hint_max_objs = 2
    else:
        belief_max_objs = 2
        static_max_objs = 2
        stm_max_len = 60
        ltm_max_items = 0
        ltm_item_len = 40
        hint_max_objs = 2

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
                "confidence": round(float(payload.get("confidence", 0.0)), 2),
                "inferred_container": str(payload.get("inferred_container", "")),
                "stale": bool(payload.get("stale", False)),
            }
        return {"objects": compact}

    def _compact_static_context(prev: Dict[str, Any], max_objs: int = 10) -> Dict[str, Any]:
        if not isinstance(prev, dict):
            return {"static_context": {}}
        raw = prev.get("static_context")
        compact: Dict[str, Any] = {}
        if isinstance(raw, dict):
            for i, (name, payload) in enumerate(raw.items()):
                if i >= int(max_objs):
                    break
                if isinstance(payload, dict):
                    compact[str(name)] = {
                        "belief_status": str(payload.get("belief_status", "static")),
                        "confidence": round(float(payload.get("confidence", 0.0)), 2),
                        "visible": bool(payload.get("visible", True)),
                        "temporal_change": str(payload.get("temporal_change", "static in sequence")),
                    }
                else:
                    compact[str(name)] = {
                        "belief_status": "static",
                        "confidence": 0.7,
                        "visible": True,
                        "temporal_change": "static in sequence",
                    }
        elif isinstance(raw, list):
            for i, name in enumerate(raw):
                if i >= int(max_objs):
                    break
                compact[str(name)] = {
                    "belief_status": "static",
                    "confidence": 0.7,
                    "visible": True,
                    "temporal_change": "static in sequence",
                }
        return {"static_context": compact}

    prev_compact = _compact_belief(prev_belief_json, max_objs=belief_max_objs)
    prev_static_compact = _compact_static_context(prev_belief_json, max_objs=static_max_objs)
    stm_compact = _clip(short_memory_summary, stm_max_len) if str(short_memory_summary or "").strip() else ""
    ltm_compact = [_clip(s, ltm_item_len) for s in list(long_memory_snippets or [])[:ltm_max_items] if str(s or "").strip()]

    motion_features = motion_features or {}
    motion_score = float(motion_features.get("motion_score", 0.0))

    def _trigger_name(value: Any) -> str:
        name = str(value or "").strip()
        if "/" in name:
            parts = [p for p in name.split("/") if p]
            if parts:
                name = parts[-1]
        return name

    changed_objects: List[str] = []
    raw_changed = trigger_dict.get("changed_objects")
    if isinstance(raw_changed, list):
        for item in raw_changed:
            n = _trigger_name(item)
            if n:
                changed_objects.append(n)

    # Detect initialization mode
    is_init_mode = "INITIALIZATION MODE" in user_text
    has_existing_belief = prev_compact.get("objects") and len(prev_compact.get("objects", {})) > 0

    init_required_names: List[str] = []
    raw_init_names = trigger_dict.get("init_interactables")
    if isinstance(raw_init_names, list):
        init_required_names.extend([str(n) for n in raw_init_names if str(n or "").strip()])
    raw_all_names = trigger_dict.get("all_interactables")
    if isinstance(raw_all_names, list):
        init_required_names.extend([str(n) for n in raw_all_names if str(n or "").strip()])
    raw_init_form = trigger_dict.get("init_belief_form")
    if isinstance(raw_init_form, dict):
        belief_block = raw_init_form.get("belief_state_update")
        if isinstance(belief_block, dict):
            form_objs = belief_block.get("objects")
            if isinstance(form_objs, dict):
                init_required_names.extend([str(k) for k in form_objs.keys() if str(k or "").strip()])
    init_required_names = _dedupe_names(init_required_names)

    # Extract GT-discovered object names from the augmented user_text (injected at runtime
    # by the agent graph from StateMonitor). These are NOT hardcoded — they come from
    # runtime USD stage discovery. See docs/rules.md §1.
    gt_names = []
    _gt_match = re.search(r"Known scene object names[^:]*:\s*(.+?)(?:\.|$)", user_text)
    if _gt_match:
        gt_names = [n.strip() for n in _gt_match.group(1).split(",") if n.strip()]
    # Also pull from existing belief keys.
    if not gt_names and has_existing_belief:
        gt_names = list(prev_compact.get("objects", {}).keys())[:8]
    if is_init_mode and init_required_names:
        gt_names = list(init_required_names)

    # --- Three-Stage Cognitive Process ---
    # System prompt instructs the model to perform TWO-STEP reasoning:
    #   Step A = Perception/STM (transient): what the images literally show
    #   Step B = Belief Update (persistent): compare perception with existing belief
    #
    # KEY: JSON field order matters — models fill fields in declaration order.
    if is_init_mode:
        if compact_level <= 1:
            system = (
                "Initialization inquiry (single static frame).\n"
                "Return strict JSON only with keys in this order: belief_state_update, stm_observation, reply, action.\n"
                "PRIORITY: fill belief_state_update.objects first for every key in RequiredForm.\n"
                "Use semantic names from provided context only; never emit generic IDs.\n"
                "For each object include: belief_status, visible, confidence, inferred_container, temporal_change.\n"
                "Default to belief_status='present' for existing parts in scene.\n"
                "Set inferred_container='' unless clearly and explicitly inside another container.\n"
                "In init mode set temporal_change='initial_state'.\n"
                "Keep stm_observation/reply concise and non-repetitive. action.type must be noop.\n"
            )
        else:
            system = (
                "Init mode, one static frame.\n"
                "Return strict JSON keys: belief_state_update, stm_observation, reply, action.\n"
                "Fill all RequiredForm object keys; temporal_change='initial_state'; action.type='noop'.\n"
            )
    elif is_auto_gt:
        system = (
            "Auto inquiry for temporal tracking. Frames are ordered oldest->newest.\n"
            "PRIORITY: output machine-readable belief_state_update first.\n"
            "Return JSON only with keys in order: belief_state_update, stm_observation, reply, action.\n"
            "belief_state_update.objects is primary. For each object include: "
            "belief_status, confidence, inferred_container, temporal_change, visible.\n"
            "Infer temporal transitions from the sequence (e.g., descending/falling, moving, entered container, occluded).\n"
            "Use semantic object names from context; never emit generic IDs.\n"
            "Only include static_context when structural/background objects changed.\n"
            "reply <=15 words. action.type defaults to noop.\n"
        )
    elif compact_level <= 0:
        system = (
            "Robot scene observer. Images: oldest→newest sequence.\n"
            "Use provided object names. NEVER use generic IDs like object_1.\n"
            "TWO-STEP reasoning:\n"
            "A) Perception: What images literally show. Max 12 words.\n"
            "B) Belief Update: Compare with CURRENT BELIEF. Object permanence: "
            "if object vanished near container -> contained. "
            "If not visible but belief=contained -> keep contained.\n"
            "CONTAINMENT RULE: Distinguish 'ON' a surface from 'INSIDE' a container.\n"
            "  ON a surface = object rests on top of a flat structure (table, shelf, floor, ground plane).\n"
            "  INSIDE a container = object is enclosed inside a bucket, box, bin, or hollow vessel.\n"
            "  Tables, floors, and ground planes are NEVER containers. "
            "Never set inferred_container to a table or floor prim.\n"
            "Return JSON. Write 'reply' key FIRST.\n"
            "reply: <=25 words. TWO sentences only: "
            "(1) direct visual evidence from current frames "
            "(2) belief conclusion consistent with prior belief + current evidence.\n"
            "belief_state_update.objects: Active Tracking Targets (interactable/movable).\n"
            "belief_state_update.static_context: Background structural objects only; omit if unchanged.\n"
            "Wake-up: if static_context object moves, promote it to objects.\n"
            "objects.{name}: {belief_status, confidence, inferred_container}\n"
            "belief_status: visible|moving|contained|occluded|unknown\n"
            "stm_observation: <=12 words. Raw literal image content only.\n"
            "action.type: noop|home|inspect|move_ee_pose|open_gripper|close_gripper"
            "|combine|separate\n"
            "combine: snap a part onto another part at a socket."
            " args: {partA: str, partB: str, plug: str, socket: str}\n"
            "separate: detach a part from its assembly. args: {part: str}\n"
        )
    elif compact_level == 1:
        system = (
            "Robot scene observer. Images are ordered oldest->newest.\n"
            "Use semantic names from context. Never use object_1 style IDs.\n"
            "Return JSON only, with keys in this order: reply, belief_state_update, stm_observation, action.\n"
            "reply: <=25 words, two short sentences (visual evidence + belief conclusion).\n"
            "belief_state_update.objects fields: belief_status, confidence, inferred_container.\n"
            "belief_state_update.static_context: include only if structural/background changed.\n"
            "Containment rule: table/floor are surfaces (ON), never containers (INSIDE).\n"
        )
    else:
        system = (
            "Return compact JSON only: reply, belief_state_update, stm_observation, action.\n"
            "Use semantic object names from context; no generic IDs.\n"
            "reply <=20 words, evidence then belief conclusion.\n"
            "Do not set inferred_container to table/floor.\n"
        )

    # Build JSON shape hint with runtime-discovered GT names (not hardcoded).
    # reply is listed FIRST so the model writes it before stm_observation.
    hint_objects: Dict[str, Any] = {}
    if gt_names:
        for name in gt_names[:hint_max_objs]:
            hint_objects[name] = {
                "belief_status": "<visible|moving|contained|occluded|unknown>",
                "confidence": 0.85,
                "inferred_container": "",
            }
    else:
        hint_objects["<object_name>"] = {
            "belief_status": "<visible|moving|contained|occluded|unknown>",
            "confidence": 0.85,
            "inferred_container": "",
        }

    if is_init_mode:
        json_shape_hint = {
            "belief_state_update": {
                "objects": {"<required_key>": {}},
                "static_context": {},
            },
            "stm_observation": "<<=30 words>",
            "reply": "<<=12 words>",
            "action": {"type": "noop", "args": {}},
        }
    elif is_auto_gt:
        json_shape_hint = {
            "belief_state_update": {
                "objects": hint_objects,
                "static_context": {},
            },
            "stm_observation": "<≤12 words literal>",
            "reply": "<≤15 words short confirmation>",
            "action": {"type": "noop"},
        }
    else:
        # reply key is intentionally first — preserves field-order hinting for the model.
        json_shape_hint = {
            "reply": "<≤25 words: visibility sentence + belief sentence>",
            "belief_state_update": {
                "objects": hint_objects,
                "static_context": {},
            },
            "stm_observation": "<≤12 words literal>",
            "action": {"type": "noop"},
        }

    required_objects: Dict[str, Dict[str, Any]] = {}
    for name in init_required_names:
        key = str(name or "").strip()
        if key:
            required_objects[key] = {}
    required_keys_json = json.dumps(list(required_objects.keys()), ensure_ascii=True, separators=(",", ":")) if required_objects else ""
    required_form_json = (
        json.dumps(
            {"belief_state_update": {"objects": required_objects, "static_context": {}}},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if required_objects
        else ""
    )

    if is_init_mode:
        if compact_level <= 1:
            user_msg = (
                f"{user_text}\n"
                f"Images: {num_images} (static)\n"
                f"{no_visual_guard}"
                "Fill RequiredForm object keys only. Do not add/remove keys.\n"
                "INTERACTABLE objects -> belief_state_update.objects.\n"
                "STRUCTURAL/background objects -> belief_state_update.static_context.\n"
                "Keep stm_observation/reply concise, no repetition.\n"
                + (f"RequiredForm:\n{required_form_json}\n" if required_form_json else "")
                + f"JSON skeleton:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
            )
        else:
            user_msg = (
                f"{user_text}\n"
                f"Images: {num_images} (static)\n"
                f"{no_visual_guard}"
                + (f"RequiredForm:{required_form_json}\n" if required_form_json else "")
                + "Do not change RequiredForm keys. Return strict JSON only.\n"
            )
    elif is_auto_gt and required_keys_json:
        changed_names = ", ".join(changed_objects[:8]) if changed_objects else "unknown"
        user_msg = (
            f"AUTO_TRIGGER=gt_change changed_objects={changed_names}\n"
            f"RequiredKeys={required_keys_json}\n"
            f"Images: {num_images} oldest->newest | motion={motion_score:.4f}\n"
            f"{no_visual_guard}"
            "Analyze frames against RequiredKeys one by one.\n"
            "Return belief_state_update.objects for CHANGED keys only. Omit unchanged keys entirely.\n"
            "Do not hallucinate changes.\n"
            "For changed keys, provide belief_status/confidence/inferred_container/temporal_change/visible.\n"
            f"JSON skeleton:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
        )
    elif has_existing_belief:
        if is_auto_gt:
            if not required_keys_json:
                print(
                    "[VLM] WARN: gt_change trigger missing all_interactables — "
                    "falling back to legacy sparse prompt (no RequiredKeys contract)",
                    flush=True,
                )
            prev_names = ", ".join(list(prev_compact.get("objects", {}).keys())[:belief_max_objs]) or "(none)"
            bg_names = ", ".join(list(prev_static_compact.get("static_context", {}).keys())[:static_max_objs]) or "(none)"
            changed_names = ", ".join(changed_objects[:6]) if changed_objects else "unknown"
            user_msg = (
                f"AUTO_TRIGGER=gt_change changed_objects={changed_names}\n"
                f"Active targets: {prev_names}\n"
                f"Background (ignore unless moving): {bg_names}\n"
                f"Images: {num_images} oldest->newest | motion={motion_score:.4f}\n"
                f"{no_visual_guard}"
                "Primary task: fill belief_state_update first with per-object temporal_change.\n"
                "Use directional transitions when visible (e.g., descending/falling, entered container, became occluded).\n"
                f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
            )
        else:
            memory_block = ""
            if stm_compact:
                memory_block += f"STM SUMMARY: {stm_compact}\n"
            if ltm_compact:
                memory_block += "RECENT MEMORY: " + " || ".join(ltm_compact) + "\n"
            if compact_level >= 2:
                prev_names = ", ".join(list(prev_compact.get("objects", {}).keys())[:3]) or "(none)"
                bg_names = ", ".join(list(prev_static_compact.get("static_context", {}).keys())[:3]) or "(none)"
                user_msg = (
                    f"Prev tracked objects: {prev_names}\n"
                    f"Prev background objects: {bg_names}\n"
                    f"{memory_block}"
                    f"{user_text}\n"
                    f"Images: {num_images} | motion={motion_score:.3f}\n"
                    f"{no_visual_guard}"
                    "Prioritize temporal change between oldest and newest frame.\n"
                    "Keep output compact and valid JSON.\n"
                    f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
                )
            else:
                user_msg = (
                    f"Here is what you believed before: {json.dumps(prev_compact, ensure_ascii=True)}\n"
                    f"Here is the background: {json.dumps(prev_static_compact, ensure_ascii=True)}\n"
                    f"{memory_block}"
                    "Look at the video. What changed? Update the belief state (position, occlusion, containment).\n"
                    f"{user_text}\n"
                    f"Images: {num_images} | motion={motion_score:.3f}\n"
                    f"{no_visual_guard}"
                    "REPLY FORMAT (≤25 words, reply key first): two sentences: "
                    "(1) direct visual evidence from this frame sequence "
                    "(2) belief conclusion using prior state plus current evidence. "
                    "If visibility is uncertain, say uncertain instead of asserting not-visible. "
                    "Update BELIEF only when frames show CHANGE. "
                    "Omit static_context if nothing structural changed.\n"
                    "CONTAINMENT: tables/floors are surfaces (ON), not containers (INSIDE). "
                    "Never set inferred_container to a table or floor prim.\n"
                    f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
                )
    else:
        memory_block = ""
        if stm_compact:
            memory_block += f"STM SUMMARY: {stm_compact}\n"
        if ltm_compact:
            memory_block += "RECENT MEMORY: " + " || ".join(ltm_compact) + "\n"
        if is_auto_gt:
            changed_names = ", ".join(changed_objects[:6]) if changed_objects else "unknown"
            user_msg = (
                f"AUTO_TRIGGER=gt_change changed_objects={changed_names}\n"
                "Active targets: (none yet)\n"
                "Background (ignore unless moving): (none)\n"
                f"{memory_block}"
                f"{user_text or 'Describe temporal changes and initialize active objects.'}\n"
                f"Images: {num_images} oldest->newest | motion={motion_score:.4f}\n"
                f"{no_visual_guard}"
                "Primary task: emit belief_state_update first with temporal_change per object.\n"
                f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
            )
        else:
            user_msg = (
                f"ACTIVE TRACKING TARGETS: (none yet)\n"
                f"BACKGROUND ENVIRONMENT (IGNORE UNLESS MOVING):\n{json.dumps(prev_static_compact, ensure_ascii=True)}\n"
                f"{memory_block}"
                f"{user_text or 'Describe the scene.'}\n"
                f"Images: {num_images} | motion={motion_score:.4f}\n"
                f"{no_visual_guard}"
                "Identify interactable objects -> objects, structural -> static_context.\n"
                f"JSON:\n{json.dumps(json_shape_hint, ensure_ascii=True)}\n"
            )

    parts = [{"type": "text", "text": user_msg}]
    parts.extend(frames_to_parts(frames))

    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


def _estimate_request_budget(messages: Sequence[dict], *, max_tokens: int = 512) -> Dict[str, int]:
    """Rough prompt budget estimate matching call_reason2() logic."""
    model_max_context = max(256, int(COSMOS_MAX_MODEL_LEN))
    safety_margin = 20
    tokens_per_image = 280

    num_images = 0
    text_chars = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_chars += len(str(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        num_images += 1
    estimated_text_tokens = text_chars // 5
    estimated_input_tokens = estimated_text_tokens + (num_images * tokens_per_image)
    available_tokens = model_max_context - estimated_input_tokens - safety_margin
    adjusted_max_tokens = min(int(max_tokens), max(16, int(available_tokens)))
    return {
        "num_images": int(num_images),
        "estimated_text_tokens": int(estimated_text_tokens),
        "estimated_input_tokens": int(estimated_input_tokens),
        "available_tokens": int(available_tokens),
        "adjusted_max_tokens": int(adjusted_max_tokens),
    }


_RESERVED_BELIEF_KEYS = {"objects", "static_context", "scene", "meta", "task"}
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
    "rolling",
    "slide",
    "sliding",
    "swing",
    "swinging",
    "rotate",
    "rotating",
    "spin",
    "spinning",
    "tilt",
    "tilting",
    "shift",
    "shifted",
    "displaced",
    "changed",
)


def _contains_any(text: str, words: Tuple[str, ...]) -> bool:
    for w in words:
        if w in text:
            return True
    return False


def _canon_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _is_dynamic_signal(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    status = str(obj.get("belief_status", "")).lower().strip()
    if status in {"moving", "contained"}:
        return True
    temporal = str(obj.get("temporal_change", "")).lower()
    if _contains_any(temporal, _MOTION_WORDS):
        return True
    for key in ("displacement", "max_displacement", "delta"):
        try:
            if float(obj.get(key, 0.0)) > 1e-4:
                return True
        except Exception:
            pass
    return False


def _normalize_static_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(obj) if isinstance(obj, dict) else {}
    status = str(out.get("belief_status", "static") or "static").lower().strip()
    if status in {"moving", "contained"}:
        status = "static"
    out["belief_status"] = status if status else "static"
    out["visible"] = bool(out.get("visible", True))
    try:
        out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.7))))
    except Exception:
        out["confidence"] = 0.7
    tc = str(out.get("temporal_change", "") or "").strip()
    out["temporal_change"] = tc if tc else "static in sequence"
    out["stale"] = bool(out.get("stale", False))
    return out


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
    "object",
    "objects",
    "visible",
    "static",
    "moving",
    "contained",
}


def _extract_scene_objects_from_text(reply: str) -> List[str]:
    """Extract candidate object names from free-form model reply without scene-specific mappings."""
    text = str(reply or "").lower()
    seen = set()
    names: List[str] = []

    # Most common noun phrases in model replies.
    candidates: List[str] = []
    candidates.extend(re.findall(r"\b(?:a|an|the)\s+([a-z][a-z0-9_-]{2,})\b", text))
    candidates.extend(re.findall(r"\b([a-z][a-z0-9_-]{2,})\s+(?:is|are|was|were)\b", text))

    for cand in candidates:
        name = str(cand or "").strip("_- ").lower()
        if not name or name in _GENERIC_OBJECT_STOPWORDS:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= 8:
            break
    return names


def _dedupe_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names:
        key = str(name or "").strip()
        if not key:
            continue
        c = _canon_name(key)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(key)
    return out


def _extract_allowed_object_names(user_text: str, prev_belief_json: Dict[str, Any]) -> List[str]:
    """Collect stable object naming hints from GT context + existing belief snapshot."""
    names: List[str] = []
    text = str(user_text or "")
    for line in text.splitlines():
        if "Known scene object names" not in line:
            continue
        if ":" not in line:
            continue
        tail = line.split(":", 1)[1]
        tail = tail.split("Use these names", 1)[0]
        for part in tail.split(","):
            n = str(part or "").strip().strip(".")
            if n:
                names.append(n)

    if isinstance(prev_belief_json, dict):
        objs = prev_belief_json.get("objects")
        if isinstance(objs, dict):
            names.extend([str(k) for k in objs.keys() if str(k or "").strip()])
        static_ctx = prev_belief_json.get("static_context")
        if isinstance(static_ctx, dict):
            names.extend([str(k) for k in static_ctx.keys() if str(k or "").strip()])

    return _dedupe_names(names)


def _resolve_allowed_name(name: str, allowed_names: Sequence[str]) -> str:
    """Resolve model-emitted name to a known GT/belief name; return empty string when ambiguous."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    allowed = _dedupe_names(allowed_names)
    if not allowed:
        return raw

    canon = _canon_name(raw)
    if not canon:
        return ""

    canon_to_name = {_canon_name(n): n for n in allowed if _canon_name(n)}
    if canon in canon_to_name:
        return canon_to_name[canon]

    contains_matches = [n for n in allowed if canon in _canon_name(n) or _canon_name(n) in canon]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1:
        # Ambiguous across instances (e.g., socket_bolt_hub_1..6). Keep strictness.
        return ""

    cand_tokens = set(re.findall(r"[a-z0-9]+", raw.lower()))
    if not cand_tokens:
        return ""

    best_name = ""
    best_score = 0.0
    tie = False
    for n in allowed:
        tokens = set(re.findall(r"[a-z0-9]+", n.lower()))
        overlap = len(cand_tokens & tokens)
        if overlap <= 0:
            continue
        jaccard = overlap / max(1, len(cand_tokens | tokens))
        score = float(overlap) + jaccard
        if score > best_score + 1e-9:
            best_name = n
            best_score = score
            tie = False
        elif abs(score - best_score) <= 1e-9:
            tie = True
    if tie:
        return ""
    return best_name


def _dedupe_adjacent_sentences(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", value) if p.strip()]
    if not parts:
        return value
    out: List[str] = []
    seen = set()
    for part in parts:
        key = _canon_name(part)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(part)
    return " ".join(out).strip()


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
    # Detect containment language generically — any "in/inside/into <noun>" pattern.
    containment_match = re.search(r"\b(in|inside|into)\b\s+(?:the\s+|a\s+)?(\w+)", text)
    has_contained = containment_match is not None
    container_name = containment_match.group(2) if containment_match else ""
    object_names = _extract_scene_objects_from_text(reply)

    primary_dynamic = object_names[0] if object_names else "moving_object_1"

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
            container = container_name

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
        inferred["unknown_object"] = {
            "belief_status": "moving" if has_motion else "visible",
            "visible": True,
            "confidence": 0.42 if has_motion else 0.36,
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

    dynamic_hint = extracted[0] if extracted else ""

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
    tracks: Dict[str, Dict[str, Any]], *, motion_score: float, temporal_text: str, fallback_name: str = "moving_object_1"
) -> Dict[str, Dict[str, Any]]:
    """Guarantee at least one dynamic object when diagnostics indicate motion."""
    out = copy.deepcopy(tracks if isinstance(tracks, dict) else {})
    if float(motion_score) < MOTION_DETECTED_THRESHOLD:
        return out

    # If any object already has a dynamic state (moving/contained), no promotion needed.
    _dynamic_states = {"moving", "contained"}
    for _name, _obj in out.items():
        if isinstance(_obj, dict) and str(_obj.get("belief_status", "")).lower() in _dynamic_states:
            return out

    candidate = ""
    for name in out.keys():
        candidate = name
        break
    if not candidate and out:
        candidate = next(iter(out.keys()))
    if not candidate:
        candidate = str(fallback_name or "").strip()
    if not candidate:
        return out

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
    raw: Dict[str, Any], *, prev_belief_json: Dict[str, Any], num_images: int, motion_score: float = 0.0,
    allowed_object_names: Sequence[str] | None = None,
    ensure_all_allowed_objects: bool = False,
    init_contract: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    allowed_names = _dedupe_names(list(allowed_object_names or []))
    enforce_allowed_names = bool(allowed_names)
    is_init_contract = bool(init_contract)

    # --- Three-Stage Cognitive Process: extract STM observation ---
    stm_observation = _dedupe_adjacent_sentences(str(out.get("stm_observation", "")).strip())

    reply = _dedupe_adjacent_sentences(str(out.get("reply") or "").strip())

    # If reply is empty, generate one from belief objects for better UX
    if not reply:
        # Check both new and legacy field names for the belief source.
        _bu = out.get("belief_state_update") or out.get("belief_update") or {}
        if isinstance(_bu, dict):
            objects = _bu.get("objects", {})
            if isinstance(objects, dict) and objects:
                obj_descriptions = []
                for name, obj in list(objects.items())[:3]:
                    if isinstance(obj, dict):
                        status = obj.get("belief_status", "unknown")
                        obj_descriptions.append(f"{name}: {status}")
                if obj_descriptions:
                    reply = "Observing: " + ", ".join(obj_descriptions)
        if not reply:
            reply = stm_observation if stm_observation else (
                f"Processing {num_images} frames" if num_images > 0 else "Scene observed"
            )

    action = out.get("action")
    if not isinstance(action, dict):
        action = {"type": "noop", "args": {}}
    action.setdefault("type", "noop")
    if not isinstance(action.get("args"), dict):
        action["args"] = {}
    a_type = str(action.get("type") or "").strip().lower()
    if a_type == "combine":
        raw_args = action.get("args") or {}
        part_a = str(
            raw_args.get("partA")
            or raw_args.get("child")
            or raw_args.get("child_name")
            or ""
        ).strip()
        part_b = str(
            raw_args.get("partB")
            or raw_args.get("parent")
            or raw_args.get("parent_name")
            or ""
        ).strip()
        plug = str(raw_args.get("plug") or raw_args.get("plug_name") or "").strip()
        socket = str(raw_args.get("socket") or raw_args.get("socket_name") or "").strip()
        if part_a and part_b and plug and socket:
            action["args"] = {
                "partA": part_a,
                "partB": part_b,
                "plug": plug,
                "socket": socket,
            }
        else:
            # Reject incomplete combine actions early; runtime pipeline now
            # requires explicit 4-parameter combine.
            action = {"type": "noop", "args": {}}
    elif a_type == "separate":
        raw_args = action.get("args") or {}
        part = str(raw_args.get("part") or raw_args.get("part_name") or "").strip()
        action["args"] = {"part": part} if part else {}
        if not part:
            action = {"type": "noop", "args": {}}

    # Accept both new field name (belief_state_update) and legacy (belief_update).
    belief_update = out.get("belief_state_update") or out.get("belief_update")
    if isinstance(belief_update, str):
        try:
            belief_update = json.loads(belief_update)
        except Exception:
            belief_update = {}
    if not isinstance(belief_update, dict):
        belief_update = {}

    prev_objects = {}
    prev_static_context = {}
    if isinstance(prev_belief_json, dict):
        p = prev_belief_json.get("objects")
        if isinstance(p, dict):
            prev_objects = copy.deepcopy(p)
        ps = prev_belief_json.get("static_context")
        if isinstance(ps, dict):
            prev_static_context = copy.deepcopy(ps)

    incoming_objects: Dict[str, Dict[str, Any]] = {}
    incoming_static_context: Dict[str, Dict[str, Any]] = {}

    objects_block = belief_update.get("objects")
    static_block = belief_update.get("static_context")
    if static_block is None:
        static_block = out.get("static_context")
    if isinstance(objects_block, str):
        try:
            objects_block = json.loads(objects_block)
        except Exception:
            objects_block = {}
    if isinstance(static_block, str):
        try:
            static_block = json.loads(static_block)
        except Exception:
            static_block = {}
    if isinstance(objects_block, dict):
        for name, value in objects_block.items():
            parsed = _coerce_object_entry(value)
            if parsed is not None:
                if not parsed and not is_init_contract:
                    continue
                resolved_name = _resolve_allowed_name(str(name), allowed_names) if enforce_allowed_names else str(name)
                if not resolved_name:
                    continue
                incoming_objects[resolved_name] = parsed
    if isinstance(static_block, dict):
        for name, value in static_block.items():
            parsed = _coerce_object_entry(value)
            if parsed is not None:
                resolved_name = _resolve_allowed_name(str(name), allowed_names) if enforce_allowed_names else str(name)
                if not resolved_name:
                    continue
                incoming_static_context[resolved_name] = parsed
    elif isinstance(static_block, list):
        for name in static_block:
            key = str(name or "").strip()
            if key:
                resolved_name = _resolve_allowed_name(key, allowed_names) if enforce_allowed_names else key
                if not resolved_name:
                    continue
                incoming_static_context[resolved_name] = _normalize_static_entry({})

    for key, value in belief_update.items():
        if key in _RESERVED_BELIEF_KEYS:
            continue
        parsed = _coerce_object_entry(value)
        if parsed is not None and _looks_like_object_entry(parsed):
            if not parsed and not is_init_contract:
                continue
            resolved_name = _resolve_allowed_name(str(key), allowed_names) if enforce_allowed_names else str(key)
            if not resolved_name:
                continue
            incoming_objects[resolved_name] = parsed

    for key, value in out.items():
        if key in {"reply", "action", "belief_update", "belief_state_update", "static_context", "meta"}:
            continue
        parsed = _coerce_object_entry(value)
        if parsed is not None and _looks_like_object_entry(parsed):
            if not parsed and not is_init_contract:
                continue
            resolved_name = _resolve_allowed_name(str(key), allowed_names) if enforce_allowed_names else str(key)
            if not resolved_name:
                continue
            incoming_objects[resolved_name] = parsed

    if not incoming_objects and reply:
        inferred = _infer_objects_from_reply(reply, motion_score=float(motion_score))
        if enforce_allowed_names:
            filtered: Dict[str, Dict[str, Any]] = {}
            for name, obj in inferred.items():
                resolved_name = _resolve_allowed_name(str(name), allowed_names)
                if not resolved_name:
                    continue
                filtered[resolved_name] = obj
            incoming_objects = filtered
        else:
            incoming_objects = inferred

    normalized_objects: Dict[str, Dict[str, Any]] = {}
    normalized_static_context: Dict[str, Dict[str, Any]] = {}

    # Keep previously tracked objects unless the new update explicitly overwrites them.
    for name, prev in prev_objects.items():
        if isinstance(prev, dict):
            normalized_objects[str(name)] = copy.deepcopy(prev)
    for name, prev in prev_static_context.items():
        if isinstance(prev, dict):
            normalized_static_context[str(name)] = _normalize_static_entry(prev)

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
        prev_status = str(cur.get("belief_status") or "").strip().lower()
        if status in {"falling", "dropping"}:
            status = "moving"
        if is_init_contract and status == "unknown":
            status = "present"
        # For single-frame/manual follow-ups, keep prior "contained" state unless there is
        # strong fresh motion evidence.
        if (
            prev_status == "contained"
            and status in {"moving", "unknown"}
            and int(num_images) <= 1
            and float(motion_score) < MOTION_DETECTED_THRESHOLD
        ):
            status = "contained"
            if not str(merged.get("inferred_container", "")).strip():
                merged["inferred_container"] = str(cur.get("inferred_container", "")).strip()
            merged["stale"] = False
        if status == "unknown" and float(motion_score) >= MOTION_DETECTED_THRESHOLD and _contains_any(
            reply.lower(), _MOTION_WORDS
        ):
            status = "moving"
            if not str(merged.get("temporal_change") or "").strip():
                merged["temporal_change"] = "temporal change detected from sequence"
        # Keep container continuity: if model omits inferred_container on follow-ups,
        # retain previously known container for contained/occluded/unknown states.
        prev_container = str(cur.get("inferred_container", "") or "").strip()
        cur_container = str(merged.get("inferred_container", "") or "").strip()
        if (not cur_container) and prev_container and status in {"contained", "occluded", "unknown"}:
            cur_container = prev_container

        # Reject surface labels as containers.
        surface_like = {
            "surface",
            "onasurface",
            "onsurface",
            "table",
            "floor",
            "ground",
            "groundplane",
            "groundsuface",
            "groundsurface",
        }
        if _canon_name(cur_container) in surface_like:
            cur_container = prev_container if prev_container else ""

        # contained must not end with an empty container field in normalized output.
        if status == "contained" and not cur_container:
            cur_container = prev_container if prev_container else "unknown_container"

        # Init grounding should not invent deep containment chains.
        if is_init_contract:
            if status != "contained":
                cur_container = ""
            elif _canon_name(cur_container) == _canon_name(name):
                cur_container = ""

        merged["inferred_container"] = cur_container
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
        if is_init_contract:
            merged["temporal_change"] = "initial_state"
        normalized_objects[name] = merged

    for name, obj in incoming_static_context.items():
        if not isinstance(obj, dict):
            continue
        if _is_dynamic_signal(obj):
            promoted = copy.deepcopy(normalized_objects.get(name, {}))
            if not isinstance(promoted, dict):
                promoted = {}
            promoted.update(copy.deepcopy(obj))
            if str(promoted.get("belief_status", "")).lower() == "static":
                promoted["belief_status"] = "moving"
            normalized_objects[name] = promoted
            normalized_static_context.pop(name, None)
            continue
        cur = normalized_static_context.get(name, {})
        if not isinstance(cur, dict):
            cur = {}
        merged = copy.deepcopy(cur)
        merged.update(copy.deepcopy(obj))
        normalized_static_context[name] = _normalize_static_entry(merged)

    # Initialization contract: keep a complete interactable object form from GT names.
    backfilled_keys: list[str] = []
    if ensure_all_allowed_objects and enforce_allowed_names:
        for name in allowed_names:
            key = str(name or "").strip()
            if not key:
                continue
            if key in normalized_objects or key in normalized_static_context:
                continue
            backfilled_keys.append(key)
            normalized_objects[key] = {
                "belief_status": "present" if is_init_contract else "unknown",
                "visible": False,
                "confidence": 0.35 if is_init_contract else 0.0,
                "inferred_container": "",
                "temporal_change": "initial_state" if is_init_contract else "unknown",
                "stale": False if is_init_contract else True,
                "source": "gt_seed" if is_init_contract else "gt_required",
            }
    if backfilled_keys:
        total_required = sum(1 for n in allowed_names if str(n or "").strip())
        ratio = len(backfilled_keys) / max(1, total_required)
        level = "WARN" if ratio > 0.5 else "INFO"
        print(
            f"[VLM] {level}: backfilled {len(backfilled_keys)}/{total_required} required keys "
            f"({ratio:.0%}) not returned by model: {backfilled_keys[:8]}",
            flush=True,
        )

    # Wake-up: if a background object is now dynamic, promote it to active tracking.
    for name, obj in list(normalized_static_context.items()):
        if not isinstance(obj, dict):
            continue
        if _is_dynamic_signal(obj):
            promoted = copy.deepcopy(obj)
            if str(promoted.get("belief_status", "")).lower() == "static":
                promoted["belief_status"] = "moving"
            normalized_objects[name] = promoted
            normalized_static_context.pop(name, None)

    # Derive meaningful temporal_change from belief_status when model doesn't provide it.
    # The new schema asks the model for stm_observation (global) instead of per-object
    # temporal_change, so we derive it from the belief state for downstream consumers.
    for _tc_name, _tc_obj in normalized_objects.items():
        if not isinstance(_tc_obj, dict):
            continue
        tc = str(_tc_obj.get("temporal_change", "")).strip()
        if tc and tc.lower() not in ("unknown", "none", ""):
            continue  # Model provided a meaningful value; keep it.
        _tc_status = str(_tc_obj.get("belief_status", "unknown")).lower()
        _tc_container = str(_tc_obj.get("inferred_container", "")).strip()
        if _tc_status == "contained" and _tc_container:
            _tc_obj["temporal_change"] = f"inside {_tc_container}"
        elif _tc_status == "moving":
            _tc_obj["temporal_change"] = "in motion"
        elif _tc_status == "visible":
            _tc_obj["temporal_change"] = "stationary"
        elif _tc_status == "occluded":
            _tc_obj["temporal_change"] = "not visible"
        else:
            _tc_obj["temporal_change"] = "status unknown"

    if not normalized_objects and not normalized_static_context and not enforce_allowed_names:
        normalized_objects = _infer_objects_from_reply(reply, motion_score=float(motion_score))

    if normalized_objects and not enforce_allowed_names:
        normalized_objects = _ensure_minimum_tracks(
            normalized_objects, reply=reply, motion_score=float(motion_score), min_tracks=1
        )

    # If diagnostics indicate motion, force a dynamic anchor even when model output is weak.
    temporal_hint = ""
    low_reply = str(reply or "").lower()
    if _contains_any(low_reply, _MOTION_WORDS):
        temporal_hint = "position changed across sequence"
    if normalized_objects or (not normalized_static_context and not enforce_allowed_names):
        normalized_objects = _promote_dynamic_track(
            normalized_objects,
            motion_score=float(motion_score),
            temporal_text=temporal_hint,
            fallback_name="" if enforce_allowed_names else "moving_object_1",
        )

    # If language indicates the object ended up inside a container, enforce a contained state.
    low_reply = str(reply or "").lower()
    m_contained = re.search(r"\b(?:inside|into|in)\s+(?:the\s+|a\s+|an\s+)?([a-z0-9_ -]{2,40})", low_reply)
    if m_contained:
        container_hint = str(m_contained.group(1) or "").strip(" .,:;")
        container_canon = _canon_name(container_hint)
        if container_canon and container_canon not in _GENERIC_OBJECT_STOPWORDS:
            # Skip if a contained track already exists.
            has_contained = any(
                isinstance(v, dict) and str(v.get("belief_status", "")).lower() == "contained"
                for v in normalized_objects.values()
            )
            if not has_contained:
                target_name = None
                # Prefer moving/unknown non-container objects.
                for name, obj in normalized_objects.items():
                    if not isinstance(obj, dict):
                        continue
                    if _canon_name(name) == container_canon:
                        continue
                    s = str(obj.get("belief_status", "")).lower()
                    if s in {"moving", "unknown"}:
                        target_name = name
                        break
                if target_name is None:
                    for name, obj in normalized_objects.items():
                        if not isinstance(obj, dict):
                            continue
                        if _canon_name(name) == container_canon:
                            continue
                        target_name = name
                        break
                if target_name is not None:
                    obj = normalized_objects.get(target_name, {})
                    if isinstance(obj, dict):
                        obj["belief_status"] = "contained"
                        obj["inferred_container"] = container_hint
                        obj["visible"] = bool(obj.get("visible", True))
                        cur_change = str(obj.get("temporal_change", "")).strip()
                        if not cur_change or cur_change.lower() in {"unknown", "static", "static in sequence"}:
                            obj["temporal_change"] = "likely moved into container based on sequence"
                        obj["stale"] = False
                        normalized_objects[target_name] = obj

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
        if not has_dynamic_track and (normalized_objects or (not normalized_static_context and not enforce_allowed_names)):
            if enforce_allowed_names:
                candidate_name = next(iter(normalized_objects.keys()), "")
                if not candidate_name:
                    candidate_name = ""
                if candidate_name:
                    normalized_objects[candidate_name] = {
                        **copy.deepcopy(normalized_objects.get(candidate_name, {})),
                        "belief_status": "moving",
                        "visible": True,
                        "confidence": max(
                            0.45,
                            float(normalized_objects.get(candidate_name, {}).get("confidence", 0.0) or 0.0),
                        ),
                        "inferred_container": str(normalized_objects.get(candidate_name, {}).get("inferred_container", "") or ""),
                        "temporal_change": "temporal change detected from frame sequence",
                        "stale": False,
                    }
            else:
                normalized_objects["moving_object_1"] = {
                    "belief_status": "moving",
                    "visible": True,
                    "confidence": 0.45,
                    "inferred_container": "",
                    "temporal_change": "temporal change detected from frame sequence",
                    "stale": False,
                }

    # If an object is actively tracked, remove duplicate background entry.
    for name, obj in list(normalized_objects.items()):
        if not isinstance(obj, dict):
            continue
        if _is_dynamic_signal(obj) or name in normalized_static_context:
            normalized_static_context.pop(name, None)

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
    elif normalized_objects or normalized_static_context:
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

    # If stm_observation is still empty, derive from reply as fallback.
    if not stm_observation:
        stm_observation = reply[:120] if reply else "no observation"

    return {
        "stm_observation": stm_observation,
        "reply": reply or "<empty response>",
        "action": action,
        "belief_update": {"objects": normalized_objects, "static_context": normalized_static_context},
        "meta": meta,
    }


def call_reason2(messages: List[dict], *, max_tokens: int = 512, temperature: float = 0.2) -> str:
    if not cosmos_is_configured():
        raise RuntimeError(
            "Cosmos is not configured. Set COSMOS_CHAT_COMPLETIONS_URL or COSMOS_BASE_URL to enable the optional Cosmos observer."
        )
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        raise RuntimeError("COSMOS_MODEL is not set.")

    endpoint = cosmos_endpoint()

    # Keep prompt-budget checks configurable to match server-side --max-model-len.
    MODEL_MAX_CONTEXT = max(256, int(COSMOS_MAX_MODEL_LEN))
    SAFETY_MARGIN = 20  # Reserve tokens for tokenization variance
    # Calibrated from actual run logs: (prompt_tokens - text_est) / num_images ≈ 245-289.
    # 280 is the calibrated midpoint; conservative enough to avoid overflow while
    # still allowing 5-image GT-change requests to get ~300 output tokens.
    TOKENS_PER_IMAGE = 280

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
    # Never force a large completion budget when input is near context limit.
    # This avoids server-side OOM/EngineCore failures on multimodal requests.
    adjusted_max_tokens = min(int(max_tokens), max(16, int(available_tokens)))

    if adjusted_max_tokens < max_tokens:
        print(
            f"[VLM] Adjusted max_tokens: {max_tokens} → {adjusted_max_tokens} "
            f"(ctx={MODEL_MAX_CONTEXT}, est_input={estimated_input_tokens}, "
            f"text={estimated_text_tokens}, imgs={num_images}x{TOKENS_PER_IMAGE}, available={available_tokens})",
            flush=True,
        )

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
    trigger: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """High-level call used by the agent graph.

    Returns parsed JSON dict (stm_observation/reply/action/belief_update/meta).
    """
    motion_features = _estimate_motion_features(frames)
    motion_score = float(motion_features.get("motion_score", 0.0))
    trigger_dict = trigger if isinstance(trigger, dict) else {}
    trigger_type = str(trigger_dict.get("type") or "").strip().lower()
    is_init_mode = "INITIALIZATION MODE" in str(user_text or "")
    ensure_all_allowed = bool(is_init_mode or trigger_type == "gt_change")
    allowed_object_names = _extract_allowed_object_names(user_text, prev_belief_json)
    raw_init_names = trigger_dict.get("init_interactables")
    if isinstance(raw_init_names, list):
        allowed_object_names = _dedupe_names(
            list(allowed_object_names) + [str(n) for n in raw_init_names if str(n or "").strip()]
        )
    raw_all_names = trigger_dict.get("all_interactables")
    if isinstance(raw_all_names, list):
        allowed_object_names = _dedupe_names(
            list(allowed_object_names) + [str(n) for n in raw_all_names if str(n or "").strip()]
        )
    raw_init_form = trigger_dict.get("init_belief_form")
    if isinstance(raw_init_form, dict):
        belief_block = raw_init_form.get("belief_state_update")
        if isinstance(belief_block, dict):
            form_objects = belief_block.get("objects")
            if isinstance(form_objects, dict):
                allowed_object_names = _dedupe_names(
                    list(allowed_object_names) + [str(k) for k in form_objects.keys() if str(k or "").strip()]
                )
    # Keep initialization responses bounded to reduce latency and timeout risk.
    request_max_tokens = 384 if is_init_mode else 512

    def _is_overlength_error(message: str) -> bool:
        msg_l = str(message or "").lower()
        return bool(
            "maximum model length" in msg_l
            or "maximum context length" in msg_l
            or "decoder prompt (length" in msg_l
            or ("prompt" in msg_l and "longer than" in msg_l and "max" in msg_l)
            or ("input tokens" in msg_l and "reduce the length of the input" in msg_l)
        )

    def _fallback_from_text(raw_text: str, parse_error: str, *, num_images_hint: int | None = None) -> Dict[str, Any]:
        raw_text = (raw_text or "").strip()
        print(f"[VLM] FALLBACK: parse_error={parse_error} | reply_preview={raw_text[:200]}", flush=True)

        # Try to extract stm_observation from partial/unterminated JSON.
        stm = ""
        reply_text = ""
        _stm_match = re.search(r'"stm_observation"\s*:\s*"((?:[^"\\]|\\.)*)', raw_text)
        if _stm_match:
            stm = _stm_match.group(1).strip()
        _reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', raw_text)
        if _reply_match:
            reply_text = _reply_match.group(1).strip()

        # If we extracted stm but no reply, derive a short reply from stm.
        if stm and not reply_text:
            # Truncate stm to first sentence for the reply.
            first_sent = stm.split(".")[0].strip()
            reply_text = first_sent + "." if first_sent else stm[:120]
        elif not reply_text:
            reply_text = raw_text[:200] if raw_text else "<empty response>"

        # Build a scene summary for meta (not the full raw dump).
        scene_summary = stm[:200] if stm else reply_text[:200]

        fallback = {
            "stm_observation": stm,
            "reply": reply_text,
            "action": {"type": "noop", "args": {}},
            "belief_update": {},
            "meta": {
                "num_images": int(num_images_hint if num_images_hint is not None else len(frames)),
                "temporal_summary": "fallback_text_mode",
                "parse_error": str(parse_error),
                "motion_score": float(motion_score),
                "scene_summary": scene_summary,
            },
            "_timing_sec": round(time.time() - t0, 3),
        }
        return _normalize_model_output(
            fallback,
            prev_belief_json=prev_belief_json,
            num_images=int(num_images_hint if num_images_hint is not None else len(frames)),
            motion_score=float(motion_score),
            allowed_object_names=allowed_object_names,
            ensure_all_allowed_objects=ensure_all_allowed,
            init_contract=is_init_mode,
        )

    t0 = time.time()
    try:
        # First attempt: multimodal (frames included).
        used_frames = list(frames)
        if trigger_type == "gt_change":
            used_compact_level = 3 if len(used_frames) >= 3 else 2
        else:
            used_compact_level = 2 if len(used_frames) >= 5 else 0

        def _build_messages(_frames: Sequence[Any], _compact_level: int = 0) -> List[dict]:
            return build_reason2_messages(
                frames=_frames,
                user_text=user_text,
                short_memory_summary=short_memory_summary,
                long_memory_snippets=long_memory_snippets,
                prev_belief_json=prev_belief_json,
                motion_features=motion_features,
                prompt_compact_level=_compact_level,
                trigger=trigger_dict,
            )

        # Preflight: compact prompt text first when completion budget is too tight.
        candidate_levels = [used_compact_level]
        if used_compact_level < 3:
            candidate_levels.extend([1, 2, 3, 4])
        seen_levels = set()
        ordered_levels: List[int] = []
        for lvl in candidate_levels:
            ilvl = max(0, int(lvl))
            if ilvl in seen_levels:
                continue
            seen_levels.add(ilvl)
            ordered_levels.append(ilvl)

        messages = _build_messages(used_frames, used_compact_level)
        budget = _estimate_request_budget(messages, max_tokens=request_max_tokens)
        if budget.get("adjusted_max_tokens", 16) < MIN_COMPLETION_TOKENS:
            for lvl in ordered_levels:
                trial_messages = _build_messages(used_frames, lvl)
                trial_budget = _estimate_request_budget(trial_messages, max_tokens=request_max_tokens)
                if trial_budget.get("adjusted_max_tokens", 16) >= MIN_COMPLETION_TOKENS or lvl == ordered_levels[-1]:
                    used_compact_level = lvl
                    messages = trial_messages
                    budget = trial_budget
                    break
            print(
                f"[VLM] Budget preflight: images={len(used_frames)}, compact_level={used_compact_level}, "
                f"est_prompt={budget.get('estimated_input_tokens')}, est_max_tokens={budget.get('adjusted_max_tokens')}",
                flush=True,
            )
        try:
            content = call_reason2(messages, max_tokens=request_max_tokens, temperature=0.1)
        except Exception as exc:
            msg = str(exc)
            msg_l = msg.lower()
            recovered_from_overlength = False
            overlength_error = _is_overlength_error(msg)

            # Retry 1: same frames, compact prompt text first.
            if overlength_error:
                compact_recovered = False
                for compact_level in (1, 2, 3, 4):
                    if compact_level <= used_compact_level:
                        continue
                    print(
                        f"[VLM] Overlength retry: compacting prompt text (level={compact_level}, images={len(used_frames)})",
                        flush=True,
                    )
                    try:
                        content = call_reason2(
                            _build_messages(used_frames, compact_level), max_tokens=request_max_tokens, temperature=0.1
                        )
                        used_compact_level = compact_level
                        compact_recovered = True
                        recovered_from_overlength = True
                        break
                    except Exception as exc_retry:
                        retry_l = str(exc_retry).lower()
                        retry_overlength = _is_overlength_error(retry_l)
                        if retry_overlength:
                            continue
                        exc = exc_retry
                        msg = str(exc_retry)
                        msg_l = retry_l
                        break
                if compact_recovered:
                    overlength_error = False

            # Retry 2 (last resort): reduce frames when prompt still exceeds model context.
            if used_frames and overlength_error:
                recovered = False
                last_exc: Exception = exc
                for keep in range(len(used_frames) - 1, -1, -1):
                    candidate_frames = list(used_frames[-keep:]) if keep > 0 else []
                    print(f"[VLM] Overlength retry: reducing images to {len(candidate_frames)}", flush=True)
                    try:
                        content = call_reason2(
                            _build_messages(candidate_frames, max(1, used_compact_level)),
                            max_tokens=request_max_tokens,
                            temperature=0.1,
                        )
                        used_frames = candidate_frames
                        recovered = True
                        recovered_from_overlength = True
                        break
                    except Exception as exc_retry:
                        last_exc = exc_retry
                        retry_l = str(exc_retry).lower()
                        retry_overlength = _is_overlength_error(retry_l)
                        if retry_overlength:
                            continue
                        # Not a length error; handle below via existing fallback branches.
                        exc = exc_retry
                        msg = str(exc_retry)
                        msg_l = retry_l
                        break

                if not recovered and overlength_error:
                    # Keep original control-flow behavior (handled by outer exception path).
                    raise last_exc

            # Fallback: retry text-only if the endpoint rejects images.
            if recovered_from_overlength:
                pass
            elif used_frames and ("image" in msg_l) and ("at most 0" in msg_l or "may be provided" in msg_l):
                messages = _build_messages([])
                content = call_reason2(messages, max_tokens=request_max_tokens, temperature=0.1)
                try:
                    out = parse_json_response(content)
                except Exception as parse_exc:
                    out = _fallback_from_text(
                        content,
                        f"text-only fallback parse failed: {parse_exc}",
                        num_images_hint=0,
                    )
                    out.setdefault("meta", {})
                    if isinstance(out["meta"], dict):
                        out["meta"]["images_dropped"] = True
                        out["meta"]["num_images"] = 0
                    return out
                normalized = _normalize_model_output(
                    out,
                    prev_belief_json=prev_belief_json,
                    num_images=0,
                    motion_score=float(motion_score),
                    allowed_object_names=allowed_object_names,
                    ensure_all_allowed_objects=ensure_all_allowed,
                    init_contract=is_init_mode,
                )
                normalized.setdefault("meta", {})
                if isinstance(normalized["meta"], dict):
                    normalized["meta"]["images_dropped"] = True
                    normalized["meta"]["num_images"] = 0
                    normalized["meta"]["motion_score"] = float(motion_score)
                normalized["_timing_sec"] = round(time.time() - t0, 3)
                return normalized
            else:
                raise
        try:
            out = parse_json_response(content)
        except Exception as parse_exc:
            return _fallback_from_text(content, str(parse_exc), num_images_hint=len(used_frames))
        if not isinstance(out, dict):
            return _fallback_from_text(str(content), "non-dict JSON output", num_images_hint=len(used_frames))

        normalized = _normalize_model_output(
            out,
            prev_belief_json=prev_belief_json,
            num_images=int(len(used_frames)),
            motion_score=float(motion_score),
            allowed_object_names=allowed_object_names,
            ensure_all_allowed_objects=ensure_all_allowed,
            init_contract=is_init_mode,
        )
        normalized.setdefault("meta", {})
        elapsed_sec = round(time.time() - t0, 3)
        if isinstance(normalized["meta"], dict):
            normalized["meta"]["motion_score"] = float(motion_score)
        normalized["_timing_sec"] = elapsed_sec
        if is_init_mode and elapsed_sec > 30.0:
            print(
                f"[VLM] WARN: init grounding took {elapsed_sec:.1f}s "
                f"(images={len(used_frames)}, compact={used_compact_level}) — "
                "prompt compaction may be insufficient",
                flush=True,
            )
        return normalized
    except Exception as exc:
        # Never raise to worker loop on model parsing/formatting issues.
        import traceback
        print(f"[VLM] EXCEPTION in reason2_decide: {exc}", flush=True)
        traceback.print_exc()
        return _fallback_from_text("", f"reason2_decide exception: {exc}")
