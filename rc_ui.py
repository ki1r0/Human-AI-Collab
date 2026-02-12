from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List

import omni.kit.app
import omni.timeline
import omni.ui as ui

from rc_belief_manager import BeliefManager
from rc_camera import capture_rgb_uint8_async, init_camera, save_last_frame, save_sent_frames
from rc_cognitive_worker import cognitive_worker
from rc_config import (
    CAPTURE_FPS,
    COSMOS_MODEL,
    INQUIRY_FRAME_COUNT,
    INQUIRY_INTERVAL_SEC,
    MEM0_API_KEY,
    MIN_INFER_INTERVAL_SEC,
    STREAM_DEFAULT_PROMPT,
    TIMEOUT_SEC,
)
from rc_franka_control import FrankaControlPolicy
from rc_ghost_visualizer import GhostVisualizer
from rc_perception import StateMonitor
from rc_log import log_error, log_info, log_line, log_warn, render_log
from rc_long_term_memory import LongTermMemory
from rc_short_term_memory import ShortTermMemory
from rc_state import STATE
from rc_vlm import cosmos_endpoint, test_cosmos_connection


def _get_recent_frames(last_seconds: float, max_frames: int) -> List[Any]:
    """Return up to max_frames from the recent window, evenly spread oldest->newest."""
    if not STATE.frame_buffer:
        return []
    items = list(STATE.frame_buffer)  # (ts, rgb)
    now = time.time()
    cutoff = now - max(0.0, float(last_seconds))
    recent_items = [(ts, rgb) for ts, rgb in items if ts >= cutoff]
    if not recent_items:
        return []
    max_frames = max(1, int(max_frames))
    if len(recent_items) <= max_frames:
        return [rgb for _ts, rgb in recent_items]

    # Evenly sample across the window so motion is visible to the VLM.
    n = len(recent_items)
    step = (n - 1) / float(max_frames - 1)
    idxs = []
    for i in range(max_frames):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    # Keep index order monotonic and unique.
    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)
    return [recent_items[i][1] for i in dedup]


def _get_frames_since(last_ts: float, max_frames: int) -> tuple[List[Any], float]:
    """Return up to max_frames newer than last_ts, sampled oldest->newest.

    Includes one anchor frame from <= last_ts (when available) so each request has temporal continuity
    with the previous request.
    """
    if not STATE.frame_buffer:
        return [], float(last_ts)
    items = list(STATE.frame_buffer)  # (ts, rgb)
    newer_items = [(ts, rgb) for ts, rgb in items if float(ts) > float(last_ts)]
    if not newer_items:
        return [], float(last_ts)

    # Add one anchor frame to preserve change across request boundaries.
    anchor = None
    if last_ts > 0.0:
        for ts, rgb in reversed(items):
            if float(ts) <= float(last_ts):
                anchor = (ts, rgb)
                break
    if anchor is not None:
        newer_items = [anchor] + newer_items

    max_frames = max(1, int(max_frames))
    if len(newer_items) <= max_frames:
        return [rgb for _ts, rgb in newer_items], float(newer_items[-1][0])

    # Evenly sample across all newly captured frames to preserve motion over the whole wait period.
    n = len(newer_items)
    step = (n - 1) / float(max_frames - 1)
    idxs = []
    for i in range(max_frames):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)
    sampled = [newer_items[i][1] for i in dedup]
    return sampled, float(newer_items[-1][0])


def _log_long(prefix: str, text: str, chunk: int = 400) -> None:
    if not text:
        log_line(prefix, "<empty response>")
        return
    for i in range(0, len(text), int(chunk)):
        log_line(prefix, text[i : i + int(chunk)])


def _compact_belief_from_snapshot(snapshot: Dict[str, Any]) -> str:
    """Render a compact belief summary directly from belief snapshot."""
    if not isinstance(snapshot, dict):
        return "no objects tracked"
    objects = snapshot.get("objects")
    if not isinstance(objects, dict) or not objects:
        return "no objects tracked"
    parts: List[str] = []
    for name, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        status = str(obj.get("belief_status", "unknown"))
        stale = bool(obj.get("stale", False))
        conf = obj.get("confidence")
        container = str(obj.get("inferred_container", "")).strip()
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
    if not parts:
        return "no objects tracked"
    return " | ".join(parts)[:600]


def _configure_quiet_kit_logging() -> None:
    """Reduce known-noisy Kit warning channels for this demo.

    This does *not* suppress our own logs (we render them in the UI), it only reduces terminal spam from
    common rendering/perf warnings.
    """
    try:
        import carb

        s = carb.settings.get_settings()
        # Keep global level as-is; tune only the noisy channels.
        s.set("/log/channels/omni.hydra", "error")
        s.set("/log/channels/omni.syntheticdata.plugin", "error")
        s.set("/log/channels/rtx.postprocessing.plugin", "error")
        s.set("/log/channels/omni.fabric.plugin", "error")
        s.set("/log/channels/omni.kit.window.property.*", "error")
        s.set("/log/channels/semantics.schema.property.*", "error")
        # Audio warnings are common in containers.
        s.set("/log/channels/carb.audio.*", "error")
    except Exception:
        pass


def _ensure_asset_browser_cache_dir() -> None:
    """Prevent noisy FileNotFoundError from isaacsim.asset.browser cache writes in some container layouts."""
    try:
        cache_dir = "/workspace/isaacsim/_build/linux-x86_64/release/exts/isaacsim.asset.browser/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "isaacsim.asset.browser.cache.json")
        if not os.path.exists(cache_file):
            # Create a valid empty json file.
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("{}\n")
    except Exception:
        pass


def _init_pipeline_once() -> None:
    """Initialize cognition + memory + (optional) robot control once per session."""
    # Recreate LTM/worker when Mem0 key changes (common during interactive tuning).
    mem0_fp = (str(MEM0_API_KEY or "").strip() or "none")[-8:]
    need_worker_restart = False
    if getattr(STATE, "_ltm_key_fp", None) != mem0_fp:
        STATE._ltm_key_fp = mem0_fp
        if STATE.long_memory is not None:
            need_worker_restart = True
            STATE.long_memory = None
            log_line("INFO ", "Mem0 key change detected; reinitializing long-term memory.")

    if STATE.belief_manager is None:
        STATE.belief_manager = BeliefManager(initial_state={"objects": {}})
        log_line("INFO ", "BeliefManager initialized.")

    if STATE.short_memory is None:
        STATE.short_memory = ShortTermMemory(ttl_sec=6.0, logger=lambda m: log_line("INFO ", m))
        log_line("INFO ", "ShortTermMemory initialized.")

    if STATE.long_memory is None:
        STATE.long_memory = LongTermMemory(logger=lambda m: log_line("INFO ", m))
        log_line("INFO ", "LongTermMemory initialized.")

    if STATE.ghost_visualizer is None:
        # PXR-only (no omni.isaac.core dependency).
        STATE.ghost_visualizer = GhostVisualizer(logger=lambda m: log_line("INFO ", m))

    if STATE.robot_controller is None:
        # Safe to create even when no Franka is present; it self-disables and logs.
        STATE.robot_controller = FrankaControlPolicy(logger=lambda m: log_line("INFO ", m))

    if STATE.state_monitor is None:
        from rc_config import GT_TRACKED_PRIMS, GT_POSITION_THRESHOLD, GT_ORIENTATION_THRESHOLD, GT_COOLDOWN_SEC
        STATE.state_monitor = StateMonitor(
            tracked_prim_paths=GT_TRACKED_PRIMS if GT_TRACKED_PRIMS else None,
            position_threshold=GT_POSITION_THRESHOLD,
            orientation_threshold=GT_ORIENTATION_THRESHOLD,
            cooldown_sec=GT_COOLDOWN_SEC,
            logger=lambda m: log_line("INFO ", m)
        )
        # Force initial discovery and log results
        try:
            discovered = STATE.state_monitor.auto_discover_objects()
            if discovered:
                STATE.state_monitor.set_tracked_prims(discovered)
                log_line("INFO ", f"GT StateMonitor tracking {len(discovered)} objects: {discovered[:10]}")
            else:
                log_line("WARN ", "GT StateMonitor: no trackable objects found in scene")
        except Exception as exc:
            log_line("WARN ", f"GT StateMonitor discovery failed: {exc}")

    if STATE.worker_in_q is None or STATE.worker_out_q is None:
        STATE.worker_in_q = queue.Queue(maxsize=1)
        STATE.worker_out_q = queue.Queue()

    if need_worker_restart and STATE.worker_thread is not None and STATE.worker_thread.is_alive():
        try:
            _clear_worker_input_queue()
            STATE.worker_in_q.put_nowait(None)
            STATE.worker_thread.join(timeout=1.0)
        except Exception:
            pass
        STATE.worker_thread = None
        STATE.worker_in_q = queue.Queue(maxsize=1)
        STATE.worker_out_q = queue.Queue()
        log_line("INFO ", "Cognitive worker restarted for new memory backend configuration.")

    if STATE.worker_thread is None or (not STATE.worker_thread.is_alive()):
        STATE.worker_thread = threading.Thread(
            target=cognitive_worker,
            args=(STATE.belief_manager, STATE.short_memory, STATE.long_memory, STATE.worker_in_q, STATE.worker_out_q),
            name="cognitive_worker",
            daemon=True,
        )
        STATE.worker_thread.start()
        log_line("INFO ", "Cognitive worker thread started.")

    if STATE.worker_poll_task is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = omni.kit.app.get_app().get_async_loop()
        STATE.worker_poll_task = loop.create_task(_worker_poll_loop())


def _queue_put_latest(q: queue.Queue, item: Dict[str, Any]) -> None:
    """Keep at most one pending request; always replace with the latest."""
    if q is None:
        return
    try:
        while True:
            q.get_nowait()
            if hasattr(q, "task_done"):
                try:
                    q.task_done()
                except Exception:
                    pass
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        # Best-effort drop+retry once.
        try:
            q.get_nowait()
        except Exception:
            return
        try:
            q.put_nowait(item)
        except Exception:
            return


def _clear_worker_input_queue() -> None:
    if STATE.worker_in_q is None:
        return
    try:
        while True:
            STATE.worker_in_q.get_nowait()
            if hasattr(STATE.worker_in_q, "task_done"):
                try:
                    STATE.worker_in_q.task_done()
                except Exception:
                    pass
    except queue.Empty:
        pass


def _enqueue_inference(*, frames: List[Any], user_text: str, trigger: Dict[str, Any], force: bool) -> bool:
    """Queue an inference request to the worker (never blocks the Kit thread)."""
    _init_pipeline_once()
    if STATE.worker_in_q is None:
        return False

    now = time.time()
    trigger_type = str((trigger or {}).get("type") or "auto")

    # Backpressure: while one request is in-flight, drop auto heartbeats.
    # Manual/user requests still pass through.
    if (not force) and STATE.vlm_busy and trigger_type != "user":
        return False

    if (not force) and trigger_type != "after_reply" and (
        now - float(STATE.last_infer_time or 0.0) < float(MIN_INFER_INTERVAL_SEC)
    ):
        return False
    STATE.last_infer_time = now

    # Get current GT state from StateMonitor
    gt_state = {}
    if STATE.state_monitor is not None:
        try:
            current_poses = STATE.state_monitor.get_current_poses()
            if current_poses:
                gt_objects = {}
                for name, pose in current_poses.items():
                    gt_objects[pose.name] = {
                        "prim_path": pose.prim_path,
                        "position": list(pose.position),
                        "orientation": list(pose.orientation),
                        "timestamp": pose.timestamp
                    }
                gt_state = {"ground_truth_objects": gt_objects, "timestamp": now}
        except Exception as exc:
            log_line("WARN ", f"Failed to get GT state: {exc}")

    req = {
        "frames": list(frames or []),
        "user_text": str(user_text or ""),
        "trigger": dict(trigger or {}),
        "gt_state": gt_state,
        "ts": now
    }
    req["run_id"] = int(STATE.run_id)

    # Log GT state for debugging
    gt_obj_names = list(gt_state.get("ground_truth_objects", {}).keys()) if gt_state else []
    if gt_obj_names:
        log_line("INFO ", f"GT objects in request: {gt_obj_names}")
    else:
        log_line("INFO ", "GT state: empty (StateMonitor has no tracked objects)")

    _queue_put_latest(STATE.worker_in_q, req)
    STATE.vlm_busy = True
    STATE.vlm_busy_since = now

    t = trigger_type
    if t == "user":
        log_line("INFO ", f"Manual: queued for model (images={len(req['frames'])})")
        log_line("AI   ", "🤔 Model is thinking...")
    else:
        log_line("INFO ", f"Auto: queued for model (trigger={t}, images={len(req['frames'])})")
        log_line("AI   ", "🤔 Processing...")
    return True


def on_test_cosmos() -> None:
    try:
        test_cosmos_connection()
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_init_cam() -> None:
    try:
        init_camera()
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_save_frame() -> None:
    try:
        path = save_last_frame()
        log_line("INFO ", f"Saved last camera frame: {path}")
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_initiate_cosmos() -> None:
    """Pre-simulation grounding: capture static frame and initialize belief state with semantic object names."""
    try:
        log_line("INFO ", "═══ Initiating Cosmos Grounding ═══")
        _init_pipeline_once()

        # Capture a single static frame
        async def _capture_and_ground():
            try:
                from rc_camera import capture_rgb_uint8_async
                rgb = await capture_rgb_uint8_async()
                log_line("INFO ", "✓ Static frame captured for grounding")

                # Build grounding prompt
                grounding_prompt = (
                    "INITIALIZATION MODE: Analyze this static scene carefully.\n\n"
                    "Your task is to identify ALL visible objects and create an initial belief state.\n\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Use SEMANTIC object names (e.g., 'orange', 'grey_basket', 'white_basket', 'table', 'cup')\n"
                    "2. NEVER use generic IDs like 'object_1', 'object_2', etc.\n"
                    "3. If you see multiple similar objects (e.g., two baskets), distinguish them by color or position\n"
                    "   (e.g., 'grey_basket', 'white_basket' or 'left_basket', 'right_basket')\n"
                    "4. For each object, provide:\n"
                    "   - belief_status: 'visible' (since this is a static frame)\n"
                    "   - confidence: your confidence level (0.0-1.0)\n"
                    "   - visible: true\n"
                    "   - temporal_change: 'initial_state'\n\n"
                    "Expected objects in this scene:\n"
                    "- An orange (fruit)\n"
                    "- One or more baskets/containers\n"
                    "- A table/surface\n"
                    "- Any other visible objects\n\n"
                    "Return JSON with the exact structure shown in previous examples, using meaningful object names."
                )

                # Enqueue grounding inference
                frames = [rgb]
                _enqueue_inference(
                    frames=frames,
                    user_text=grounding_prompt,
                    trigger={"type": "grounding", "mode": "initialization"},
                    force=True
                )

                # Mark that grounding was initiated
                STATE.grounding_complete = False
                STATE.grounding_in_progress = True
                log_line("INFO ", "🧠 Grounding inference queued - VLM will identify objects...")

            except Exception as exc:
                log_error(str(exc))
                log_line("ERR  ", f"Grounding failed: {exc}")

        # Run async capture
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = omni.kit.app.get_app().get_async_loop()
        loop.create_task(_capture_and_ground())

    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_play() -> None:
    """User-facing PLAY that starts physics/animation."""
    try:
        omni.timeline.get_timeline_interface().play()
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def _reset_episode_runtime(reason: str) -> None:
    """Stop run tasks and reset simulation state to initial pose."""
    # De-bounce duplicated callbacks (toolbar + UI stop can both fire).
    now = time.time()
    last_ts = float(getattr(STATE, "_last_reset_ts", 0.0))
    if now - last_ts < 0.25:
        return
    setattr(STATE, "_last_reset_ts", now)

    _end_run(reason=reason)
    # Invalidate stale in-flight worker replies from previous run.
    STATE.run_id += 1
    _clear_worker_input_queue()
    STATE.vlm_busy = False
    STATE.vlm_busy_since = 0.0
    STATE.last_auto_infer_ts = 0.0
    try:
        STATE.frame_buffer.clear()
    except Exception:
        pass

    try:
        from isaaclab.sim import SimulationContext

        sim_ctx = SimulationContext.instance()
        if sim_ctx is not None:
            # Full reset restores object states (e.g., orange back to initial pose).
            sim_ctx.reset()
            log_line("INFO ", "Episode reset via SimulationContext.reset().")
    except Exception as exc:
        log_warn(f"Episode reset not available in this workflow: {exc}")


def on_clear() -> None:
    STATE.log_lines.clear()
    render_log()
    log_info("Cleared logs.")


def on_send() -> None:
    """Queue a user message for the cognition worker (non-blocking)."""
    user_text = (STATE.input_model.as_string or "").strip() if STATE.input_model else ""
    if not user_text:
        return

    log_line("YOU  ", user_text)
    if STATE.input_model:
        STATE.input_model.set_value("")

    attach = True
    if STATE.attach_latest_model is not None:
        try:
            attach = bool(STATE.attach_latest_model.as_bool)
        except Exception:
            attach = True

    frames: List[Any] = []
    if attach:
        # Manual prompts use the latest full temporal window (oldest->newest).
        # This helps the model reason over scene dynamics instead of a near-duplicate pair.
        frames = _get_recent_frames(max(INQUIRY_INTERVAL_SEC * 2.5, 1.8), int(INQUIRY_FRAME_COUNT))

    if attach and not frames:
        log_line("INFO ", "Manual prompt has no buffered frames; sending text-only.")

    # Save frames used for manual prompts for traceability (no blocking on disk I/O).
    if frames:

        async def _save_manual() -> None:
            try:
                paths = await asyncio.to_thread(save_sent_frames, "manual", list(frames))
                if paths:
                    log_line("INFO ", f"Manual sent frames: {', '.join(paths)}")
            except Exception as exc:
                log_warn(f"Failed to save manual frames: {exc}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = omni.kit.app.get_app().get_async_loop()
        loop.create_task(_save_manual())

    # Pause auto briefly so the user's request has priority.
    STATE.pause_inquiry_until = time.time() + float(INQUIRY_INTERVAL_SEC)
    _enqueue_inference(frames=frames, user_text=user_text, trigger={"type": "user"}, force=True)


async def _capture_loop() -> None:
    """Capture RGB frames into the RAM ring buffer (does not write to disk)."""
    interval = 1.0 / max(0.1, float(CAPTURE_FPS))
    count = 0
    last_log = time.time()
    while STATE.playing:
        try:
            rgb = await capture_rgb_uint8_async()
            ts = time.time()
            # Freeze the frame in RAM; some backends reuse the same underlying buffer.
            try:
                frozen = rgb.copy()
            except Exception:
                frozen = rgb
            STATE.frame_buffer.append((ts, frozen))
            count += 1
            if (ts - last_log) >= 1.0:
                log_line("INFO ", f"Captured frames: {count}/s (buffer={len(STATE.frame_buffer)})")
                count = 0
                last_log = ts
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break
        await asyncio.sleep(interval)


async def _worker_poll_loop() -> None:
    """Poll worker outputs and update belief/ghost/robot from the main thread.

    This loop runs for the whole UI session (not only while timeline is playing),
    so manual prompts still receive/log model replies when simulation is paused.
    """
    while True:
        try:
            while True:
                msg = STATE.worker_out_q.get_nowait() if STATE.worker_out_q is not None else None
                if not msg:
                    break

                # Accept worker outputs even if run_id mismatches (worker was still processing when STOP was pressed)
                # Only drop if the output is truly ancient (>30s old)
                try:
                    msg_run_id = int(msg.get("run_id", STATE.run_id))
                    msg_ts = float(msg.get("ts", 0))
                    age = time.time() - msg_ts if msg_ts > 0 else 999
                except Exception:
                    msg_run_id = int(STATE.run_id)
                    age = 0

                if msg_run_id != int(STATE.run_id) and age > 30.0:
                    log_line("WARN ", f"Dropped ancient worker output (age={age:.1f}s, run_id={msg_run_id})")
                    continue

                status = msg.get("status")
                if status != "Done":
                    STATE.vlm_busy = False
                    STATE.vlm_busy_since = 0.0
                    err = str(msg.get("error") or "")
                    log_line("WARN ", f"Worker status={status}: {err}")
                    # Avoid spamming a dead/OOM model server with immediate retries.
                    if "HTTP 500" in err or "EngineDeadError" in err or "out of memory" in err.lower():
                        STATE.pause_inquiry_until = time.time() + 8.0
                        log_line("WARN ", "Auto inquiry paused for 8s after model error.")
                    continue

                # Calculate response time
                response_time = time.time() - STATE.vlm_busy_since if STATE.vlm_busy_since > 0 else 0
                STATE.vlm_busy = False
                STATE.vlm_busy_since = 0.0

                # Check if this was a grounding request
                trigger = msg.get("trigger", {})
                is_grounding = trigger.get("type") == "grounding" and trigger.get("mode") == "initialization"

                if is_grounding:
                    log_line("INFO ", f"═══ Grounding Complete ({response_time:.1f}s) ═══")
                    STATE.grounding_complete = True
                    STATE.grounding_in_progress = False
                else:
                    log_line("INFO ", f"Worker response received ({response_time:.1f}s)")

                reply = str(msg.get("reply") or "")
                action = msg.get("action") if isinstance(msg.get("action"), dict) else {}
                ltm_snips = msg.get("ltm_snippets") if isinstance(msg.get("ltm_snippets"), list) else []
                snapshot = STATE.belief_manager.get_snapshot() if STATE.belief_manager else {}

                # Defensive sync: if worker produced canonical belief but STM didn't ingest it, repair here.
                if STATE.short_memory is not None and isinstance(snapshot, dict):
                    try:
                        STATE.short_memory.update_objects_from_belief(snapshot, now=time.time())
                    except Exception as exc:
                        log_line("WARN ", f"ShortMemory sync failed: {exc}")

                # Always log model output to terminal
                raw_output = msg.get("raw") or {}
                belief_update = msg.get("belief_update") or {}

                # Also print to stdout so it appears in terminal regardless of UI
                print(f"[WORKER] reply={reply[:200]!r}", flush=True)
                if isinstance(raw_output, dict):
                    raw_reply = str(raw_output.get("reply") or "")
                    raw_meta = raw_output.get("meta", {})
                    print(f"[WORKER] raw.reply={raw_reply[:200]!r} | raw.meta={raw_meta}", flush=True)

                # Find the best reply text from multiple sources
                display_reply = ""
                if reply and reply not in ("<empty response>", ""):
                    display_reply = reply
                elif isinstance(raw_output, dict):
                    raw_reply = str(raw_output.get("reply") or "")
                    if raw_reply and raw_reply not in ("<empty response>", ""):
                        display_reply = raw_reply

                if display_reply:
                    _log_long("AI   ", display_reply)
                    STATE.last_vlm_response = display_reply
                else:
                    # Build summary from belief_update objects
                    obj_parts = []
                    bu_objs = belief_update.get("objects", {}) if isinstance(belief_update, dict) else {}
                    if isinstance(bu_objs, dict):
                        for oname, odata in list(bu_objs.items())[:5]:
                            if isinstance(odata, dict):
                                status = odata.get("belief_status", "?")
                                conf = odata.get("confidence", "?")
                                change = odata.get("temporal_change", "")
                                obj_parts.append(f"{oname}: {status} (conf={conf}, {change})")
                    if obj_parts:
                        summary = "Observed: " + " | ".join(obj_parts)
                        log_line("AI   ", summary)
                        STATE.last_vlm_response = summary
                    else:
                        # Log parse error if present in meta
                        parse_err = ""
                        if isinstance(raw_output, dict):
                            meta = raw_output.get("meta", {})
                            if isinstance(meta, dict):
                                parse_err = str(meta.get("parse_error", ""))
                        if parse_err:
                            log_line("WARN ", f"Model error: {parse_err[:300]}")
                        log_line("AI   ", "<no model output - check terminal for [VLM] logs>")

                # Log belief update and action
                if belief_update:
                    log_line("INFO ", f"Belief update: {json.dumps(belief_update, ensure_ascii=True)[:500]}")
                if action and action.get("type") != "noop":
                    log_line("INFO ", f"Action: {json.dumps(action, ensure_ascii=True)}")

                # Update UI labels.
                if STATE.action_label is not None:
                    try:
                        STATE.action_label.text = json.dumps(action or {}, ensure_ascii=True, indent=2)[:2000]
                    except Exception:
                        STATE.action_label.text = str(action)

                if STATE.belief_label is not None and STATE.short_memory is not None:
                    belief_text = STATE.short_memory.compact_summary()
                    # Fallback to raw belief snapshot if STM looks empty.
                    if belief_text.startswith("no objects tracked"):
                        belief_text = _compact_belief_from_snapshot(snapshot)
                    STATE.belief_label.text = belief_text

                if STATE.ltm_label is not None:
                    STATE.ltm_label.text = "\n".join([str(s) for s in ltm_snips][:5])

                gv = STATE.ghost_visualizer
                if gv is not None and getattr(gv, "enabled", False):
                    try:
                        gv.sync_ghosts(snapshot)
                    except Exception as exc:
                        log_line("WARN ", f"Ghost sync failed: {exc}")

                # Execute robot action on the main thread (non-blocking, persistent target).
                rc = STATE.robot_controller
                if rc is not None and getattr(rc, "enabled", False) and isinstance(action, dict) and action:
                    a_type = str(action.get("type") or "").strip().lower()
                    if a_type and a_type != "noop":
                        rc.set_action(action)
                        if STATE.short_memory is not None:
                            STATE.short_memory.set_last_action(action, status="running")
        except queue.Empty:
            # Watchdog: if worker has been busy for too long without output, log it
            if STATE.vlm_busy and STATE.vlm_busy_since > 0:
                busy_duration = time.time() - STATE.vlm_busy_since
                if busy_duration > 30.0:  # 30 second timeout
                    log_line("WARN ", f"Worker busy for {busy_duration:.1f}s without response - may be stuck")
                    if busy_duration > 60.0:  # 1 minute - force release
                        STATE.vlm_busy = False
                        STATE.vlm_busy_since = 0.0
                        log_line("WARN ", "Worker watchdog: released stuck busy flag")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break

        await asyncio.sleep(0.05)


async def _inquiry_loop() -> None:
    """Completion-driven auto trigger.

    While playing:
    - keep at most one in-flight request,
    - when free, enqueue the latest recent frame sequence,
    - do not build periodic backlog.
    """
    while STATE.playing:
        tick = 0.05
        try:
            if time.time() < float(STATE.pause_inquiry_until or 0.0):
                await asyncio.sleep(tick)
                continue

            # Completion-driven chaining: only queue a new auto request when the previous one finished.
            if STATE.vlm_busy:
                # If worker got stuck/crashed without callback, release the busy flag after timeout.
                if (time.time() - float(STATE.vlm_busy_since or 0.0)) > max(30.0, float(TIMEOUT_SEC) * 1.5):
                    STATE.vlm_busy = False
                    STATE.vlm_busy_since = 0.0
                    log_line("WARN ", "Auto inquiry watchdog released a stale in-flight request.")
                await asyncio.sleep(tick)
                continue

            # Strict async mode: after each completed reply, send newest unseen frames only.
            auto_frame_limit = max(1, int(INQUIRY_FRAME_COUNT))
            frames, newest_ts = _get_frames_since(float(STATE.last_auto_infer_ts or 0.0), auto_frame_limit)
            min_auto_frames = max(3, min(auto_frame_limit, 5))
            if len(frames) < min_auto_frames:
                await asyncio.sleep(tick)
                continue

            # Save the exact batch used for auto inference (traceability).
            try:
                paths = await asyncio.to_thread(save_sent_frames, "auto", list(frames))
                if paths:
                    log_line("INFO ", f"Auto sent frames: {', '.join(paths)}")
            except Exception as exc:
                log_warn(f"Failed to save auto frames: {exc}")

            if _enqueue_inference(frames=frames, user_text=STREAM_DEFAULT_PROMPT, trigger={"type": "after_reply"}, force=False):
                STATE.last_auto_infer_ts = max(float(STATE.last_auto_infer_ts or 0.0), float(newest_ts))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break

        await asyncio.sleep(tick)


async def _robot_update_loop() -> None:
    """Apply robot targets and monitor GT state every Kit tick while playing (main thread only)."""
    rc = STATE.robot_controller
    sm = STATE.state_monitor
    last_status = None
    while STATE.playing:
        try:
            # Update robot controller if enabled
            if rc is not None and getattr(rc, "enabled", False):
                status = rc.update()
                # Trigger a refresh when an action completes.
                if last_status and last_status.status == "running" and status.status in ("done", "error"):
                    # Update STM status for downstream prompting / UI.
                    if STATE.short_memory is not None:
                        try:
                            snap = STATE.short_memory.get_snapshot()
                            last_action = {}
                            if isinstance(snap.get("task"), dict) and isinstance(snap["task"].get("last_action"), dict):
                                last_action = snap["task"]["last_action"]
                            STATE.short_memory.set_last_action(last_action, status=status.status)
                        except Exception:
                            pass
                    log_line("INFO ", f"Robot action finished: {status.status} ({status.detail})")
                    frames = _get_recent_frames(INQUIRY_INTERVAL_SEC, INQUIRY_FRAME_COUNT)
                    _enqueue_inference(
                        frames=frames,
                        user_text="Describe scene changes after the robot action completed.",
                        trigger={"type": "action_finished", "status": status.status, "detail": status.detail},
                        force=False,
                    )
                last_status = status

            # Monitor GT state changes and trigger inference
            if sm is not None:
                trigger_event = sm.update()
                if trigger_event is not None:
                    # Skip if displacement is negligible (initial baseline or noise)
                    if trigger_event.max_displacement < 0.001:
                        continue

                    changed_names = ", ".join(trigger_event.changed_objects) if trigger_event.changed_objects else "objects"
                    log_line("INFO ", f"GT trigger: {trigger_event.changed_objects} changed (displacement={trigger_event.max_displacement:.3f}m)")
                    frames = _get_recent_frames(INQUIRY_INTERVAL_SEC, INQUIRY_FRAME_COUNT)

                    # Skip if frame buffer is empty (happens at startup before capture loop fills buffer)
                    if not frames:
                        log_line("WARN ", f"GT trigger skipped - frame buffer empty (need {INQUIRY_FRAME_COUNT} frames)")
                    else:
                        _enqueue_inference(
                            frames=frames,
                            user_text=f"Objects moved: {changed_names}. Describe what changed and update belief states.",
                            trigger={
                                "type": "gt_change",
                                "changed_objects": trigger_event.changed_objects,
                                "max_displacement": trigger_event.max_displacement,
                                "timestamp": trigger_event.timestamp
                            },
                            force=False,
                        )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_warn(f"Robot/GT update failed: {exc}")
            # Don't break - keep the loop running so GT monitoring continues
        await omni.kit.app.get_app().next_update_async()


def start_capture_and_inquiry() -> None:
    if STATE.capture_task is not None or STATE.inquiry_task is not None:
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = omni.kit.app.get_app().get_async_loop()

    STATE.capture_task = loop.create_task(_capture_loop())
    STATE.inquiry_task = loop.create_task(_inquiry_loop())
    if STATE.robot_update_task is None:
        STATE.robot_update_task = loop.create_task(_robot_update_loop())


def stop_capture_and_inquiry() -> None:
    if STATE.capture_task is not None:
        STATE.capture_task.cancel()
        STATE.capture_task = None
    if STATE.inquiry_task is not None:
        STATE.inquiry_task.cancel()
        STATE.inquiry_task = None
    if STATE.robot_update_task is not None:
        STATE.robot_update_task.cancel()
        STATE.robot_update_task = None


def _set_run_frames_dir() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(STATE.sent_frames_root, ts)
    os.makedirs(run_dir, exist_ok=True)
    STATE.sent_frames_dir = run_dir
    log_line("INFO ", f"Run frames dir: {run_dir}")


def _start_run() -> None:
    if STATE.playing:
        return
    STATE.run_id += 1
    STATE.playing = True
    STATE.run_active = True
    STATE.run_start_time = time.time()
    STATE.last_auto_infer_ts = float(STATE.run_start_time)
    _set_run_frames_dir()
    start_capture_and_inquiry()


def _end_run(*, reason: str) -> None:
    if not STATE.playing:
        return
    STATE.playing = False
    STATE.run_active = False
    stop_capture_and_inquiry()
    if STATE.run_start_time:
        duration = time.time() - STATE.run_start_time
        log_line("INFO ", f"Run duration: {duration:.2f}s (ended via {reason})")

    # Store an episode record asynchronously (avoid blocking UI/timeline callback thread).
    if STATE.long_memory is not None:
        snap = STATE.short_memory.get_snapshot() if STATE.short_memory is not None else {}
        belief = STATE.belief_manager.get_snapshot() if STATE.belief_manager is not None else {}
        payload = {"task": "falling_orange", "last_belief": belief, "stm": snap}

        async def _store_episode_async() -> None:
            try:
                await asyncio.to_thread(STATE.long_memory.end_episode, payload)
            except Exception as exc:
                log_warn(f"Failed to store episode in LTM: {exc}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = omni.kit.app.get_app().get_async_loop()
        loop.create_task(_store_episode_async())


def on_end_run() -> None:
    """User-facing STOP: pause timeline and reset episode to initial state."""
    try:
        tl = omni.timeline.get_timeline_interface()
        # Explicit STOP to match user expectation and force timeline state transition.
        tl.stop()
    except Exception:
        pass
    # Perform reset immediately so STOP semantics are deterministic.
    _reset_episode_runtime("ui_stop")


def on_reset_episode() -> None:
    """Backward-compatible alias for STOP/reset."""
    on_end_run()


def _install_timeline_run_hooks() -> None:
    """Start/stop the run on Timeline events without polling."""
    if getattr(STATE, "_timeline_hooks_installed", False):
        return
    setattr(STATE, "_timeline_hooks_installed", True)

    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()

    def _on_timeline_event(e):
        try:
            et = int(getattr(e, "type", -1))
        except Exception:
            et = -1

        if et == int(omni.timeline.TimelineEventType.PLAY):
            _start_run()
        elif et == int(omni.timeline.TimelineEventType.PAUSE):
            # Convert PAUSE into full STOP semantics so reset is deterministic.
            try:
                timeline.stop()
            except Exception:
                pass
            _reset_episode_runtime("timeline_pause")
        elif et == int(omni.timeline.TimelineEventType.STOP):
            # Stop means reset to initial episode state.
            _reset_episode_runtime("timeline_stop")

    try:
        sub = stream.create_subscription_to_pop(_on_timeline_event)
        STATE._kit_subs.append(sub)
    except Exception as exc:
        log_warn(f"Failed to subscribe to timeline events: {exc}")
        return

    # If the user hit PLAY before hooks installed, sync once.
    try:
        if timeline.is_playing():
            _start_run()
    except Exception:
        pass


def build_ui() -> None:
    if STATE.window is not None:
        STATE.window.visible = True
        return

    STATE.window = ui.Window("HeadCam Terminal Chat", width=760, height=640)

    with STATE.window.frame:
        with ui.VStack(spacing=6):
            # Top controls: compact fixed height.
            with ui.Frame(height=120):
                with ui.VStack(spacing=6):
                    ui.Label("Free chat + agent loop. Frames captured in RAM; model runs off-thread (sim never blocks).")

                    with ui.HStack(height=28, spacing=8):
                        ui.Button("Init Camera", clicked_fn=on_init_cam, height=28)
                        ui.Button("Initiate Cosmos", clicked_fn=on_initiate_cosmos, height=28, tooltip="Pre-simulation: VLM identifies objects and creates initial belief state")
                        ui.Button("Test Cosmos", clicked_fn=on_test_cosmos, height=28)
                        ui.Button("PLAY", clicked_fn=on_play, height=28)
                        ui.Button("STOP", clicked_fn=on_end_run, height=28)
                        ui.Spacer()
                        ui.Button("Clear", clicked_fn=on_clear, height=28)

                    with ui.HStack(height=22, spacing=8):
                        STATE.attach_latest_model = ui.SimpleBoolModel(True)
                        ui.CheckBox(STATE.attach_latest_model, width=18, height=18)
                        ui.Label("Attach recent frames to manual prompts (last 1s)")

                    ui.Label(f"Camera Prim: {STATE.active_camera_prim_path}", word_wrap=True)
                    ui.Label(f"Cosmos Endpoint: {cosmos_endpoint()}", word_wrap=True)
                    ui.Label(f"Cosmos Model: {COSMOS_MODEL}", word_wrap=True)

            ui.Separator()
            ui.Label("Terminal:")

            # Use a larger fixed terminal area for better visibility.
            STATE.log_frame = ui.ScrollingFrame(height=300)
            with STATE.log_frame:
                STATE.log_label = ui.Label("", word_wrap=True)

            ui.Separator()

            # Bottom status + input panel: compact fixed height.
            with ui.Frame(height=145):
                with ui.VStack(spacing=6):
                    ui.Label("Last action:")
                    STATE.action_label = ui.Label("{}", word_wrap=True, height=24)

                    ui.Label("Belief (compact):")
                    STATE.belief_label = ui.Label("", word_wrap=True, height=20)

                    ui.Label("Long-term memory (top snippets):")
                    STATE.ltm_label = ui.Label("", word_wrap=True, height=24)

                    ui.Separator()

                    ui.Label("Type anything:")
                    STATE.input_model = ui.SimpleStringModel("")
                    ui.StringField(STATE.input_model, height=30)

                    with ui.HStack(height=30, spacing=8):
                        ui.Button("Send", clicked_fn=on_send, height=30)
                        ui.Spacer()


def _install_startup_autostop_hooks() -> None:
    """Install hooks to keep the stage static on launch until the user clicks PLAY.

    In Isaac Lab "python" experiences, the timeline can sometimes resume PLAY from persistent settings,
    or get toggled to PLAY after a stage finishes loading. We force STOP around stage-open time so the
    initial state is stable/static and the demo only proceeds after an explicit user PLAY.
    """
    if getattr(STATE, "_startup_hooks_installed", False):
        return
    setattr(STATE, "_startup_hooks_installed", True)

    # Best-effort: mark stage as opened if one already exists (script-editor use case).
    try:
        import omni.usd

        if omni.usd.get_context().get_stage() is not None:
            STATE.startup_stage_opened = True
    except Exception:
        pass

    try:
        import omni.usd

        ctx = omni.usd.get_context()
        stream = ctx.get_stage_event_stream()
        stage_event_type = getattr(omni.usd, "StageEventType", None)
        opened_type = None
        if stage_event_type is not None and hasattr(stage_event_type, "OPENED"):
            try:
                opened_type = int(stage_event_type.OPENED)
            except Exception:
                opened_type = stage_event_type.OPENED
        if opened_type is None:
            # Fallback: rely on the startup timeline guard instead of guessing stage event types.
            raise RuntimeError("omni.usd.StageEventType.OPENED not available")

        timeline = omni.timeline.get_timeline_interface()

        def _force_show_play_button() -> None:
            # Keep compatibility with existing startup guards if the toolbar is present.
            try:
                import omni.kit.widget.toolbar

                toolbar = omni.kit.widget.toolbar.get_instance()
                play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore[attr-defined]
                if play_button_group is not None:
                    play_btn = play_button_group._play_button  # type: ignore[attr-defined]
                    play_btn.visible = True
                    play_btn.enabled = True
            except Exception:
                pass

        def _on_stage_event(e):
            if getattr(STATE, "_startup_stage_event_handled", False):
                return
            try:
                et = int(getattr(e, "type", -1))
            except Exception:
                et = -1
            if et != int(opened_type):
                return

            STATE.startup_stage_opened = True
            setattr(STATE, "_startup_stage_event_handled", True)
            try:
                if timeline.is_playing():
                    timeline.pause()
                    log_warn(
                        "Timeline PLAY detected during startup/stage-load; forced PAUSE. Click PLAY once the stage finishes loading."
                    )
                else:
                    # Still call PAUSE to ensure we are not advancing.
                    timeline.pause()
                _force_show_play_button()
            except Exception:
                pass

        sub = stream.create_subscription_to_pop(_on_stage_event)
        STATE._kit_subs.append(sub)

        # Panel PLAY/STOP controls are authoritative for this demo UI.
    except Exception as exc:
        log_warn(f"Failed to install stage event hook for auto-play suppression: {exc}")


def run() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = omni.kit.app.get_app().get_async_loop()

    _configure_quiet_kit_logging()
    _ensure_asset_browser_cache_dir()

    # Install stage/timeline hooks immediately so we don't miss early stage-open events.
    _install_startup_autostop_hooks()
    # Best-effort: force PAUSE immediately on script start (keeps toolbar PLAY available).
    try:
        omni.timeline.get_timeline_interface().pause()
    except Exception:
        pass
    # Hide built-in timeline controls to avoid PLAY/PAUSE ambiguity; use panel PLAY/STOP.
    try:
        import omni.kit.widget.toolbar

        toolbar = omni.kit.widget.toolbar.get_instance()
        play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore[attr-defined]
        if play_button_group is not None:
            play_button_group.visible = False
            play_button_group.enabled = False
    except Exception:
        pass

    async def _startup_timeline_guard(max_duration_sec: float = 15.0, stable_updates: int = 15) -> None:
        """Prevent unintended auto-play during startup/stage-load.

        Exits early once the stage is opened and the timeline has remained STOPPED for a short
        number of Kit updates. This avoids the "must click PLAY twice" regression while still
        catching late auto-play triggers after the stage loads.
        """
        timeline = omni.timeline.get_timeline_interface()
        warned = False
        stable = 0
        deadline = time.time() + float(max_duration_sec)

        # Force STOP once immediately (best-effort).
        try:
            timeline.pause()
        except Exception:
            pass

        while time.time() < deadline:
            try:
                if timeline.is_playing():
                    stable = 0
                    timeline.pause()
                    if not warned:
                        log_warn("Timeline PLAY detected during startup; forced PAUSE. Click PLAY once the stage finishes loading.")
                        warned = True
                else:
                    if STATE.startup_stage_opened:
                        stable += 1
                        if stable >= int(stable_updates):
                            break
            except Exception:
                pass
            await omni.kit.app.get_app().next_update_async()

        STATE.startup_autostop_done = True

    async def _startup_sequence() -> None:
        await _startup_timeline_guard()
        _install_timeline_run_hooks()

    # Kick startup sequence (do not await; keep UI responsive).
    loop.create_task(_startup_sequence())
    build_ui()
    render_log()

    log_info("UI ready. Click 'Init Camera' once, then click PLAY to start the sim loop.")
    log_info(f"Cosmos endpoint: {cosmos_endpoint()}")
    log_info(f"Cosmos model: {COSMOS_MODEL}")
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        log_warn("COSMOS_MODEL is not set. Update COSMOS_MODEL or set COSMOS_MODEL env var.")

    # Best-effort cleanup on Kit shutdown.
    try:
        app = omni.kit.app.get_app()

        def _on_shutdown(_e=None):
            try:
                if STATE.worker_in_q is not None:
                    STATE.worker_in_q.put_nowait(None)
            except Exception:
                pass
            try:
                if STATE.worker_thread is not None and STATE.worker_thread.is_alive():
                    STATE.worker_thread.join(timeout=1.0)
            except Exception:
                pass

        app.get_shutdown_event_stream().create_subscription_to_pop(_on_shutdown)
    except Exception:
        pass
