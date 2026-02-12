import asyncio
import base64
import os
import time
from io import BytesIO

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom

from rc_config import CAMERA_PRIM_PATH, RESOLUTION, IMAGE_FORMAT, IMAGE_QUALITY, SAVE_EACH_SENT_FRAME
from rc_state import STATE
from rc_log import log_info, log_warn


def init_camera():
    """Initialize the Replicator render product + RGB annotator for the configured camera."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not ready yet.")

    prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not prim or not prim.IsValid():
        found_path = None
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Camera):
                path = p.GetPath().pathString
                if path.endswith("/head_camera"):
                    found_path = path
                    break
                if found_path is None:
                    found_path = path
        if not found_path:
            raise RuntimeError(f"No valid camera prim found. Check path: {CAMERA_PRIM_PATH}")
        STATE.active_camera_prim_path = found_path
        log_warn(f"Camera prim not found at {CAMERA_PRIM_PATH}. Using {found_path} instead.")
    else:
        STATE.active_camera_prim_path = CAMERA_PRIM_PATH

    # Recreate render product + annotator (stage changes / play-stop can invalidate these).
    STATE.rp = rep.create.render_product(STATE.active_camera_prim_path, RESOLUTION)
    STATE.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    STATE.rgb_annot.attach([STATE.rp])

    STATE.camera_ready = True
    log_info(f"Camera initialized: {STATE.active_camera_prim_path} @ {RESOLUTION}")


async def capture_rgb_uint8_async() -> np.ndarray:
    """Capture the latest RGB frame as uint8 HxWx3."""
    if (not STATE.camera_ready) or (STATE.rgb_annot is None) or (STATE.rp is None):
        init_camera()

    # During PLAY, avoid stepping Replicator (it can stall the timeline).
    # Just wait for the next Kit update and read the current render product.
    if omni.timeline.get_timeline_interface().is_playing():
        await omni.kit.app.get_app().next_update_async()
    else:
        await rep.orchestrator.step_async()

    data = STATE.rgb_annot.get_data()
    if data is None:
        # Render product can be dropped on stage changes; re-init once.
        log_warn("RGB annotator returned None. Reinitializing camera.")
        init_camera()
        data = STATE.rgb_annot.get_data()
        if data is None:
            raise RuntimeError("Annotator rgb is not attached to any render products.")

    arr = np.array(data)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    rgb = arr.astype(np.uint8, copy=False)
    STATE.last_rgb = rgb
    return rgb


def get_latest_rgb_uint8() -> np.ndarray | None:
    """Get the latest available RGB frame without stepping/waiting.

    This is intended for use inside the synchronous physics-step callback.
    It may return the same frame multiple times if rendering is slower than physics.
    """
    if (not STATE.camera_ready) or (STATE.rgb_annot is None) or (STATE.rp is None):
        init_camera()

    data = STATE.rgb_annot.get_data()
    if data is None:
        return None
    arr = np.array(data)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    rgb = arr.astype(np.uint8, copy=False)
    STATE.last_rgb = rgb
    return rgb


def rgb_to_image_base64(rgb: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow (PIL) not available in this Kit Python env: {e}")

    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_to_image_parts(frames: list[np.ndarray], mime: str) -> list[dict]:
    """Convert frames to OpenAI-style image_url parts."""
    parts = []
    for rgb in frames:
        b64 = rgb_to_image_base64(rgb)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


def save_last_frame(path: str | None = None) -> str:
    if STATE.last_rgb is None:
        raise RuntimeError("No frame captured yet. Click Init Camera and wait for capture.")
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow (PIL) not available in this Kit Python env: {e}")

    out_path = path or STATE.last_frame_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img = Image.fromarray(STATE.last_rgb)
    img.save(out_path, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
    return out_path


def save_sent_frames(prefix: str, frames: list[np.ndarray]) -> list[str]:
    """Save multiple frames into the current run directory.

    Filenames include milliseconds to avoid overwriting within the same second.
    """
    if not SAVE_EACH_SENT_FRAME:
        return []
    if not frames:
        raise RuntimeError("No frames provided to save_sent_frames.")
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow (PIL) not available in this Kit Python env: {e}")

    os.makedirs(STATE.sent_frames_dir, exist_ok=True)
    ts = time.time()
    ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    ms = int((ts - int(ts)) * 1000)

    paths = []
    for idx, rgb in enumerate(frames):
        out_path = os.path.join(STATE.sent_frames_dir, f"{prefix}_{ts_str}_{ms:03d}_{idx}.jpg")
        img = Image.fromarray(rgb)
        img.save(out_path, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
        paths.append(out_path)
    return paths
