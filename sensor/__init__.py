"""Sensor module — camera capture, perception, and VLM connection utilities."""

try:
    from .camera import (
        init_camera,
        capture_rgb_uint8_async,
        get_latest_rgb_uint8,
        rgb_to_image_base64,
        frames_to_image_parts,
        save_last_frame,
        save_sent_frames,
    )
except ImportError:
    pass  # requires omni.replicator / Isaac Sim runtime

try:
    from .vlm import cosmos_endpoint, test_cosmos_connection
except ImportError:
    pass  # requires rc_config (available at runtime)

from .perception import StateMonitor
