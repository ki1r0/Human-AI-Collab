"""Hybrid Perception Module: RingBuffer + Ground-Truth StateMonitor.

RingBuffer: Fixed-size deque of (timestamp, numpy_rgb) tuples for visual context.
StateMonitor: Queries USD stage for object poses each frame, triggers agent on significant change.
"""
from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-capacity ring buffer storing (timestamp, numpy_rgb_uint8) frames."""

    def __init__(self, capacity: int = 5) -> None:
        self._capacity = max(1, int(capacity))
        self._buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, frame: np.ndarray, ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.time()
        with self._lock:
            self._buf.append((float(ts), frame))

    def get_latest(self, n: Optional[int] = None) -> List[Tuple[float, np.ndarray]]:
        """Return up to *n* most recent frames as [(ts, rgb), ...] oldest-first."""
        with self._lock:
            items = list(self._buf)
        if n is not None:
            items = items[-n:]
        return items

    def get_frames_only(self, n: Optional[int] = None) -> List[np.ndarray]:
        """Return just the RGB arrays, oldest-first."""
        return [frame for _, frame in self.get_latest(n)]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buf) >= self._capacity


# ---------------------------------------------------------------------------
# Object Pose Snapshot
# ---------------------------------------------------------------------------

@dataclass
class ObjectPose:
    prim_path: str
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # (w, x, y, z)
    timestamp: float = 0.0


@dataclass
class TriggerEvent:
    """Describes what triggered the agent."""
    trigger_type: str  # "gt_change" | "user_input" | "periodic" | "initial"
    changed_objects: List[str] = field(default_factory=list)
    max_displacement: float = 0.0
    user_text: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.trigger_type,
            "changed_objects": self.changed_objects,
            "max_displacement": round(self.max_displacement, 5),
            "user_text": self.user_text,
            "ts": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Ground-Truth State Monitor
# ---------------------------------------------------------------------------

class StateMonitor:
    """Queries USD stage for tracked object poses and detects significant changes.

    Must be called on the main thread (USD API requirement).
    """

    def __init__(
        self,
        tracked_prim_paths: Optional[List[str]] = None,
        position_threshold: float = 0.02,
        orientation_threshold: float = 0.05,
        cooldown_sec: float = 1.0,
        logger=print,
    ) -> None:
        self._log = logger
        self._tracked_paths = list(tracked_prim_paths or [])
        self._pos_threshold = float(position_threshold)
        self._ori_threshold = float(orientation_threshold)
        self._cooldown = float(cooldown_sec)

        self._prev_poses: Dict[str, ObjectPose] = {}
        self._last_trigger_time: float = 0.0
        self._initialized = False

        # Auto-discovered tracked prims (populated on first update if no paths given).
        self._auto_discovered = False

    def set_tracked_prims(self, prim_paths: List[str]) -> None:
        """Explicitly set tracked prim paths."""
        self._tracked_paths = list(prim_paths)
        self._auto_discovered = True

    def auto_discover_objects(self) -> List[str]:
        """Discover movable objects in the stage by looking for rigid bodies.

        Only tracks objects with RigidBodyAPI (physics-simulated objects that can
        actually move). Excludes structural elements, robot parts, and scene fixtures.
        """
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return []

            discovered = []
            # Comprehensive skip list for structural / non-manipulable prims.
            skip_keywords = {
                # Structural
                "ground", "floor", "wall", "ceiling", "roof",
                "room", "house", "building", "structure",
                # Fixtures
                "light", "lamp", "bulb", "fixture", "chandelier",
                "window", "door", "frame", "trim", "molding",
                "handle", "hinge", "knob", "latch", "lock",
                "shelf", "shelving", "cabinet", "drawer", "closet",
                "rack", "towel_rack", "towelrack", "hook",
                "sink", "faucet", "tap", "pipe", "drain",
                "toilet", "bath", "shower", "tub",
                "mirror", "picture", "painting", "poster",
                "curtain", "blind", "shade",
                "vent", "outlet", "switch", "socket",
                "stair", "railing", "banister",
                "column", "pillar", "beam", "support",
                # Robot
                "camera", "franka", "panda", "robot", "gripper",
                "finger", "joint", "link", "hand",
                # Internal/debug
                "mind", "ghost", "collider", "collision",
                "physics", "material", "shader", "texture",
                "scope", "xform", "looks", "render",
            }

            for prim in stage.Traverse():
                if not prim.IsValid():
                    continue
                path_str = prim.GetPath().pathString
                path_lower = path_str.lower()
                name_lower = prim.GetName().lower()

                # Skip structural/robot prims.
                if any(kw in path_lower for kw in skip_keywords):
                    continue

                # Skip deeply nested prims (likely sub-geometry of larger objects).
                # Typical manipulable objects live at /World/ObjectName depth (3-4 segments).
                depth = path_str.count("/")
                if depth > 5:
                    continue

                # ONLY track objects with RigidBodyAPI - these are physics-simulated
                # objects that can actually move. Do NOT track plain meshes.
                has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                if has_rigid:
                    discovered.append(path_str)

            self._log(f"[INFO] StateMonitor: auto-discovered {len(discovered)} rigid-body objects")
            for p in discovered[:10]:
                self._log(f"[INFO]   tracked: {p}")
            if len(discovered) > 10:
                self._log(f"[INFO]   ... and {len(discovered) - 10} more")
            return discovered

        except Exception as exc:
            self._log(f"[WARN] StateMonitor: auto-discovery failed: {exc}")
            return []

    def get_current_poses(self) -> Dict[str, ObjectPose]:
        """Query current world-space poses for all tracked prims."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return {}

            now = time.time()
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            poses: Dict[str, ObjectPose] = {}

            for prim_path in self._tracked_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue

                try:
                    xform = xform_cache.GetLocalToWorldTransform(prim)
                    t = xform.ExtractTranslation()
                    r = xform.ExtractRotation()
                    quat = r.GetQuat()

                    pos = (float(t[0]), float(t[1]), float(t[2]))
                    ori = (
                        float(quat.GetReal()),
                        float(quat.GetImaginary()[0]),
                        float(quat.GetImaginary()[1]),
                        float(quat.GetImaginary()[2]),
                    )

                    poses[prim_path] = ObjectPose(
                        prim_path=prim_path,
                        name=prim.GetName(),
                        position=pos,
                        orientation=ori,
                        timestamp=now,
                    )
                except Exception:
                    continue

            return poses

        except Exception as exc:
            self._log(f"[WARN] StateMonitor: pose query failed: {exc}")
            return {}

    def update(self) -> Optional[TriggerEvent]:
        """Check for significant pose changes. Returns TriggerEvent if threshold exceeded.

        Must be called on the main thread.
        """
        # Auto-discover on first call if no paths specified.
        if not self._auto_discovered and not self._tracked_paths:
            discovered = self.auto_discover_objects()
            if discovered:
                self._tracked_paths = discovered[:20]  # Cap at 20 objects.
            self._auto_discovered = True

        current_poses = self.get_current_poses()
        if not current_poses:
            return None

        now = time.time()

        # First update: store baseline, no trigger.
        if not self._initialized:
            self._prev_poses = current_poses
            self._initialized = True
            return TriggerEvent(
                trigger_type="initial",
                changed_objects=list(current_poses.keys()),
                max_displacement=0.0,
                timestamp=now,
            )

        # Cooldown check.
        if (now - self._last_trigger_time) < self._cooldown:
            return None

        # Compare poses.
        changed_objects: List[str] = []
        max_disp = 0.0

        for path, cur in current_poses.items():
            prev = self._prev_poses.get(path)
            if prev is None:
                changed_objects.append(path)
                continue

            disp = math.sqrt(
                (cur.position[0] - prev.position[0]) ** 2
                + (cur.position[1] - prev.position[1]) ** 2
                + (cur.position[2] - prev.position[2]) ** 2
            )

            if disp > self._pos_threshold:
                changed_objects.append(path)
                max_disp = max(max_disp, disp)

        if changed_objects:
            self._prev_poses = current_poses
            self._last_trigger_time = now
            return TriggerEvent(
                trigger_type="gt_change",
                changed_objects=changed_objects,
                max_displacement=max_disp,
                timestamp=now,
            )

        return None

    def get_gt_state_json(self) -> Dict[str, Any]:
        """Return the current GT state as a JSON-serializable dict for the VLM."""
        poses = self.get_current_poses()
        objects = {}
        for path, pose in poses.items():
            objects[pose.name] = {
                "prim_path": pose.prim_path,
                "position": list(pose.position),
                "orientation": list(pose.orientation),
                "timestamp": pose.timestamp,
            }
        return {"ground_truth_objects": objects, "timestamp": time.time()}

    def get_prev_poses(self) -> Dict[str, ObjectPose]:
        """Return the last-stored baseline poses."""
        return dict(self._prev_poses)
