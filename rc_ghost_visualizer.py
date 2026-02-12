from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import omni.usd
from pxr import Gf, Usd, UsdGeom


class GhostVisualizer:
    """Belief ghost visualizer (main thread only).

    Uses native PXR (Usd/UsdGeom/Gf) APIs so it does not depend on `omni.isaac.core`.
    All USD writes must happen on the main Kit thread (e.g., physics callback).
    """

    def __init__(
        self,
        prim_path: str = "/Mind/Ghost_Orange",
        radius: float = 0.05,
        color: Sequence[float] = (1.0, 0.5, 0.0),
        opacity: float = 0.35,
        logger=print,
    ) -> None:
        self.prim_path = prim_path
        self.radius = float(radius)
        self.color = (float(color[0]), float(color[1]), float(color[2]))
        self.opacity = float(opacity)
        self._log = logger

        self.enabled = False
        self._prim = None
        self._imageable: Optional[UsdGeom.Imageable] = None

        try:
            self._prim = self._get_or_create_sphere()
            self._imageable = UsdGeom.Imageable(self._prim)
            self.enabled = True
            self._log(f"[INFO] GhostVisualizer ready (PXR): {self.prim_path}")
        except Exception as exc:
            self.enabled = False
            self._prim = None
            self._imageable = None
            self._log(f"[WARN] GhostVisualizer disabled (JSON-only): {exc}")

    def sync_ghosts(self, belief_snapshot: Dict[str, Any]) -> None:
        """Move/hide the ghost based on belief snapshot."""
        if not self.enabled or self._prim is None or self._imageable is None:
            return

        orange = self._get_orange_entry(belief_snapshot)
        if orange is None:
            self._set_visibility(False)
            return

        belief_status = str(orange.get("belief_status", "")).strip().lower()

        # If the belief is "contained", we can visualize at the container center (no need for 3D inference).
        if belief_status == "contained":
            container = str(orange.get("inferred_container", "")).strip().lower()
            pos = self._get_container_center(container)
            if pos is None:
                self._set_visibility(False)
                return
            self._set_translate(pos)
            self._set_visibility(True)
            return

        # Otherwise, show only when belief indicates visible=True or belief_status=="visible",
        # and we have a 3D position in the snapshot.
        visible = self._get_visible(orange)
        pos = self._get_position(orange)
        if (not visible) or pos is None:
            self._set_visibility(False)
            return

        self._set_translate(pos)
        self._set_visibility(True)

    def _get_or_create_sphere(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is not ready yet.")

        # Ensure parent prim exists.
        parent = "/".join(self.prim_path.rstrip("/").split("/")[:-1])
        if parent:
            # Define intermediate prims as Xforms.
            cur = ""
            for part in parent.strip("/").split("/"):
                cur = f"{cur}/{part}" if cur else f"/{part}"
                if not stage.GetPrimAtPath(cur).IsValid():
                    stage.DefinePrim(cur, "Xform")

        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim or not prim.IsValid():
            prim = stage.DefinePrim(self.prim_path, "Sphere")

        sphere = UsdGeom.Sphere(prim)
        sphere.GetRadiusAttr().Set(self.radius)

        gprim = UsdGeom.Gprim(prim)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*self.color)])
        try:
            gprim.GetDisplayOpacityAttr().Set([float(self.opacity)])
        except Exception:
            # Some prims may not support displayOpacity; color still helps.
            pass

        # Start hidden until first valid sync.
        UsdGeom.Imageable(prim).MakeInvisible()
        return prim

    def _set_translate(self, position: Sequence[float]) -> None:
        xform_api = UsdGeom.XformCommonAPI(self._prim)
        xform_api.SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    def _set_visibility(self, visible: bool) -> None:
        if visible:
            self._imageable.MakeVisible()
        else:
            self._imageable.MakeInvisible()

    def _get_orange_entry(self, belief_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(belief_snapshot, dict):
            return None
        if isinstance(belief_snapshot.get("objects"), dict):
            orange = belief_snapshot["objects"].get("orange")
            return orange if isinstance(orange, dict) else None
        orange = belief_snapshot.get("orange")
        return orange if isinstance(orange, dict) else None

    def _get_visible(self, orange: Dict[str, Any]) -> bool:
        if "visible" in orange:
            return bool(orange["visible"])
        return orange.get("belief_status") == "visible"

    def _get_position(self, orange: Dict[str, Any]) -> Optional[Sequence[float]]:
        for key in ("estimated_pos_3d", "position", "location"):
            val = orange.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                return val
        return None

    def _get_container_center(self, container_name: str) -> Optional[Sequence[float]]:
        """Best-effort: find container prim in the stage and return its world-space AABB center."""
        if not container_name:
            return None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None

        # Allow explicit override via env vars for deterministic behavior.
        env_map = {
            "bucket": os.getenv("BUCKET_PRIM_PATH", "").strip(),
            "basket": os.getenv("BASKET_PRIM_PATH", "").strip(),
        }
        prim_path = env_map.get(container_name, "")
        prim = stage.GetPrimAtPath(prim_path) if prim_path else None

        # Heuristic search by name if no explicit mapping or invalid path.
        if prim is None or (not prim.IsValid()):
            keywords = [container_name]
            # Common synonym in the sample scene assets.
            if container_name == "bucket":
                keywords.append("utilitybucket")
                keywords.append("pail")

            prim = None
            for p in stage.Traverse():
                name = p.GetName().lower()
                if any(k in name for k in keywords):
                    prim = p
                    break

        if prim is None or (not prim.IsValid()):
            return None

        try:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            aabb = bbox.ComputeAlignedRange()
            mn = aabb.GetMin()
            mx = aabb.GetMax()
            center = (mn + mx) * 0.5
            return (float(center[0]), float(center[1]), float(center[2]))
        except Exception:
            # Fall back to xform translate if bounds aren't available.
            try:
                xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                m = xform_cache.GetLocalToWorldTransform(prim)
                t = m.ExtractTranslation()
                return (float(t[0]), float(t[1]), float(t[2]))
            except Exception:
                return None
