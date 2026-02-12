"""Franka Panda robot controller with RMPFlow priority and DifferentialIK fallback.

Uses tuned stiffness/damping values from AI-CPS industrial benchmarks.
"""
from __future__ import annotations

import re
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import omni.usd
from pxr import UsdPhysics

from rc_config import (
    ENABLE_FRANKA_CONTROL,
    FRANKA_HOME_JOINT_POS,
    PREFER_DIFFIK,
    ROBOT_PRIM_EXPR,
    ROBOT_PRIM_PATH,
    ROBOT_FOREARM_DAMPING,
    ROBOT_FOREARM_STIFFNESS,
    ROBOT_HAND_DAMPING,
    ROBOT_HAND_STIFFNESS,
    ROBOT_SHOULDER_DAMPING,
    ROBOT_SHOULDER_STIFFNESS,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
)


@dataclass
class ActionStatus:
    status: str  # "idle"|"running"|"done"|"error"
    detail: str = ""


def _find_robot_prim_path() -> str | None:
    """Resolve a Franka/Panda robot prim path from config or by safe heuristics."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    if ROBOT_PRIM_PATH:
        prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if prim and prim.IsValid():
            return ROBOT_PRIM_PATH

    if ROBOT_PRIM_EXPR:
        try:
            import isaaclab.sim.utils as sim_utils
            paths = sim_utils.find_matching_prim_paths(ROBOT_PRIM_EXPR)
            if paths:
                return paths[0]
        except Exception:
            pass

    try:
        from pxr import Usd
        joint_pat = re.compile(r"^panda_joint([1-7])$", re.IGNORECASE)
        finger_pat = re.compile(r"^panda_finger_joint", re.IGNORECASE)

        def _looks_like_panda(root_prim) -> bool:
            if not root_prim.IsValid() or (not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
                return False
            found = set()
            has_fingers = False
            for p in Usd.PrimRange(root_prim):
                name = p.GetName()
                m = joint_pat.match(name)
                if m:
                    found.add(int(m.group(1)))
                    continue
                if finger_pat.match(name):
                    has_fingers = True
            return len(found) >= 7 and has_fingers

        for prim in stage.Traverse():
            if _looks_like_panda(prim):
                return prim.GetPath().pathString
    except Exception:
        pass

    return None


class RobotController:
    """Non-blocking Franka controller with RMPFlow + DifferentialIK fallback.

    Priority:
    1. RMPFlow (Lula) for smooth task-space motions.
    2. DifferentialIK (Jacobian pseudo-inverse) as fallback.

    Applies tuned stiffness/damping from rc_config for jitter-free operation.
    """

    def __init__(self, *, logger=print) -> None:
        self._log = logger
        self.enabled = False
        self.controller_type = "none"  # "rmpflow" | "diffik" | "none"

        self.prim_path: str | None = None
        self._robot = None
        self._rmp = None
        self._diffik = None
        self._diffik_cfg = None
        self._active_joint_ids: list[int] = []
        self._finger_joint_ids: list[int] = []
        self._ee_body_idx: int | None = None

        self._pending_action: Dict[str, Any] | None = None
        self._action_started_ts: float = 0.0
        self._status = ActionStatus(status="idle")

        if not ENABLE_FRANKA_CONTROL:
            self._log("[WARN] RobotController disabled by ENABLE_FRANKA_CONTROL=0.")
            return

        self.prim_path = _find_robot_prim_path()
        if not self.prim_path:
            self._log("[WARN] RobotController: robot prim not found. Set ROBOT_PRIM_PATH or ROBOT_PRIM_EXPR.")
            return

        try:
            self._ensure_sim_context()
            self._init_robot()
            self.enabled = True
            self._log(f"[INFO] RobotController ready ({self.controller_type}): {self.prim_path}")
        except Exception as exc:
            self.enabled = False
            self._log(f"[WARN] RobotController disabled: {exc}")

    @property
    def status(self) -> ActionStatus:
        return self._status

    def set_action(self, action: Dict[str, Any]) -> None:
        """Set a new action dict (non-blocking)."""
        if not self.enabled:
            return
        if not isinstance(action, dict):
            return
        self._pending_action = action
        self._action_started_ts = time.time()
        self._status = ActionStatus(status="running", detail=str(action.get("type") or ""))

    def update(self) -> ActionStatus:
        """Apply the current action targets (call every frame on main thread)."""
        if not self.enabled or self._robot is None:
            return self._status

        if not getattr(self._robot, "is_initialized", False):
            return self._status

        try:
            from isaacsim.core.api.simulation_context import SimulationContext
            dt = float(SimulationContext.instance().get_physics_dt())
        except Exception:
            dt = 1.0 / 60.0
        try:
            self._robot.update(dt)
        except Exception:
            pass

        action = self._pending_action or {"type": "noop", "args": {}}
        a_type = str(action.get("type") or "noop")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        try:
            if a_type == "noop":
                self._status = ActionStatus(status="idle", detail="noop")
                return self._status
            elif a_type == "home":
                self._apply_home()
                return self._maybe_done(timeout_sec=5.0)
            elif a_type == "open_gripper":
                self._apply_gripper(open_=True)
                return self._maybe_done(timeout_sec=2.0)
            elif a_type == "close_gripper":
                self._apply_gripper(open_=False)
                return self._maybe_done(timeout_sec=2.0)
            elif a_type == "inspect":
                self._apply_move_ee_pose(pos=(0.5, 0.0, 0.6), quat=(1.0, 0.0, 0.0, 0.0))
                return self._maybe_done(timeout_sec=6.0)
            elif a_type == "move_ee_pose":
                pos = args.get("pos")
                quat = args.get("quat")
                if not _is_vec3(pos):
                    self._status = ActionStatus(status="error", detail="move_ee_pose missing valid pos")
                    return self._status
                if quat is None or not _is_quat_wxyz(quat):
                    quat = (1.0, 0.0, 0.0, 0.0)
                self._apply_move_ee_pose(pos=pos, quat=quat)
                return self._maybe_done(timeout_sec=8.0)
            else:
                self._status = ActionStatus(status="error", detail=f"unknown action type: {a_type}")
                return self._status
        except Exception as exc:
            self._status = ActionStatus(status="error", detail=str(exc))
            return self._status

    # --- internals ---

    def _ensure_sim_context(self) -> None:
        try:
            from isaaclab.sim import SimulationCfg, SimulationContext
            try:
                _ = SimulationContext.instance()
            except Exception:
                SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cpu"))
        except Exception as exc:
            raise RuntimeError(f"SimulationContext not available: {exc}") from exc

    def _init_robot(self) -> None:
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
        import isaaclab.sim as sim_utils

        # Build config with tuned physics values.
        cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        cfg.prim_path = self.prim_path
        cfg.spawn = None  # Reference existing prim, do not spawn.

        # Apply tuned stiffness/damping from AI-CPS research.
        if "panda_shoulder" in cfg.actuators:
            cfg.actuators["panda_shoulder"].stiffness = ROBOT_SHOULDER_STIFFNESS
            cfg.actuators["panda_shoulder"].damping = ROBOT_SHOULDER_DAMPING
        if "panda_forearm" in cfg.actuators:
            cfg.actuators["panda_forearm"].stiffness = ROBOT_FOREARM_STIFFNESS
            cfg.actuators["panda_forearm"].damping = ROBOT_FOREARM_DAMPING
        if "panda_hand" in cfg.actuators:
            cfg.actuators["panda_hand"].stiffness = ROBOT_HAND_STIFFNESS
            cfg.actuators["panda_hand"].damping = ROBOT_HAND_DAMPING

        # Apply solver iteration overrides for stability.
        if cfg.spawn is not None and hasattr(cfg.spawn, "articulation_props"):
            if cfg.spawn.articulation_props is not None:
                cfg.spawn.articulation_props.solver_position_iteration_count = SOLVER_POSITION_ITERATIONS
                cfg.spawn.articulation_props.solver_velocity_iteration_count = SOLVER_VELOCITY_ITERATIONS

        self._robot = Articulation(cfg=cfg)

        # Resolve joints.
        self._active_joint_ids, _ = self._robot.find_joints(["panda_joint[1-7]"], preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(["panda_finger_joint.*"], preserve_order=True)

        # --- Controller initialization (priority: RMPFlow > DiffIK) ---
        if not PREFER_DIFFIK:
            self._try_init_rmpflow()

        if self._rmp is None:
            self._try_init_diffik()

        if self._rmp is None and self._diffik is None and PREFER_DIFFIK:
            self._try_init_rmpflow()

        if self._rmp is not None:
            self.controller_type = "rmpflow"
        elif self._diffik is not None:
            self.controller_type = "diffik"
        else:
            self.controller_type = "joint_pos_only"
            self._log("[WARN] No task-space controller available; move_ee_pose will use joint-level only.")

    def _try_init_rmpflow(self) -> None:
        try:
            from isaaclab.controllers.rmp_flow import RmpFlowController
            from isaaclab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG

            self._rmp = RmpFlowController(FRANKA_RMPFLOW_CFG, device="cpu")
            self._rmp.initialize(self.prim_path)
            self._log("[INFO] RMPFlow controller initialized.")
        except Exception as exc:
            self._rmp = None
            self._log(f"[WARN] RMPFlow unavailable: {exc}")

    def _try_init_diffik(self) -> None:
        try:
            from isaaclab.controllers.differential_ik import DifferentialIKController
            from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

            self._diffik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="pinv",
                ik_params={"k_val": 1.0},
            )
            self._diffik = DifferentialIKController(self._diffik_cfg, num_envs=1, device="cpu")
            self._log("[INFO] DifferentialIK controller initialized (pinv).")
        except Exception as exc:
            self._diffik = None
            self._log(f"[WARN] DifferentialIK unavailable: {exc}")

    def _apply_home(self) -> None:
        import torch
        joint_pos = self._robot.data.joint_pos.clone()
        for name_expr, value in FRANKA_HOME_JOINT_POS.items():
            joint_ids, _ = self._robot.find_joints(name_expr)
            if not joint_ids:
                continue
            joint_pos[:, joint_ids] = float(value)
        self._robot.set_joint_position_target(joint_pos)
        self._robot.write_data_to_sim()

    def _apply_gripper(self, *, open_: bool) -> None:
        if not self._finger_joint_ids:
            return
        joint_pos = self._robot.data.joint_pos.clone()
        target = 0.04 if open_ else 0.0
        joint_pos[:, self._finger_joint_ids] = float(target)
        self._robot.set_joint_position_target(joint_pos)
        self._robot.write_data_to_sim()

    def _apply_move_ee_pose(self, *, pos: Sequence[float], quat: Sequence[float]) -> None:
        import torch

        if self._rmp is not None:
            self._apply_move_rmpflow(pos=pos, quat=quat)
        elif self._diffik is not None:
            self._apply_move_diffik(pos=pos, quat=quat)
        else:
            raise RuntimeError("No task-space controller available for move_ee_pose.")

    def _apply_move_rmpflow(self, *, pos: Sequence[float], quat: Sequence[float]) -> None:
        import torch
        cmd = torch.tensor([[
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        ]])
        self._rmp.set_command(cmd)
        dof_pos, _dof_vel = self._rmp.compute()
        joint_pos = self._robot.data.joint_pos.clone()
        joint_pos[:, self._active_joint_ids] = dof_pos
        self._robot.set_joint_position_target(joint_pos)
        self._robot.write_data_to_sim()

    def _apply_move_diffik(self, *, pos: Sequence[float], quat: Sequence[float]) -> None:
        import torch

        ee_pos_des = torch.tensor([[float(pos[0]), float(pos[1]), float(pos[2])]])
        ee_quat_des = torch.tensor([[float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]])
        self._diffik.set_command(torch.cat([ee_pos_des, ee_quat_des], dim=1))

        # Get current EE state from the robot.
        ee_pos_curr = self._robot.data.body_pos_w[:, -1:, :]  # Last body = EE (approximate)
        ee_quat_curr = self._robot.data.body_quat_w[:, -1:, :]

        # Flatten for controller.
        ee_pos_curr = ee_pos_curr.squeeze(1)
        ee_quat_curr = ee_quat_curr.squeeze(1)

        joint_pos = self._robot.data.joint_pos[:, self._active_joint_ids]
        jacobian = None
        # Try to get Jacobian from PhysX view (version-dependent)
        try:
            physx_view = getattr(self._robot, 'root_physx_view', None) or getattr(self._robot, '_root_physx_view', None)
            if physx_view is not None and hasattr(physx_view, 'get_jacobians'):
                jacobian = physx_view.get_jacobians()
                if jacobian is not None and jacobian.ndim == 4:
                    # Jacobian shape: (num_envs, num_bodies, 6, num_dofs). Take EE body.
                    jacobian = jacobian[:, -1, :, :len(self._active_joint_ids)]
        except Exception:
            jacobian = None

        joint_pos_des = self._diffik.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)

        full_joint_pos = self._robot.data.joint_pos.clone()
        full_joint_pos[:, self._active_joint_ids] = joint_pos_des
        self._robot.set_joint_position_target(full_joint_pos)
        self._robot.write_data_to_sim()

    def _maybe_done(self, *, timeout_sec: float) -> ActionStatus:
        if (time.time() - self._action_started_ts) >= float(timeout_sec):
            self._status = ActionStatus(status="done", detail="timeout")
            self._pending_action = None
            return self._status
        self._status = ActionStatus(status="running", detail=self._status.detail)
        return self._status


# Keep FrankaControlPolicy as an alias for backwards compatibility.
FrankaControlPolicy = RobotController


def _is_vec3(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x)


def _is_quat_wxyz(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 4 and all(_is_num(v) for v in x)


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False
