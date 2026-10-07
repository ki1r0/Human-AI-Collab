"""Thin physical skill adapter around the custom M1 RoCo environment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .contracts import Observation, SkillResult, new_observation_id
from .evaluator import SeatMeasurement


class RocoTaskAdapter:
    backend_name = "roco_isaaclab_m1"
    # Exact USD/Isaac wxyz form of MagicAssembly's XYZ=(-90, 180, 0).
    _M1_HUB_QUAT_WXYZ = (0.0, 0.0, 0.7071067812, 0.7071067812)
    # RoCo's R1 rule skill uses this fixed arm_link6 -> tool TCP offset.
    _R1_TCP_OFFSET_M = (0.0079, 0.0, 0.09089)
    # Hub orientation composed with the tool's 180-degree x flip.
    _R1_GRIPPER_QUAT_WXYZ = (0.0, 0.0, 0.7071067812, -0.7071067812)
    # MagicAssembly's final child-local override for this exact pair is
    # (0, 43.41859, 31.0) authored units.  At the task's 0.002 scale this is
    # the root pose used by the canonical sequence.  The un-overridden
    # plug/socket fit is kept in calibration_manifest.json for comparison.
    _SOCKET_CENTER_Y_M = 0.08683718
    _SEATED_ROOT_Z_OFFSET_M = 0.062
    # Collision-on R1 calibration for the requested one-inside/one-outside
    # annular clamp.  These values are task-scoped controller waypoints, not
    # evaluator ground truth and not a rigid-body pose override.
    _GRASP_PAIR_OFFSET_M = (0.0, 0.0975, 0.0)
    _GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
    _GRASP_APPROACH_Z_M = 0.25
    _GRASP_CLOSE_Z_M = 0.037
    _GRASP_LIFT_Z_M = 0.15
    _GRASP_APPROACH_OPENING = 0.045
    _GRASP_CLOSED_OPENING = 0.020

    def __init__(self, env: Any, *, frame_dir: str | Path | None = None, video_path: str | Path | None = None, fault_profile: dict[str, Any] | None = None) -> None:
        import torch

        self.env = env
        self.torch = torch
        self.frame_dir = Path(frame_dir) if frame_dir else None
        if self.frame_dir:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.episode_id = ""
        self.fault_profile = fault_profile or {}
        self.last_public: dict[str, Any] = {}
        self.video_path = Path(video_path) if video_path else None
        self.video_writer = None
        # RoCo records one synchronized three-camera frame per control step.
        # Keep the M1 camera artifact on the same 20 Hz cadence instead of
        # silently downsampling the live rollout.
        self.video_every = 1
        self.video_index = 0
        self.video_frame_count = 0
        self.video_error: str | None = None
        self._stable_time_s = 0.0
        self._seat_quat = None
        if self.video_path:
            import imageio.v2 as imageio
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = imageio.get_writer(str(self.video_path), fps=20, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None, ffmpeg_log_level="error")

    def _update_stability(self) -> None:
        """Accumulate a real low-speed/contact settling window for the evaluator."""
        try:
            state = self.env.root_states()["hub"]
            speed = float(self.torch.linalg.vector_norm(state[0, 7:10]).item())
            contact = self._contact_observation()[0]
            dt = float(self.env.cfg.sim_dt) * int(self.env.cfg.decimation)
            self._stable_time_s = self._stable_time_s + dt if contact and speed <= 0.01 else 0.0
        except Exception:
            self._stable_time_s = 0.0

    def _contact_observation(self) -> tuple[bool, float | None, float | None]:
        """Return contact validity, minimum separation and force magnitude."""
        sensor = getattr(self.env, "hub_contact", None)
        if sensor is None:
            return False, None, None
        try:
            data = sensor.data
            matrix = getattr(data, "force_matrix_w", None)
            force = float(self.torch.linalg.vector_norm(matrix).item()) if matrix is not None and matrix.numel() else 0.0
            separation: float | None = None
            # ContactSensor's high-capacity point view is enabled by the M1
            # scene config.  Keep this query defensive: if a future Isaac
            # build does not expose the view, force remains a valid contact
            # signal and penetration stays explicitly unavailable.
            view = getattr(sensor, "contact_physx_view", None)
            if view is not None:
                _forces, _points, _normals, distances, counts, _starts = view.get_contact_data(
                    dt=float(self.env.cfg.sim_dt)
                )
                count = int(counts.reshape(-1)[0].item()) if counts.numel() else 0
                if count > 0:
                    values = distances.reshape(-1)[:count]
                    finite = values[self.torch.isfinite(values)]
                    if finite.numel():
                        separation = float(finite.min().item())
            return bool(force > 1.0e-3 or separation is not None), separation, force
        except Exception:
            return False, None, None

    def reset(self, episode_id: str, seed: int, mode: str) -> Observation:
        self.episode_id = episode_id
        self.index = 0
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.reset()
        # Fault injection is reset-only and is recorded privately by the
        # runner; no episode action may write the object root pose.
        offset = self.fault_profile.get("offset_m", [0.0, 0.0, 0.0])
        tilt_deg = float(self.fault_profile.get("tilt_deg", 0.0))
        if any(float(v) != 0.0 for v in offset) or tilt_deg:
            state = self.env.hub.data.default_root_state.clone()
            state[:, :3] += self.torch.tensor(offset, device=self.env.device)
            if tilt_deg:
                theta = math.radians(tilt_deg) / 2.0
                tilt = self.torch.tensor([math.cos(theta), 0.0, math.sin(theta), 0.0], device=self.env.device)
                base = state[:, 3:7]
                w1, x1, y1, z1 = tilt
                w2, x2, y2, z2 = base.unbind(-1)
                state[:, 3:7] = self.torch.stack(
                    (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
                    dim=-1,
                )
            self.env.hub.write_root_state_to_sim(state)
        self._stable_time_s = 0.0
        self._seat_quat = self.torch.tensor(self._M1_HUB_QUAT_WXYZ, device=self.env.device)
        return self.observe(episode_id, 0)

    def _save_image(self, value: Any, name: str) -> str | None:
        if self.frame_dir is None or value is None:
            return None
        try:
            import numpy as np
            from PIL import Image

            if hasattr(value, "detach"):
                array = value[0].detach().cpu().numpy() if getattr(value, "ndim", 0) == 4 else value.detach().cpu().numpy()
            else:
                array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(f"expected HWC image, got shape {array.shape}")
            if array.shape[-1] == 4:
                array = array[..., :3]
            if array.dtype != np.uint8:
                # Isaac RGB cameras normally return uint8.  Keep the helper
                # correct for normalized test doubles without changing the
                # public observation contract.
                array = np.clip(array, 0.0, 255.0).astype("uint8")
            path = self.frame_dir / f"{self.index:04d}_{name}.png"
            Image.fromarray(array).save(path)
            # Keep an absolute local reference so the provider boundary can
            # load the exact frame even when the launcher is not run from the
            # repository root.  The image itself remains in the run artifact;
            # logs contain only this public frame reference.
            return str(path)
        except Exception:
            return None

    def _append_video(self, value: Any) -> None:
        if self.video_writer is None or value is None or self.video_index % self.video_every:
            self.video_index += 1
            return
        try:
            import numpy as np
            values = value if isinstance(value, (list, tuple)) else [value]
            frames = []
            for item in values:
                if item is None:
                    continue
                frame = item[0].detach().cpu().numpy() if hasattr(item, "detach") and item.ndim == 4 else item.detach().cpu().numpy()
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame.astype("uint8", copy=False))
            if not frames:
                raise ValueError("no camera frame")
            height = max(frame.shape[0] for frame in frames)
            width = max(frame.shape[1] for frame in frames)
            padded = []
            for frame in frames:
                canvas = np.zeros((height, width, 3), dtype="uint8")
                canvas[: frame.shape[0], : frame.shape[1]] = frame
                padded.append(canvas)
            if len(padded) == 4:
                top = np.concatenate(padded[:2], axis=1)
                bottom = np.concatenate(padded[2:], axis=1)
                self.video_writer.append_data(np.concatenate((top, bottom), axis=0))
            else:
                self.video_writer.append_data(np.concatenate(padded, axis=1))
            self.video_frame_count += 1
        except Exception as exc:
            # Preserve the physical rollout, but expose recorder failures in
            # the metrics instead of silently producing an empty MP4.
            self.video_error = f"{type(exc).__name__}: {exc}"
        self.video_index += 1

    @staticmethod
    def _values(value: Any) -> list[float]:
        if value is None:
            return []
        return [float(v) for v in value[0].detach().cpu().reshape(-1).tolist()]

    def observe(self, episode_id: str, index: int, recent_skill: dict[str, Any] | None = None) -> Observation:
        self.index = int(index)
        raw = getattr(self.env, "_last_obs", {})
        self._append_video([
            raw.get("head_rgb"), raw.get("overhead_rgb"),
            raw.get("left_hand_rgb"), raw.get("right_hand_rgb"),
        ])
        frames: dict[str, str] = {}
        for name in ("head_rgb", "left_hand_rgb", "right_hand_rgb"):
            ref = self._save_image(raw.get(name), name)
            if ref:
                frames[name] = ref
        qpos = self._values(raw.get("left_arm_joint_pos")) + self._values(raw.get("right_arm_joint_pos"))
        qvel = self._values(raw.get("left_arm_joint_vel")) + self._values(raw.get("right_arm_joint_vel"))
        gripper = {"left": (self._values(raw.get("left_gripper_joint_pos")) or [0.0])[0], "right": (self._values(raw.get("right_gripper_joint_pos")) or [0.0])[0]}
        self.last_public = {"frames": frames, "qpos": qpos, "qvel": qvel, "gripper": gripper}
        return Observation(
            episode_id=episode_id,
            observation_id=new_observation_id(episode_id, index),
            timestamp=float(index),
            frames=frames,
            qpos=qpos,
            qvel=qvel,
            gripper=gripper,
            sensor_availability={
                "rgb": bool(frames),
                "qpos": bool(qpos),
                "qvel": bool(qvel),
                "gripper": True,
                "force": False,
                "grasp_verification": callable(getattr(self.env, "grasp_verification", None)),
            },
            recent_skill=recent_skill,
        )

    def _step(self, count: int) -> None:
        action = self.torch.zeros((1, 14), device=self.env.device)
        for _ in range(int(count)):
            self.env.step(action)
            self.index += 1
            self._update_stability()
            raw = getattr(self.env, "_last_obs", {})
            self._append_video([
                raw.get("head_rgb"), raw.get("overhead_rgb"),
                raw.get("left_hand_rgb"), raw.get("right_hand_rgb"),
            ])

    def prepare_free_seat_trial(self, *, velocity_mps: float = -0.008) -> None:
        """Reset-only placement for a collision-on interface trial.

        This is a calibration diagnostic, not a robot-policy action.  The
        dynamic source receives one initial velocity and is then advanced by
        PhysX; no per-step pose write or snap is used.
        """
        casing = self.env.root_states()["casing"][0, :3].clone()
        state = self.env.hub.data.default_root_state.clone()
        state[:, :3] = casing + self.torch.tensor([0.0, self._SOCKET_CENTER_Y_M, 0.22], device=self.env.device)
        state[:, 3:7] = self.torch.tensor(self._M1_HUB_QUAT_WXYZ, device=self.env.device)
        state[:, 7:] = 0.0
        self.env.hub.write_root_state_to_sim(state)
        velocity = self.torch.zeros((self.env.scene.num_envs, 6), device=self.env.device)
        velocity[:, 2] = float(velocity_mps)
        self.env.hub.write_root_velocity_to_sim(velocity)
        self.env.sim.forward()
        self._stable_time_s = 0.0
        self._seat_quat = state[0, 3:7].clone()

    def _pose(self, xyz: list[float], *, arm: str = "left") -> None:
        # Skill waypoints describe the Hub root; convert them to the R1
        # arm_link6 target using the tool transform used by the pinned RoCo
        # rule skill.  The world->base conversion is performed in _ik_target.
        tcp_offset = self.torch.tensor(self._R1_TCP_OFFSET_M, dtype=self.torch.float32, device=self.env.device)
        position = self.torch.tensor([xyz], dtype=self.torch.float32, device=self.env.device) + tcp_offset
        orientation = self.torch.tensor([self._R1_GRIPPER_QUAT_WXYZ], dtype=self.torch.float32, device=self.env.device)
        self.env.set_pose_target(arm, position, orientation)

    def _calibrated_grasp_pose(self, hub_root: Any, z_offset_m: float) -> None:
        """Command the collision-on one-inside/one-outside R1 grasp pose."""
        pair = self.torch.tensor(self._GRASP_PAIR_OFFSET_M, dtype=self.torch.float32, device=self.env.device)
        position = hub_root + pair + self.torch.tensor([0.0, 0.0, float(z_offset_m)], device=self.env.device)
        orientation = self.torch.tensor([self._GRASP_QUAT_WXYZ], dtype=self.torch.float32, device=self.env.device)
        self.env.set_pose_target("left", position.unsqueeze(0), orientation)

    def _held_status(self) -> tuple[str, str]:
        """Return a held verdict only when a public grasp signal is present.

        The calibrated M1 scene exposes a categorical Hub-to-gripper contact
        and object-following callback.  A reached TCP and a commanded gripper
        position alone are never allowed to imply ``HELD_CONFIRMED``; a future
        end-effector can implement the same callback without changing this
        skill contract.
        """
        checker = getattr(self.env, "grasp_verification", None)
        if callable(checker):
            try:
                verdict = checker()
                if isinstance(verdict, tuple) and len(verdict) == 2:
                    status, reason = verdict
                else:
                    status, reason = verdict, "grasp_verification_callback"
                status = str(status).upper()
                if status in {"HELD_CONFIRMED", "HELD_FAILED", "UNKNOWN"}:
                    return status, str(reason)
            except Exception:
                return "UNKNOWN", "grasp_verification_callback_error"
        return "UNKNOWN", "public_grasp_sensor_unavailable"

    def execute_skill(self, observation: Observation, skill_id: str, args: dict[str, Any]) -> SkillResult:
        start = self.index
        held_status = "UNKNOWN"
        held_reason: str | None = None
        seat_status = "UNKNOWN"
        try:
            hub = self.env.root_states()["hub"][0, :3].detach().cpu().tolist()
            casing = self.env.root_states()["casing"][0, :3].detach().cpu().tolist()
            if skill_id == "pick":
                hub_root = self.env.root_states()["hub"][0, :3].clone()
                self.env.set_gripper(self._GRASP_APPROACH_OPENING)
                self._calibrated_grasp_pose(hub_root, self._GRASP_APPROACH_Z_M)
                self._step(80)
                self._calibrated_grasp_pose(hub_root, self._GRASP_CLOSE_Z_M)
                self._step(100)
                self.env.set_gripper(self._GRASP_CLOSED_OPENING)
                self._step(60)
                begin_check = getattr(self.env, "begin_grasp_verification", None)
                if callable(begin_check):
                    begin_check()
                self._calibrated_grasp_pose(hub_root, self._GRASP_LIFT_Z_M)
                self._step(80)
                held_status, held_reason = self._held_status()
                self.env.clear_pose_target()
                if held_status != "HELD_CONFIRMED":
                    return SkillResult(
                        observation.episode_id,
                        observation.observation_id,
                        skill_id,
                        "FAILED",
                        held_status=held_status,
                        seat_status="UNKNOWN",
                        steps=self.index - start,
                        failure_code="HELD_UNVERIFIED",
                        feedback={"verification_reason": held_reason},
                    )
            elif skill_id == "preinsert":
                # The target frame is from the calibrated isolated trial.  It
                # is intentionally kept as a pending calibration parameter.
                self._pose([casing[0], casing[1] + self._SOCKET_CENTER_Y_M, casing[2] + 0.16])
                self._step(120)
            elif skill_id == "insert":
                self._pose([casing[0], casing[1] + self._SOCKET_CENTER_Y_M, casing[2] + self._SEATED_ROOT_Z_OFFSET_M])
                self._step(100)
                seat_status = "SEAT_CANDIDATE"
            elif skill_id == "verify":
                self._step(20)
            elif skill_id == "release_retract":
                self.env.set_gripper(0.04)
                self._step(50)
                self._pose([casing[0], casing[1] + self._SOCKET_CENTER_Y_M, casing[2] + 0.16])
                self._step(100)
            else:
                return SkillResult(observation.episode_id, observation.observation_id, skill_id, "FAILED", failure_code="UNSUPPORTED_SKILL")
            self.env.clear_pose_target()
            feedback = {"verification_reason": held_reason} if held_reason else {}
            return SkillResult(
                observation.episode_id,
                observation.observation_id,
                skill_id,
                "SUCCEEDED",
                held_status=held_status,
                seat_status=seat_status,
                steps=self.index - start,
                feedback=feedback,
            )
        except Exception as exc:
            self.env.clear_pose_target()
            return SkillResult(observation.episode_id, observation.observation_id, skill_id, "FAILED", failure_code=type(exc).__name__, steps=self.index - start)

    def evaluator_measurement(self) -> SeatMeasurement:
        state = self.env.root_states()
        hub = state["hub"][0, :3]
        casing = state["casing"][0, :3]
        # This is a deliberately conservative observable proxy.  Collision
        # contacts, penetration and settling still require the Isaac evaluator;
        # unavailable fields remain unavailable instead of being guessed.
        radial = float(self.torch.linalg.vector_norm((hub - casing)[:2] - self.torch.tensor([0.0, self._SOCKET_CENTER_Y_M], device=hub.device)).item())
        axial = float((hub[2] - casing[2]).item())
        contact_valid, separation, _force = self._contact_observation()
        penetration = max(0.0, -float(separation)) if separation is not None else None
        speed = float(self.torch.linalg.vector_norm(state["hub"][0, 7:10]).item())
        quat = state["hub"][0, 3:7]
        target = self._seat_quat if self._seat_quat is not None else quat
        dot = float(self.torch.abs(self.torch.dot(quat, target)).clamp(0.0, 1.0).item())
        tilt = math.degrees(2.0 * math.acos(dot))
        released = bool(float(getattr(self.env, "_gripper_target", 0.0)) >= 0.02)
        return SeatMeasurement(
            axial_depth_m=axial,
            radial_error_m=radial,
            tilt_deg=tilt,
            penetration_m=penetration,
            released=released,
            stable=bool(contact_valid and speed <= 0.01),
            contact_valid=contact_valid,
            settle_speed_mps=speed,
            settle_window_s=self._stable_time_s,
        )

    def human_jog(self, delta_m: Any) -> None:
        values = [float(value) for value in delta_m]
        if len(values) != 3 or any(abs(value) > 0.01 for value in values):
            raise ValueError("human jog is limited to 10 mm per axis")
        body_cfg = self.env.left_arm_cfg if self.env._active_arm == "left" else self.env.right_arm_cfg
        body_id = body_cfg.body_ids[0]
        current = self.env.robot.data.body_state_w[:, body_id, :3].clone()
        orientation = self.env.robot.data.body_state_w[:, body_id, 3:7].clone()
        target = current + self.torch.tensor([values], device=self.env.device)
        self.env.set_pose_target(self.env._active_arm, target, orientation)
        self._step(20)
        self.env.clear_pose_target()

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()


__all__ = ["RocoTaskAdapter"]
