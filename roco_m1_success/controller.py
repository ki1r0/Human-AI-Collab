"""Minimal deterministic controller for the RoCo M1 assembly proof.

The class reuses the official R1 Differential IK and grasp/mount primitives.
Only the phase schedule and the reducer's release-before-contact trajectory are
local: the upstream reducer path opens the gripper while it is loading the
assembled stack, which drags the stack during release.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from Galaxea_Lab_External.robots.galaxea_rule_policy import GalaxeaRulePolicy


@dataclass(frozen=True)
class Manipulation:
    phase: str
    gear_id: int
    pickup: torch.Tensor
    mount: torch.Tensor
    arm: str
    rotate: bool = False
    reducer_drop: bool = False


class M1AssemblyController(GalaxeaRulePolicy):
    """Official rule controller with a selectable, slower M1 phase schedule."""

    MOVEMENT_S = 1.0
    GRIP_S = 0.75
    INSERT_S = 1.0
    RELEASE_S = 0.75
    RETREAT_S = 1.0
    SETTLE_S = 1.0
    # R31's official-arm-reset trajectory reaches the ring but still leaves a
    # release-side lateral error.  Test a stronger bounded measured correction
    # only after that reset is active; R30's identical gain test was before
    # the reset and is not a valid ablation of this state.
    RING_XY_FEEDBACK_GAIN = 1.6
    RING_XY_FEEDBACK_LIMIT_M = 0.080
    RING_INSERT_GRIP_M = 0.040
    # R38 measured 80--109 N reducer/centre contact while crossing the
    # assembled stack at the ordinary 0.20 m lift.  Raise only the lateral
    # transfer waypoint; the socket approach height is unchanged.
    REDUCER_HIGH_EXTRA_M = 0.150
    REDUCER_ARM = "right"

    def __init__(self, sim, scene, obj_dict, *, phase: str = "full"):
        super().__init__(sim, scene, obj_dict)
        self.phase = phase
        self.event = "stabilize"
        self.event_history: list[dict[str, object]] = []
        self._last_event = None
        self._active_mount_spec: Manipulation | None = None
        self.manipulations: list[Manipulation] = []
        self.total_time_steps = torch.tensor(
            self._ticks(0.5), dtype=torch.int32, device=self.device
        )

    def _ticks(self, seconds: float) -> int:
        raw = int(round(seconds / self.sim_dt))
        return max(5, ((raw + 4) // 5) * 5)

    def _boundaries(self, start: int, durations: tuple[float, ...]) -> torch.Tensor:
        values = [start]
        for duration in durations:
            values.append(values[-1] + self._ticks(duration))
        return torch.tensor(values, dtype=torch.int32, device=self.device)

    def configure_schedule(self) -> None:
        """Build the selected schedule after the runner fixes the initial layout."""
        self.prepare_mounting_plan()
        all_specs = [
            ("planet1", 1, False, False),
            ("planet2", 2, False, False),
            ("planet3", 3, False, False),
            ("center", 4, False, False),
            ("ring", 5, True, False),
            ("reducer", 6, False, True),
        ]
        selected = all_specs if self.phase == "full" else [s for s in all_specs if s[0] == self.phase]
        if not selected:
            raise ValueError(f"Unknown phase: {self.phase}")

        cursor = self._ticks(0.5)
        self.manipulations = []
        for phase, gear_id, rotate, reducer_drop in selected:
            pickup = self._boundaries(
                cursor,
                (self.MOVEMENT_S, self.MOVEMENT_S, self.GRIP_S, self.MOVEMENT_S),
            )
            cursor = int(pickup[-1].item())
            if rotate:
                mount = self._boundaries(
                    cursor,
                    (
                        self.MOVEMENT_S,
                        self.INSERT_S,
                        self.INSERT_S,
                        self.RELEASE_S,
                        self.RETREAT_S,
                    ),
                )
            elif reducer_drop:
                mount = self._boundaries(
                    cursor,
                    (
                        self.MOVEMENT_S,
                        self.INSERT_S,
                        self.RELEASE_S,
                        self.SETTLE_S,
                        self.RETREAT_S,
                    ),
                )
            else:
                mount = self._boundaries(
                    cursor,
                    (self.MOVEMENT_S, self.INSERT_S, self.RELEASE_S, self.RETREAT_S),
                )
            cursor = int(mount[-1].item())
            name = self._object_name(gear_id)
            arm = self._arm_for(name, gear_id)
            self.manipulations.append(
                Manipulation(phase, gear_id, pickup, mount, arm, rotate, reducer_drop)
            )
            cursor += self._ticks(self.SETTLE_S)
        self.total_time_steps = torch.tensor(cursor, dtype=torch.int32, device=self.device)

    @staticmethod
    def _object_name(gear_id: int) -> str:
        if gear_id <= 4:
            return f"sun_planetary_gear_{gear_id}"
        return "ring_gear" if gear_id == 5 else "planetary_reducer"

    def _arm_for(self, object_name: str, gear_id: int) -> str:
        if gear_id == 4:
            return "right"
        if gear_id == 6:
            return self.REDUCER_ARM
        return self.gear_to_pin_map[object_name]["arm"]

    def _entities(self, arm: str):
        if arm == "left":
            return self.left_arm_entity_cfg, self.left_gripper_entity_cfg
        return self.right_arm_entity_cfg, self.right_gripper_entity_cfg

    def pick_up_target_gear(self, gear_id, count_step, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller):
        """Use a root-near reducer grasp without changing mount geometry.

        The upstream helper adds a 50 mm reducer pickup-height offset.  A
        temporary 50 mm reduction of ``grasping_height`` cancels that offset
        only while the helper computes its pickup target; the controller's
        normal grasping height is restored before the next call.
        """
        if gear_id != 6:
            return super().pick_up_target_gear(
                gear_id, count_step, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller
            )
        old_grasping_height = self.grasping_height
        self.grasping_height = old_grasping_height - 0.050
        try:
            return super().pick_up_target_gear(
                gear_id, count_step, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller
            )
        finally:
            self.grasping_height = old_grasping_height

    def _refresh_pickup_state(self, spec: Manipulation) -> None:
        """Use the object's measured pose when a delayed pickup begins.

        Parts are dynamic rigid bodies and can settle/roll on the table while
        earlier assembly phases run.  The upstream policy caches the reset
        pose, which is valid for a single manipulation but stale in a long
        six-part episode.  Refreshing only at the pickup boundary leaves the
        physics and grasp semantics unchanged while removing that stale-target
        failure mode.
        """
        name = self._object_name(spec.gear_id)
        self.initial_root_state[name] = self.obj_dict[name].data.root_state_w.clone()

    def _refresh_mount_target(self, spec: Manipulation) -> None:
        """Track the currently measured assembly reference during insertion."""
        carrier_state = self.planetary_carrier.data.root_state_w.clone()
        if spec.gear_id <= 3:
            pin_local = self.gear_to_pin_map[f"sun_planetary_gear_{spec.gear_id}"]["pin_local_pos"]
            _, pin_world = torch_utils.tf_combine(
                carrier_state[:, 3:7],
                carrier_state[:, :3],
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
                pin_local.unsqueeze(0),
            )
            self.current_target_position = pin_world.clone()
        elif spec.gear_id == 4:
            self.current_target_position = carrier_state[:, :3].clone()
        elif spec.gear_id == 5:
            self.current_target_orientation, self.current_target_position = torch_utils.tf_combine(
                carrier_state[:, 3:7],
                carrier_state[:, :3],
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
                torch.zeros((1, 3), device=self.device),
            )
            if self._active_mount_spec is not None and self.count >= int(self._active_mount_spec.mount[1].item()):
                ring_state = self.obj_dict["ring_gear"].data.root_state_w
                xy_error = carrier_state[:, :2] - ring_state[:, :2]
                xy_error = torch.clamp(
                    xy_error,
                    -self.RING_XY_FEEDBACK_LIMIT_M,
                    self.RING_XY_FEEDBACK_LIMIT_M,
                )
                self.current_target_position = self.current_target_position.clone()
                self.current_target_position[:, :2] += self.RING_XY_FEEDBACK_GAIN * xy_error
        else:
            center_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            self.current_target_position = center_state[:, :3].clone()
            self.current_target_orientation = center_state[:, 3:7].clone()

    def _set_event(self, value: str) -> None:
        self.event = value
        if value != self._last_event:
            self.event_history.append({"physics_step": int(self.count), "event": value})
            print(f"M1_EVENT physics_step={self.count} event={value}", flush=True)
            self._last_event = value

    def _event_for(self, spec: Manipulation, *, pickup: bool) -> str:
        boundaries = spec.pickup if pickup else spec.mount
        index = max(i for i, value in enumerate(boundaries[:-1]) if self.count >= int(value.item()))
        if pickup:
            stages = ("approach", "descend", "grasp", "lift")
        elif spec.reducer_drop:
            stages = ("transport", "align", "release", "settle", "retreat")
        elif spec.rotate:
            stages = ("transport", "align", "insert", "release", "retreat")
        else:
            stages = ("transport", "align_insert", "release", "retreat")
        return f"{spec.phase}.{stages[index]}"

    def _mount_reducer_drop(self, spec: Manipulation, arm_cfg, gripper_cfg):
        """Align above the center, open in clearance, let gravity seat, retreat."""
        boundaries = spec.mount
        center_state = self.sun_planetary_gear_4.data.root_state_w.clone()
        if self.count == int(boundaries[0].item()):
            self.current_target_position = center_state[:, :3].clone()
            self.current_target_orientation = center_state[:, 3:7].clone()

        target = self.current_target_position.clone()
        target[:, 2] = self.table_height + self.grasping_height + 0.043
        target += torch.tensor(
            [self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device
        )
        high = target + torch.tensor(
            [0.0, 0.0, self.lifting_height + self.REDUCER_HIGH_EXTRA_M],
            device=self.device,
        )
        # Keep the 40 mm clearance that preserves the five assembled
        # relations; the narrow-axis bias below is the seating intervention.
        release_height = target + torch.tensor([0.0, 0.0, 0.040], device=self.device)
        release_height[:, 1] += 0.004
        orientation, _ = torch_utils.tf_combine(
            self.current_target_orientation,
            torch.zeros_like(target),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device),
            torch.zeros_like(target),
        )

        # Keep the reducer at the high waypoint for the complete lateral
        # transfer.  The previous diagonal transfer (XY motion while lowering)
        # entered the ring/centre stack from the side in the full episode: the
        # reducer and centre gear then both acquired a large tilt at the
        # align->release boundary.  The upstream policy uses the same safe
        # ordering (high transfer, low approach, then open), so separate the
        # horizontal and vertical motions here while retaining the existing
        # phase durations.
        if self.count < int(boundaries[2].item()):
            return self.move_robot_to_position(arm_cfg, gripper_cfg, self.diff_ik_controller, high, orientation, None)
        if self.count < int(boundaries[3].item()):
            # The gripper remains closed because this command addresses only
            # the arm joints.  Lower onto the socket before relaxing the grip.
            return self.move_robot_to_position(
                arm_cfg, gripper_cfg, self.diff_ik_controller, release_height, orientation, None
            )
        if self.count < int(boundaries[4].item()):
            # Controlled slip: relax grip just enough for a low-energy axial
            # insertion. A 4 mm narrow-axis bias lets the shaft bear on the
            # bore wall instead of free-falling through its 0.2 mm clearance.
            return torch.tensor([[0.010]], device=self.device), gripper_cfg.joint_ids
        return self.move_robot_to_position(arm_cfg, gripper_cfg, self.diff_ik_controller, high, orientation, None)

    def get_action(self):
        robot = self.scene["robot"]
        if self.count < self._ticks(0.5):
            self._set_event("stabilize")
            return robot.data.joint_pos[:, self.left_arm_entity_cfg.joint_ids].clone(), self.left_arm_entity_cfg.joint_ids

        for spec in self.manipulations:
            arm_cfg, gripper_cfg = self._entities(spec.arm)
            if self.count < int(spec.pickup[-1].item()) and self.count >= int(spec.pickup[0].item()):
                if self.count == int(spec.pickup[0].item()):
                    self._refresh_pickup_state(spec)
                self._set_event(self._event_for(spec, pickup=True))
                pick_action, pick_joint_ids = self.pick_up_target_gear(
                    spec.gear_id, spec.pickup, arm_cfg, gripper_cfg, self.diff_ik_controller
                )
                return pick_action, pick_joint_ids
            if self.count < int(spec.mount[-1].item()) and self.count >= int(spec.mount[0].item()):
                self._set_event(self._event_for(spec, pickup=False))
                self._active_mount_spec = spec
                # The ring is already geometrically seated before release.  Do
                # not chase a carrier that moves a few millimetres under
                # contact: during retreat that feedback made the open gripper
                # push the whole stack laterally.  Keep the dynamic target for
                # the pin/centre/reducer phases, where it removes stale-target
                # error, but freeze the ring reference at mount entry.
                if (
                    spec.gear_id != 5
                    or self.count == int(spec.mount[0].item())
                    or (
                        self.count >= int(spec.mount[1].item())
                        and self.count < int(spec.mount[3].item())
                    )
                ):
                    self._refresh_mount_target(spec)
                if spec.reducer_drop:
                    return self._mount_reducer_drop(spec, arm_cfg, gripper_cfg)
                if spec.rotate:
                    action, joint_ids = self.mount_gear_to_target_and_rotate(
                        spec.gear_id, spec.mount, arm_cfg, gripper_cfg
                    )
                    if (
                        spec.gear_id == 5
                        and self.count >= int(spec.mount[2].item())
                        and self.count < int(spec.mount[3].item())
                        and len(joint_ids) == 6
                    ):
                        # Open before the ring is fully seated so the ring can
                        # settle under gravity rather than storing a lateral
                        # squeeze that prevents the later release.
                        action = torch.cat(
                            [
                                action,
                                torch.tensor(
                                    [[self.RING_INSERT_GRIP_M]], device=self.device
                                ),
                            ],
                            dim=1,
                        )
                        arm_joint_ids = torch.as_tensor(
                            joint_ids, device=self.device, dtype=torch.long
                        )
                        grip_joint_ids = torch.as_tensor(
                            gripper_cfg.joint_ids, device=self.device, dtype=torch.long
                        )
                        joint_ids = torch.cat([arm_joint_ids, grip_joint_ids])
                    return action, joint_ids
                action, joint_ids = self.mount_gear_to_target(
                    spec.gear_id, spec.mount, arm_cfg, gripper_cfg
                )
                if (
                    spec.gear_id <= 3
                    and self.count >= int(spec.mount[1].item())
                    and self.count < int(spec.mount[2].item())
                    and len(joint_ids) == 6
                ):
                    # Let the gear relax against the pin while the wrist is
                    # still descending.  The small opening retains the part
                    # but avoids a sudden lateral impulse when the full-open
                    # release begins.
                    action = torch.cat(
                        [action, torch.tensor([[0.010]], device=self.device)],
                        dim=1,
                    )
                    arm_joint_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
                    grip_joint_ids = torch.as_tensor(
                        gripper_cfg.joint_ids, device=self.device, dtype=torch.long
                    )
                    joint_ids = torch.cat([arm_joint_ids, grip_joint_ids])
                # With two planets already seated, carrier contact yaws the
                # base while the third gear relaxes in the opposite direction.
                # The measured post-release mismatch was 0.129 rad (limit
                # 0.100). A small counter-rotation of the wrist keeps the
                # released gear carrier-relative without changing
                # its target position or any simulator state.
                if (
                    spec.phase == "planet3"
                    and self.count >= int(spec.mount[1].item())
                    and self.count < int(spec.mount[3].item())
                    and len(joint_ids) == 6
                ):
                    action = action.clone()
                    action[:, 5] += torch.deg2rad(
                        torch.tensor(-4.0, device=self.device)
                    )
                return action, joint_ids
            settle_end = int(spec.mount[-1].item()) + self._ticks(self.SETTLE_S)
            if self.count < settle_end and self.count >= int(spec.mount[-1].item()):
                self._set_event(f"{spec.phase}.settle")
                # The pinned official policy resets the arm after planet2 and
                # after the center gear before starting the next pickup.  The
                # old compact schedule kept the arm at its retreat pose,
                # which leaves the left arm in a different IK basin for the
                # ring when the carrier has yawed.  Reproduce those two
                # non-object state transitions during the existing settle
                # interval; the open gripper remains at its previous target.
                if spec.phase == "planet2":
                    return self.initial_pos_left.unsqueeze(0), self.left_arm_entity_cfg.joint_ids
                if spec.phase == "center":
                    return self.initial_pos_right.unsqueeze(0), self.right_arm_entity_cfg.joint_ids
                return torch.tensor([[0.04]], device=self.device), gripper_cfg.joint_ids

        self._active_mount_spec = None
        self._set_event("complete")
        return robot.data.joint_pos[:, self.left_arm_entity_cfg.joint_ids].clone(), self.left_arm_entity_cfg.joint_ids


# Kept at module scope because the upstream policy imports this namespace.
import isaacsim.core.utils.torch as torch_utils  # noqa: E402
