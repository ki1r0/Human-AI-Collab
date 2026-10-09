import math
import unittest
from types import SimpleNamespace

from runtime.bolt_harness.executor import (
    BoltMotionPlan,
    BoltSkillExecutor,
    TCPWaypoint,
    make_nominal_bolt_motion_plan,
)


class FakeFactoryEnv:
    def __init__(
        self,
        *,
        contacts=(True, True),
        stuck=False,
        end_on_step=None,
        armed=True,
        force_after_steps=None,
        wrench_force_n=0.0,
        clear_contacts_when_open=False,
        loaded_vertical_lag_m=0.0,
        lose_contact_during_loaded_motion=False,
    ):
        self.cfg = SimpleNamespace(
            sim=SimpleNamespace(dt=0.02),
            decimation=2,
            ctrl=SimpleNamespace(pos_action_threshold=(0.01, 0.01, 0.01), rot_action_threshold=(0.1, 0.1, 0.1)),
            tcp_linear_speed_stop_mps=0.08,
        )
        self.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.commanded_pose = self.pose.copy()
        self.velocity = [0.0] * 6
        self.joints = [0.0] * 9
        self.joint_velocities = [0.0] * 9
        self.width = 0.08
        self.width_target = 0.08
        self.contacts = contacts
        self.stuck = stuck
        self.end_on_step = end_on_step
        self.motion_safety_armed = armed
        self.unarmed_zero_steps = 0
        self.actions = []
        self.gripper_targets = []
        self.gripper_force_limits = []
        self.command_history = []
        self.pose_history = []
        self.wrench = [float(wrench_force_n), 0.0, 0.0, 0.0, 0.0, 0.0]
        self.wrench_baseline_stats = {"stationary_zero_reference_only": True}
        self.force_after_steps = force_after_steps
        self.clear_contacts_when_open = clear_contacts_when_open
        self.loaded_vertical_lag_m = loaded_vertical_lag_m
        self.lose_contact_during_loaded_motion = lose_contact_during_loaded_motion

    @property
    def bolt_root_pose(self):
        raise AssertionError("executor must not read simulator bolt pose")

    def get_public_executor_state(self):
        return {
            "tcp_pose": [self.pose],
            "commanded_tcp_pose": [self.commanded_pose],
            "tcp_velocity": [self.velocity],
            "joint_positions": [self.joints],
            "joint_velocities": [self.joint_velocities],
            "motion_safety_armed": self.motion_safety_armed,
            "gripper_width_m": self.width,
            "finger_bolt_contacts": self.contacts,
        }

    def set_gripper_target_width(self, width_m, max_force_n):
        if not math.isfinite(max_force_n) or max_force_n <= 0 or max_force_n > 20.0:
            raise ValueError("fake Panda force limit must be finite and within its actuator limit")
        self.gripper_targets.append(width_m)
        self.gripper_force_limits.append(max_force_n)
        self.width_target = width_m

    def public_wrench(self):
        return [self.wrench]

    def step(self, action):
        action = action[0]
        self.actions.append(tuple(action))
        if self.clear_contacts_when_open and self.width_target >= 0.079:
            self.contacts = (False, False)
        for axis in range(3):
            self.commanded_pose[axis] += float(action[axis]) * self.cfg.ctrl.pos_action_threshold[axis]
        rotation_delta = tuple(
            float(action[axis + 3]) * self.cfg.ctrl.rot_action_threshold[axis] for axis in range(3)
        )
        rotation_angle = math.sqrt(sum(value * value for value in rotation_delta))
        if rotation_angle > 1e-12:
            half_angle = rotation_angle / 2.0
            scale = math.sin(half_angle) / rotation_angle
            delta_q = (math.cos(half_angle), *(value * scale for value in rotation_delta))
            w, x, y, z = self.commanded_pose[3:7]
            dw, dx, dy, dz = delta_q
            updated_q = (
                w * dw - x * dx - y * dy - z * dz,
                w * dx + x * dw + y * dz - z * dy,
                w * dy - x * dz + y * dw + z * dx,
                w * dz + x * dy - y * dx + z * dw,
            )
            norm = math.sqrt(sum(value * value for value in updated_q))
            self.commanded_pose[3:7] = [value / norm for value in updated_q]
        self.command_history.append(tuple(self.commanded_pose))
        if not self.motion_safety_armed and not any(float(value) for value in action):
            self.unarmed_zero_steps += 1
            self.motion_safety_armed = True
        if not self.stuck:
            self.pose[:] = self.commanded_pose
            if self.width_target < 0.079:
                self.pose[2] -= self.loaded_vertical_lag_m
        if self.lose_contact_during_loaded_motion and self.width_target < 0.079 and any(action):
            self.contacts = (False, False)
        self.width = max(self.width_target, 0.022) if self.contacts == (True, True) and self.width_target < 0.022 else self.width_target
        self.joints[7:9] = [self.width / 2.0, self.width / 2.0]
        self.pose_history.append(tuple(self.pose))
        ended = self.end_on_step is not None and len(self.actions) >= self.end_on_step
        if self.force_after_steps is not None and len(self.actions) >= self.force_after_steps:
            self.wrench[0] = 6.0
        return {}, 0.0, False, ended, {}


class OneWayInsertionStopEnv(FakeFactoryEnv):
    def __init__(self, *, preinsert_z, stop_depth_m):
        super().__init__(wrench_force_n=0.608)
        self.stop_z = preinsert_z - stop_depth_m
        self.insertion_blocked = False
        self.relief_start_index = None

    def step(self, action):
        previous_command_z = self.commanded_pose[2]
        was_blocked = self.insertion_blocked
        result = super().step(action)
        if self.commanded_pose[2] <= self.stop_z:
            self.insertion_blocked = True
        if self.insertion_blocked:
            if was_blocked and self.commanded_pose[2] > previous_command_z and self.relief_start_index is None:
                self.relief_start_index = len(self.pose_history) - 1
            self.pose[2] = max(self.stop_z, self.commanded_pose[2])
            self.pose_history[-1] = tuple(self.pose)
        return result


def _plan():
    identity = (1.0, 0.0, 0.0, 0.0)
    return BoltMotionPlan(
        pick_waypoints=(
            TCPWaypoint((0.01, 0.0, 0.0), identity),
            TCPWaypoint((0.02, 0.0, 0.0), identity),
            TCPWaypoint((0.02, 0.0, 0.01), identity),
        ),
        transport_waypoints=(TCPWaypoint((0.08, 0.0, 0.04), identity),),
        open_gripper_width_m=0.08,
        grasp_gripper_width_m=0.018,
        max_gripper_force_n=12.0,
    )


def _stationary_plan(quaternion=(1.0, 0.0, 0.0, 0.0)):
    waypoint = TCPWaypoint((0.0, 0.0, 0.0), quaternion)
    return BoltMotionPlan(
        pick_waypoints=(waypoint, waypoint, waypoint),
        transport_waypoints=(waypoint,),
        open_gripper_width_m=0.08,
        grasp_gripper_width_m=0.018,
        max_gripper_force_n=12.0,
    )


def _insertion_plan(*, force_n=5.0, travel_m=0.04, speed_mps=0.002, seat_z=0.0):
    identity = (1.0, 0.0, 0.0, 0.0)
    preinsert = TCPWaypoint((0.0, 0.0, 0.04), identity)
    return BoltMotionPlan(
        pick_waypoints=(preinsert, TCPWaypoint((0.0, 0.0, 0.02), identity), preinsert),
        transport_waypoints=(preinsert,),
        open_gripper_width_m=0.08,
        grasp_gripper_width_m=0.018,
        max_gripper_force_n=12.0,
        seat_waypoint=TCPWaypoint((0.0, 0.0, seat_z), identity),
        retract_waypoint=preinsert,
        max_insertion_force_n=force_n,
        max_insertion_travel_m=travel_m,
        insertion_speed_mps=speed_mps,
    )


def _held_at_preinsert(env):
    env.pose[:] = [0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0]
    env.commanded_pose[:] = env.pose
    env.width = 0.018
    env.width_target = 0.018
    env.contacts = (True, True)


def _mark_held(env, executor):
    env.contacts = (True, True)
    env.width = 0.022
    env.width_target = executor.plan.grasp_gripper_width_m
    executor._holding = True
    executor._last_gripper_target_m = executor.plan.grasp_gripper_width_m


class BoltHarnessExecutorTests(unittest.TestCase):
    def test_nudge_uses_bounded_planar_steps_in_calibrated_frames(self):
        cases = (
            ("world", "x_positive", "coarse", (0.001, 0.0, 0.0)),
            ("assembly", "y_negative", "fine", (0.0, -0.0001, 0.0)),
            ("world", "y_positive", "contact", (0.0, 0.00005, 0.0)),
        )
        for frame, direction, step_class, expected_delta in cases:
            with self.subTest(frame=frame, direction=direction, step_class=step_class):
                env = FakeFactoryEnv(loaded_vertical_lag_m=0.0044)
                env.pose[:] = [0.1, 0.2, 0.2956, 1.0, 0.0, 0.0, 0.0]
                env.commanded_pose[:] = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
                plan = _insertion_plan()
                executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
                _mark_held(env, executor)
                executor._insertion_motion_completed = True

                result = executor.nudge(
                    frame=frame,
                    direction=direction,
                    step_class=step_class,
                    max_duration_s=1.0,
                )

                self.assertEqual(result["status"], "completed")
                self.assertTrue(executor._holding)
                self.assertFalse(executor._insertion_motion_completed)
                measured_start = (0.1, 0.2, 0.2956)
                command_start = (0.1, 0.2, 0.3)
                for axis in range(3):
                    self.assertAlmostEqual(
                        env.pose[axis] - measured_start[axis], expected_delta[axis], delta=0.00002
                    )
                    self.assertAlmostEqual(
                        env.commanded_pose[axis] - command_start[axis], expected_delta[axis], delta=0.00002
                    )
                self.assertAlmostEqual(env.commanded_pose[2] - env.pose[2], 0.0044, places=7)
                self.assertTrue(all(target == plan.grasp_gripper_width_m for target in env.gripper_targets))
                self.assertTrue(all(force == plan.max_gripper_force_n for force in env.gripper_force_limits))
                dt = env.cfg.sim.dt * env.cfg.decimation
                command_path = [list(command_start)] + [list(command[:3]) for command in env.command_history]
                self.assertTrue(
                    all(math.dist(before, after) <= 0.005 * dt + 1e-8
                        for before, after in zip(command_path, command_path[1:]))
                )

    def test_nudge_rejects_unsupported_axes_frames_and_duration_without_stepping(self):
        env = FakeFactoryEnv()
        plan = _insertion_plan()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        _mark_held(env, executor)
        executor._insertion_motion_completed = True
        invalid_calls = (
            {"frame": "world", "direction": "z_positive", "step_class": "fine", "max_duration_s": 1.0},
            {"frame": "unmapped", "direction": "x_positive", "step_class": "fine", "max_duration_s": 1.0},
            {"frame": "assembly", "direction": "x_positive", "step_class": "large", "max_duration_s": 1.0},
            {"frame": "world", "direction": "x_positive", "step_class": "coarse", "max_duration_s": 1.01},
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments):
                self.assertEqual(executor.nudge(**arguments)["status"], "invalid_action")
        self.assertEqual(env.actions, [])
        self.assertTrue(executor._insertion_motion_completed)

    def test_nudge_force_limit_rolls_back_and_keeps_grasp_closed(self):
        env = FakeFactoryEnv(force_after_steps=1)
        plan = _insertion_plan()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        _mark_held(env, executor)
        executor._insertion_motion_completed = True

        result = executor.nudge(
            frame="world", direction="x_positive", step_class="coarse", max_duration_s=1.0
        )

        self.assertEqual(result["status"], "force_limit")
        self.assertTrue(executor._holding)
        self.assertFalse(executor._insertion_motion_completed)
        self.assertAlmostEqual(env.pose[0], 0.0, delta=1e-8)
        self.assertAlmostEqual(env.commanded_pose[0], 0.0, delta=1e-8)
        self.assertEqual(env.gripper_targets[-1], plan.grasp_gripper_width_m)
        self.assertEqual(env.actions[-1], (0.0,) * 6)

    def test_nudge_requires_measured_settling_and_bilateral_contact(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        _mark_held(env, executor)
        env.velocity[0] = 0.002

        result = executor.nudge(
            frame="world", direction="x_positive", step_class="fine", max_duration_s=0.4
        )

        self.assertEqual(result["status"], "motion_timeout")
        self.assertTrue(executor._holding)

        env = FakeFactoryEnv(contacts=(True, False))
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        _mark_held(env, executor)
        env.contacts = (True, False)

        result = executor.nudge(
            frame="world", direction="x_positive", step_class="fine", max_duration_s=0.4
        )

        self.assertEqual(result["status"], "motion_timeout")
        self.assertFalse(executor._holding)
        self.assertFalse(any(any(action) for action in env.actions))

    def test_retract_is_upward_only_and_preserves_grasp_for_short_and_full(self):
        for distance_class, start_z, expected_distance in (
            ("short", 0.02, 0.01),
            ("full", 0.0, 0.04),
        ):
            with self.subTest(distance_class=distance_class):
                env = FakeFactoryEnv()
                env.pose[2] = start_z
                env.commanded_pose[2] = start_z
                plan = _insertion_plan()
                executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
                _mark_held(env, executor)
                executor._insertion_motion_completed = True

                result = executor.retract(
                    direction="z_positive",
                    distance_class=distance_class,
                    max_duration_s=5.0,
                )

                self.assertEqual(result["status"], "completed")
                self.assertAlmostEqual(env.pose[2], start_z + expected_distance, delta=0.00002)
                self.assertAlmostEqual(env.pose[0], 0.0)
                self.assertAlmostEqual(env.pose[1], 0.0)
                self.assertTrue(executor._holding)
                self.assertFalse(executor._insertion_motion_completed)
                self.assertTrue(all(target == plan.grasp_gripper_width_m for target in env.gripper_targets))

    def test_retract_rejects_unsafe_directions_and_unconfigured_full_distance(self):
        env = FakeFactoryEnv()
        plan = _insertion_plan()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        _mark_held(env, executor)
        for direction in ("x_positive", "x_negative", "y_positive", "y_negative", "z_negative"):
            with self.subTest(direction=direction):
                self.assertEqual(
                    executor.retract(direction=direction, distance_class="short", max_duration_s=5.0)["status"],
                    "invalid_action",
                )
        self.assertEqual(
            executor.retract(direction="z_positive", distance_class="short", max_duration_s=5.01)["status"],
            "invalid_action",
        )
        self.assertEqual(env.actions, [])

        env = FakeFactoryEnv()
        plan = _plan()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        _mark_held(env, executor)
        self.assertEqual(
            executor.retract(direction="z_positive", distance_class="full", max_duration_s=5.0)["status"],
            "invalid_action",
        )
        self.assertEqual(env.actions, [])

    def test_held_microactions_require_stationary_wrench_baseline(self):
        for action_name in ("nudge", "retract"):
            with self.subTest(action=action_name):
                env = FakeFactoryEnv()
                del env.wrench_baseline_stats
                plan = _insertion_plan()
                executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
                _mark_held(env, executor)
                start_pose = tuple(env.pose)
                if action_name == "nudge":
                    action = lambda: executor.nudge(
                        frame="world",
                        direction="x_positive",
                        step_class="fine",
                        max_duration_s=1.0,
                    )
                else:
                    action = lambda: executor.retract(
                        direction="z_positive",
                        distance_class="short",
                        max_duration_s=5.0,
                    )

                with self.assertRaisesRegex(RuntimeError, "set_stationary_wrench_baseline"):
                    action()

                self.assertEqual(tuple(env.pose), start_pose)
                self.assertEqual(env.actions, [(0.0,) * 6])
                self.assertEqual(env.gripper_targets, [plan.grasp_gripper_width_m])
                self.assertTrue(executor._holding)

    def test_gripper_closes_only_after_three_consecutive_fine_pose_samples(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_stationary_plan())

        result = executor.execute_skill("pick", "socket_1", max_duration_s=1)

        self.assertEqual(result["status"], "completed")
        first_close = env.gripper_targets.index(0.018)
        self.assertEqual(first_close, 2)
        self.assertEqual(env.gripper_targets[:first_close], [0.08, 0.08])
        self.assertEqual(env.actions[:first_close], [(0.0,) * 6] * 2)

    def test_grasp_does_not_close_without_fine_pose_and_settling_criteria(self):
        cases = (
            ("position_error", 0.0, 0.0, (1.0, 0.0, 0.0, 0.0), 0.0007),
            ("linear_speed", 0.0, 0.002, (1.0, 0.0, 0.0, 0.0), 0.0),
            ("joint_speed", 0.02, 0.0, (1.0, 0.0, 0.0, 0.0), 0.0),
            ("orientation_error", 0.0, 0.0, (math.cos(0.003), math.sin(0.003), 0.0, 0.0), 0.0),
        )
        for name, joint_speed, tcp_speed, quaternion, position_error in cases:
            with self.subTest(name=name):
                env = FakeFactoryEnv(stuck=True)
                env.joint_velocities[0] = joint_speed
                env.velocity[0] = tcp_speed
                env.pose[0] = position_error
                executor = BoltSkillExecutor(env, target_id="socket_1", plan=_stationary_plan(quaternion))

                result = executor.execute_skill("pick", "socket_1", max_duration_s=0.4)

                self.assertEqual(result["status"], "motion_timeout")
                self.assertNotIn(0.018, env.gripper_targets)

    def test_final_transport_alignment_keeps_xy_fine_and_z_looser(self):
        waypoint = TCPWaypoint((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_stationary_plan())
        env.pose[0] = 0.0004
        env.pose[2] = 0.0015
        _, reached_inside_xy, _ = executor._move_to(
            waypoint,
            executor._read_state(),
            1,
            0.018,
            require_grasp=False,
            require_open=False,
            xy_position_tolerance_m=0.0005,
            position_tolerance_m=0.002,
        )
        self.assertTrue(reached_inside_xy)

        env = FakeFactoryEnv(stuck=True)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_stationary_plan())
        env.pose[0] = 0.0006
        _, reached_outside_xy, _ = executor._move_to(
            waypoint,
            executor._read_state(),
            1,
            0.018,
            require_grasp=False,
            require_open=False,
            xy_position_tolerance_m=0.0005,
            position_tolerance_m=0.002,
        )
        self.assertFalse(reached_outside_xy)

    def test_executor_waits_with_zero_delta_until_motion_safety_arms(self):
        env = FakeFactoryEnv(armed=False)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        result = executor.execute_skill("pick", "socket_1", max_duration_s=4)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(env.actions[0], (0.0,) * 6)
        self.assertEqual(env.unarmed_zero_steps, 1)
        self.assertTrue(any(any(value != 0.0 for value in action) for action in env.actions[1:]))

    def test_fake_stepper_actions_and_gripper_commands_stay_bounded(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        self.assertEqual(executor.execute_skill("pick", "socket_1", max_duration_s=4)["status"], "completed")
        self.assertEqual(executor.execute_skill("transport", "socket_1", max_duration_s=4)["status"], "completed")
        self.assertTrue(all(math.isfinite(v) and abs(v) <= 1.0 for action in env.actions for v in action))
        self.assertTrue(all(0.0 <= target <= 0.08 for target in env.gripper_targets))
        self.assertAlmostEqual(env.gripper_targets[-1], 0.018)
        self.assertTrue(all(force == 12.0 for force in env.gripper_force_limits))
        self.assertEqual(len(env.gripper_targets), len(env.actions))
        max_control_step = 0.04 * env.cfg.sim.dt * env.cfg.decimation
        for before, after in zip(env.command_history, env.command_history[1:]):
            displacement = math.sqrt(sum((after[i] - before[i]) ** 2 for i in range(3)))
            self.assertLessEqual(displacement, max_control_step + 1e-8)
        self.assertLessEqual(abs(env.pose[0] - 0.08), 0.002)
        self.assertLessEqual(abs(env.pose[2] - 0.04), 0.002)

    def test_free_space_lookahead_is_bounded_at_10mm(self):
        env = FakeFactoryEnv(stuck=True)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())
        waypoint = TCPWaypoint((0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        maximum_lead = 0.0

        for _ in range(12):
            action, _, _ = executor._action(
                waypoint,
                executor._read_state(),
                max_linear_speed_mps=0.04,
                position_tracking_tolerance_m=0.010 - (0.04 * 0.04),
            )
            action_distance = math.sqrt(
                sum((action[i] * env.cfg.ctrl.pos_action_threshold[i]) ** 2 for i in range(3))
            )
            self.assertLessEqual(action_distance, 0.04 * 0.04 + 1e-9)
            state, ended = executor._step(action, 0.08)
            self.assertFalse(ended)
            lead = math.dist(state["commanded_tcp_pose"][:3], state["tcp_pose"][:3])
            maximum_lead = max(maximum_lead, lead)

        self.assertAlmostEqual(maximum_lead, 0.010, places=7)
        self.assertLessEqual(maximum_lead, 0.020)

    def test_fine_grasp_approach_speed_remains_20mm_per_second(self):
        env = FakeFactoryEnv(stuck=True)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())
        waypoint = TCPWaypoint((0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0))

        action, _, _ = executor._action(
            waypoint,
            executor._read_state(),
            max_linear_speed_mps=0.02,
            position_tracking_tolerance_m=0.002,
        )

        action_distance = math.sqrt(
            sum((action[i] * env.cfg.ctrl.pos_action_threshold[i]) ** 2 for i in range(3))
        )
        self.assertAlmostEqual(action_distance, 0.02 * 0.04)

    def test_nominal_pick_and_transport_fit_30s_with_loaded_axial_lag(self):
        env = FakeFactoryEnv(loaded_vertical_lag_m=0.0044)
        env.pose[:] = [0.50, 0.0, 1.1297, 1.0, 0.0, 0.0, 0.0]
        env.commanded_pose[:] = env.pose
        plan = make_nominal_bolt_motion_plan(max_gripper_force_n=9.0, max_insertion_force_n=5.0)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)

        pick_result = executor.execute_skill("pick", "socket_1", max_duration_s=30)

        self.assertEqual(pick_result["status"], "completed")
        self.assertLess(len(env.actions), int(30 / (env.cfg.sim.dt * env.cfg.decimation)))
        first_close = env.gripper_targets.index(plan.grasp_gripper_width_m)
        grasp_sample = env.pose_history[first_close - 1]
        self.assertLessEqual(math.dist(grasp_sample[:3], plan.pick_waypoints[1].position_m), 0.0005)
        self.assertAlmostEqual(env.pose[0], plan.pick_waypoints[2].position_m[0], places=7)
        self.assertAlmostEqual(env.pose[1], plan.pick_waypoints[2].position_m[1], places=7)
        self.assertLessEqual(abs(env.pose[2] - plan.pick_waypoints[2].position_m[2]), 0.006)
        self.assertAlmostEqual(env.commanded_pose[2] - env.pose[2], 0.0044, places=7)

        transport_start = len(env.pose_history)
        transport_result = executor.execute_skill("transport", "socket_1", max_duration_s=30)

        self.assertEqual(transport_result["status"], "completed")
        destination = plan.transport_waypoints[-1]
        self.assertLessEqual(math.dist(env.pose[:2], destination.position_m[:2]), 0.0002)
        self.assertLessEqual(abs(env.pose[2] - destination.position_m[2]), 0.006)
        self.assertAlmostEqual(env.commanded_pose[2] - env.pose[2], 0.0044, places=7)
        lateral_samples = [
            sample
            for sample in env.pose_history[transport_start:]
            if math.dist(sample[:2], plan.pick_waypoints[2].position_m[:2]) > 1e-6
            and math.dist(sample[:2], destination.position_m[:2]) > 0.0005
        ]
        self.assertTrue(lateral_samples)
        self.assertTrue(all(sample[2] >= 0.885855 for sample in lateral_samples))

    def test_transport_does_not_accept_point_three_millimeter_preinsert_xy_error(self):
        env = FakeFactoryEnv(stuck=True, loaded_vertical_lag_m=0.0044)
        plan = _plan()
        target = plan.transport_waypoints[-1]
        env.pose[:] = [target.position_m[0] + 0.0003, target.position_m[1], target.position_m[2] - 0.0044,
                       *target.quaternion_wxyz]
        env.commanded_pose[:] = [*target.position_m, *target.quaternion_wxyz]
        env.width = env.width_target = plan.grasp_gripper_width_m
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("transport", "socket_1", max_duration_s=0.08)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertTrue(executor._holding)
        self.assertEqual(env.pose[0], target.position_m[0] + 0.0003)

    def test_pick_never_lifts_without_bilateral_contact(self):
        env = FakeFactoryEnv(contacts=(True, False))
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        result = executor.execute_skill("pick", "socket_1", max_duration_s=1)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertAlmostEqual(env.pose[2], 0.0)
        self.assertEqual(executor.execute_skill("transport", "socket_1")["status"], "invalid_action")

    def test_pick_reports_grasp_loss_separately_from_motion_timeout(self):
        env = FakeFactoryEnv(lose_contact_during_loaded_motion=True)
        plan = _plan()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)

        result = executor.execute_skill("pick", "socket_1", max_duration_s=4)

        self.assertEqual(result["status"], "grasp_lost")
        self.assertFalse(executor._holding)
        self.assertEqual(env.gripper_targets[-1], plan.grasp_gripper_width_m)
        self.assertEqual(env.actions[-1], (0.0,) * 6)

    def test_stalled_tcp_times_out_and_issues_a_zero_delta_hold(self):
        env = FakeFactoryEnv(stuck=True)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        result = executor.execute_skill("pick", "socket_1", max_duration_s=0.4)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertGreater(len(env.actions), 1)
        self.assertEqual(env.actions[-1], (0.0,) * 6)
        self.assertEqual(env.command_history[-1], env.command_history[-2])
        max_control_step = 0.04 * env.cfg.sim.dt * env.cfg.decimation
        max_tracking_lead = 0.010
        self.assertLessEqual(env.commanded_pose[0], max_tracking_lead + 1e-8)
        self.assertTrue(all(abs(pose[0]) <= max_tracking_lead + 1e-8 for pose in env.command_history))
        self.assertTrue(all(abs(v) <= 1.0 for action in env.actions for v in action))

    def test_environment_timeout_does_not_step_a_reset_episode(self):
        env = FakeFactoryEnv(end_on_step=1)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        result = executor.execute_skill("pick", "socket_1", max_duration_s=1)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertEqual(len(env.actions), 1)

    def test_tool_arguments_are_rejected_before_stepping(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

        self.assertEqual(executor.execute_skill("release_and_retract", "socket_1")["status"], "invalid_action")
        self.assertEqual(
            executor.execute_skill("insert_and_seat", "socket_1", mode="compliant")["status"],
            "invalid_action",
        )
        self.assertEqual(executor.execute_skill("pick", "other_socket")["status"], "invalid_action")
        self.assertEqual(executor.execute_skill("pick", "socket_1", max_duration_s=31)["status"], "invalid_action")
        self.assertEqual(executor.execute_skill([], "socket_1")["status"], "invalid_action")
        self.assertEqual(env.actions, [])

    def test_nominal_plan_uses_measured_upright_top_down_targets(self):
        plan = make_nominal_bolt_motion_plan(max_gripper_force_n=9.0, max_insertion_force_n=5.0)

        self.assertEqual(plan.pick_waypoints[0].position_m, (0.30, 0.25, 0.7738997358))
        self.assertEqual(plan.pick_waypoints[1].position_m, (0.30, 0.25, 0.7338997358))
        self.assertEqual(plan.pick_waypoints[2].position_m, (0.30, 0.25, 0.8138997358))
        self.assertEqual(plan.pick_waypoints[1].quaternion_wxyz, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(
            tuple(waypoint.position_m for waypoint in plan.transport_waypoints),
            (
                (0.30, 0.25, 0.90),
                (0.58, -0.15, 0.90),
                (0.58, -0.15, 0.88585544),
            ),
        )
        self.assertGreaterEqual(plan.transport_waypoints[0].position_m[2] - 0.006, 0.885855)
        self.assertEqual(plan.seat_waypoint.position_m, (0.58, -0.15, 0.82185544))
        self.assertAlmostEqual(plan.transport_waypoints[-1].position_m[2] - plan.seat_waypoint.position_m[2], 0.064)
        self.assertEqual(plan.max_insertion_travel_m, 0.08)
        self.assertEqual(plan.insertion_speed_mps, 0.003)
        self.assertEqual(plan.retract_waypoint, plan.transport_waypoints[-1])
        self.assertEqual(plan.grasp_gripper_width_m, 0.012)
        self.assertEqual(plan.max_gripper_force_n, 9.0)
        self.assertEqual(plan.max_insertion_force_n, 5.0)
        self.assertGreaterEqual(plan.held_seat_dwell_s, 0.25)

        loading_plan = make_nominal_bolt_motion_plan(max_gripper_force_n=9.0)
        self.assertIsNone(loading_plan.seat_waypoint)

    def test_nominal_insertion_accepts_axial_transport_lag_but_keeps_xy_fine(self):
        plan = make_nominal_bolt_motion_plan(max_gripper_force_n=9.0, max_insertion_force_n=5.0)
        preinsert = plan.transport_waypoints[-1]
        env = FakeFactoryEnv()
        env.pose[:] = [preinsert.position_m[0], preinsert.position_m[1], preinsert.position_m[2] - 0.005,
                       *preinsert.quaternion_wxyz]
        env.commanded_pose[:] = env.pose
        env.width = env.width_target = plan.grasp_gripper_width_m
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(env.pose[2], plan.seat_waypoint.position_m[2], delta=0.0005)

        env = FakeFactoryEnv()
        env.pose[:] = [preinsert.position_m[0] + 0.0006, preinsert.position_m[1], preinsert.position_m[2],
                       *preinsert.quaternion_wxyz]
        env.commanded_pose[:] = [*preinsert.position_m, *preinsert.quaternion_wxyz]
        env.width = env.width_target = plan.grasp_gripper_width_m
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result["status"], "invalid_action")
        self.assertEqual(env.actions, [])

    def test_transport_final_orientation_uses_insertion_tolerance(self):
        env = FakeFactoryEnv(stuck=True)
        plan = _plan()
        target = plan.transport_waypoints[-1]
        env.pose[:] = [*target.position_m, math.cos(0.003), 0.0, 0.0, math.sin(0.003)]
        env.commanded_pose[:] = [*target.position_m, *target.quaternion_wxyz]
        env.width = env.width_target = plan.grasp_gripper_width_m
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("transport", "socket_1", max_duration_s=0.08)

        self.assertEqual(result["status"], "motion_timeout")

    def test_force_limit_must_be_positive_and_finite(self):
        for force in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(force=force), self.assertRaises(ValueError):
                BoltMotionPlan(
                    pick_waypoints=_plan().pick_waypoints,
                    transport_waypoints=_plan().transport_waypoints,
                    open_gripper_width_m=0.08,
                    grasp_gripper_width_m=0.018,
                    max_gripper_force_n=force,
                )

    def test_executor_requires_public_measured_state(self):
        env = FakeFactoryEnv()
        env.get_public_executor_state = None
        with self.assertRaisesRegex(RuntimeError, "get_public_executor_state"):
            BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

    def test_public_state_requires_persistent_commanded_tcp_pose(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())
        public_state = env.get_public_executor_state()
        del public_state["commanded_tcp_pose"]
        env.get_public_executor_state = lambda: public_state

        with self.assertRaisesRegex(RuntimeError, "commanded_tcp_pose"):
            executor._read_state()

    def test_executor_requires_force_limited_gripper_setter(self):
        env = FakeFactoryEnv()
        env.set_gripper_target_width = None
        with self.assertRaisesRegex(RuntimeError, "set_gripper_target_width"):
            BoltSkillExecutor(env, target_id="socket_1", plan=_plan())

    def test_target_delta_uses_commanded_pose_and_arrival_uses_measured_tcp(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())
        env.pose[0] = 0.0072
        env.commanded_pose[0] = 0.0098
        waypoint = TCPWaypoint((0.01, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

        action, measured_position_error, measured_rotation_error = executor._action(waypoint, executor._read_state())

        self.assertAlmostEqual(action[0], 0.02)
        self.assertAlmostEqual(measured_position_error, 0.0028)
        self.assertAlmostEqual(measured_rotation_error, 0.0)

    def test_rotation_target_increment_is_rate_limited_per_env_step(self):
        env = FakeFactoryEnv()
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_plan())
        waypoint = TCPWaypoint((0.0, 0.0, 0.0), (math.cos(0.5), math.sin(0.5), 0.0, 0.0))

        action, _, _ = executor._action(waypoint, executor._read_state())

        rotation_step = math.sqrt(sum((action[i + 3] * env.cfg.ctrl.rot_action_threshold[i]) ** 2 for i in range(3)))
        max_rotation_step = 0.5 * env.cfg.sim.dt * env.cfg.decimation
        self.assertLessEqual(rotation_step, max_rotation_step + 1e-8)

    def test_insertion_is_slow_bounded_and_holds_at_seat_for_evaluator_dwell(self):
        env = FakeFactoryEnv()
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(executor._holding)
        self.assertAlmostEqual(env.pose[2], 0.0, delta=0.0005)
        dt = env.cfg.sim.dt * env.cfg.decimation
        step_limit = 0.002 * dt
        path = [[0.0, 0.0, 0.04]] + [list(command[:3]) for command in env.command_history]
        self.assertTrue(all(after[2] >= -1e-9 for after in path))
        self.assertTrue(
            all(math.dist(before, after) <= step_limit + 1e-8 for before, after in zip(path, path[1:]))
        )
        last_motion = max(i for i, action in enumerate(env.actions) if any(action))
        self.assertGreaterEqual((len(env.actions) - last_motion - 1) * dt, 0.2 - 1e-9)
        self.assertTrue(all(action == (0.0,) * 6 for action in env.actions[last_motion + 1 :]))

    def test_insertion_travel_origin_ignores_accepted_preinsert_tracking_lag(self):
        env = FakeFactoryEnv()
        _held_at_preinsert(env)
        env.pose[2] -= 0.0018
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(env.pose[2], 0.0, delta=0.0005)
        diagnostics = executor.private_diagnostics["insertion"]
        self.assertEqual(diagnostics["exit_reason"], "seat_pose_settled_and_dwell_verified")
        self.assertAlmostEqual(diagnostics["samples"][0]["tcp_pose"][2], 0.0382)

    def test_insertion_force_limit_stops_and_keeps_bolt_held(self):
        env = FakeFactoryEnv(force_after_steps=20)
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan(force_n=5.0))
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=10)

        self.assertEqual(result["status"], "force_limit")
        self.assertTrue(executor._holding)
        self.assertLess(env.pose[2], 0.04)
        self.assertGreater(env.pose[2], 0.0)
        self.assertEqual(env.actions[-1], (0.0,) * 6)

    def test_retract_preserves_force_cutoff_without_moving_over_limit(self):
        env = FakeFactoryEnv(wrench_force_n=3.1)
        plan = _insertion_plan(force_n=3.0)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        _mark_held(env, executor)
        start_pose = tuple(env.pose)

        result = executor.retract(
            direction="z_positive", distance_class="short", max_duration_s=5.0
        )

        self.assertEqual(result["status"], "force_limit")
        self.assertEqual(tuple(env.pose), start_pose)
        self.assertEqual(env.actions, [(0.0,) * 6])
        self.assertTrue(executor._holding)
        self.assertEqual(env.gripper_targets[-1], plan.grasp_gripper_width_m)

    def test_force_limit_during_seat_dwell_retracts_before_holding(self):
        env = FakeFactoryEnv(force_after_steps=502)
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan(force_n=5.0))
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result["status"], "force_limit")
        self.assertTrue(executor._holding)
        self.assertFalse(executor._insertion_motion_completed)
        self.assertGreater(env.pose[2], 0.0005)
        self.assertLessEqual(env.pose[2], 0.0011)
        self.assertEqual(env.actions[-1], (0.0,) * 6)

    def test_force_relief_never_moves_down_when_tcp_is_above_preinsert(self):
        env = FakeFactoryEnv(wrench_force_n=6.0)
        _held_at_preinsert(env)
        env.pose[2] = 0.041
        env.commanded_pose[2] = 0.041
        executor = BoltSkillExecutor(
            env,
            target_id="socket_1",
            plan=_insertion_plan(force_n=5.0, travel_m=0.05),
        )
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant")

        self.assertEqual(result["status"], "force_limit")
        self.assertAlmostEqual(env.pose[2], 0.041)
        self.assertTrue(all(abs(command[2] - 0.041) < 1e-12 for command in env.command_history))

    def test_stalled_insertion_returns_public_status_after_bounded_relief_and_hold(self):
        plan = _insertion_plan(force_n=3.0, travel_m=0.04)
        env = OneWayInsertionStopEnv(
            preinsert_z=plan.transport_waypoints[-1].position_m[2], stop_depth_m=0.007247
        )
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=10)

        self.assertEqual(result["status"], "stalled")
        self.assertTrue(executor._holding)
        self.assertFalse(executor._insertion_motion_completed)
        self.assertEqual(env.contacts, (True, True))
        self.assertAlmostEqual(env.wrench[0], 0.608)
        relief_m = env.pose[2] - env.stop_z
        self.assertGreater(relief_m, 0.0009)
        self.assertLessEqual(relief_m, 0.001 + 1e-9)
        self.assertIsNotNone(env.relief_start_index)
        relief_poses = env.pose_history[env.relief_start_index :]
        self.assertTrue(all(pose[2] <= env.stop_z + 0.001 + 1e-9 for pose in relief_poses))
        self.assertNotIn(0.08, env.gripper_targets)
        self.assertEqual(env.actions[-1], (0.0,) * 6)
        diagnostics = executor.private_diagnostics["insertion"]
        self.assertEqual(diagnostics["exit_reason"], "no_progress_before_seat")
        self.assertTrue(diagnostics["samples"])
        sample = diagnostics["samples"][-1]
        self.assertFalse(sample["at_seat"])
        self.assertTrue(sample["settled"])
        self.assertEqual(diagnostics["endpoint"]["tcp_pose"], tuple(env.pose))
        self.assertEqual(diagnostics["endpoint"]["commanded_tcp_pose"], tuple(env.commanded_pose))
        self.assertIn("force_norm_n", sample)
        self.assertIn("joint_velocities", sample)

    def test_retry_preserves_stalled_insertion_diagnostics_when_latest_attempt_is_invalid(self):
        plan = _insertion_plan(force_n=3.0, travel_m=0.04)
        env = OneWayInsertionStopEnv(
            preinsert_z=plan.transport_waypoints[-1].position_m[2], stop_depth_m=0.007247
        )
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        first = executor.execute_skill(
            "insert_and_seat", "socket_1", mode="compliant", max_duration_s=10
        )
        first_diagnostics = executor.private_diagnostics["insertion"]
        self.assertEqual(first["status"], "stalled")
        self.assertEqual(first_diagnostics["exit_reason"], "no_progress_before_seat")
        self.assertTrue(first_diagnostics["samples"])

        second = executor.execute_skill(
            "insert_and_seat", "socket_1", mode="compliant", max_duration_s=10
        )

        self.assertEqual(second["status"], "invalid_action")
        latest_diagnostics = executor.private_diagnostics["insertion"]
        self.assertEqual(latest_diagnostics["exit_reason"], "preinsert_pose_outside_tolerance")
        self.assertEqual(latest_diagnostics["samples"], [])
        self.assertEqual(len(latest_diagnostics["previous_attempts"]), 1)
        self.assertEqual(
            latest_diagnostics["previous_attempts"][0]["exit_reason"], "no_progress_before_seat"
        )
        self.assertTrue(latest_diagnostics["previous_attempts"][0]["samples"])

    def test_insertion_diagnostics_distinguish_seat_pose_from_unsettled_timeout(self):
        env = FakeFactoryEnv()
        _held_at_preinsert(env)
        plan = _insertion_plan(seat_z=0.0396)
        env.commanded_pose[2] = plan.seat_waypoint.position_m[2]
        env.velocity[2] = 0.002
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=plan)
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=0.2)

        self.assertEqual(result, {"status": "motion_timeout"})
        self.assertTrue(executor._holding)
        self.assertFalse(executor._insertion_motion_completed)
        diagnostics = executor.private_diagnostics["insertion"]
        self.assertEqual(diagnostics["exit_reason"], "skill_budget_exhausted")
        self.assertTrue(diagnostics["samples"])
        sample = diagnostics["samples"][-1]
        self.assertTrue(sample["actual_pose_at_seat"])
        self.assertTrue(sample["command_pose_at_seat"])
        self.assertTrue(sample["at_seat"])
        self.assertFalse(sample["tcp_linear_settled"])
        self.assertFalse(sample["settled"])
        self.assertEqual(diagnostics["endpoint"]["phase"], "endpoint")
        self.assertIn("tcp_position_error_m_xyz", diagnostics["endpoint"])
        self.assertIn("command_position_error_m_xyz", diagnostics["endpoint"])

    def test_insertion_dwell_uses_arm_stationarity_not_contacting_finger_speed(self):
        env = FakeFactoryEnv()
        _held_at_preinsert(env)
        env.joint_velocities[7:9] = [-0.0186, -0.0186]
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)

        self.assertEqual(result, {"status": "completed"})
        self.assertTrue(executor._insertion_motion_completed)
        self.assertEqual(env.contacts, (True, True))
        diagnostic = executor.private_diagnostics["insertion"]
        self.assertEqual(diagnostic["exit_reason"], "seat_pose_settled_and_dwell_verified")
        self.assertTrue(diagnostic["endpoint"]["arm_joints_settled"])
        self.assertEqual(diagnostic["endpoint"]["gripper_joint_velocities"], (-0.0186, -0.0186))
        self.assertFalse(diagnostic["endpoint"]["gripper_joints_settled"])
        self.assertTrue(diagnostic["endpoint"]["settled"])

    def test_insertion_wall_timeout_reserves_relief_and_hold_steps(self):
        env = FakeFactoryEnv(stuck=True)
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        result = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=0.2)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertLessEqual(len(env.actions), 5)
        self.assertEqual(env.actions[-1], (0.0,) * 6)
        self.assertTrue(executor._holding)

    def test_insertion_requires_explicit_stationary_wrench_baseline(self):
        env = FakeFactoryEnv()
        env.wrench_baseline_stats = None
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        with self.assertRaisesRegex(RuntimeError, "set_stationary_wrench_baseline"):
            executor.execute_skill("insert_and_seat", "socket_1", mode="compliant")

        self.assertFalse(any(any(value != 0.0 for value in action) for action in env.actions))

    def test_release_requires_check_task_true_and_completed_insert_motion(self):
        env = FakeFactoryEnv(clear_contacts_when_open=True)
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True

        self.assertEqual(
            executor.execute_skill("release_and_retract", "socket_1", completed=True)["status"],
            "invalid_action",
        )
        self.assertEqual(env.actions, [])

        insert = executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)
        self.assertEqual(insert["status"], "completed")
        action_count = len(env.actions)
        self.assertEqual(
            executor.execute_skill("release_and_retract", "socket_1", completed=False)["status"],
            "invalid_action",
        )
        self.assertEqual(len(env.actions), action_count)

        release = executor.execute_skill("release_and_retract", "socket_1", completed=True, max_duration_s=30)

        self.assertEqual(release["status"], "completed")
        self.assertFalse(executor._holding)
        self.assertEqual(env.gripper_targets[-1], 0.08)
        self.assertAlmostEqual(env.pose[2], 0.04, delta=0.002)
        self.assertNotIn("task_success", release)

    def test_release_does_not_open_if_tcp_left_the_verified_seat_pose(self):
        env = FakeFactoryEnv()
        _held_at_preinsert(env)
        executor = BoltSkillExecutor(env, target_id="socket_1", plan=_insertion_plan())
        executor._holding = True
        self.assertEqual(
            executor.execute_skill("insert_and_seat", "socket_1", mode="compliant", max_duration_s=30)["status"],
            "completed",
        )
        env.pose[2] += 0.001
        env.commanded_pose[2] = env.pose[2]
        result = executor.execute_skill("release_and_retract", "socket_1", completed=True)

        self.assertEqual(result["status"], "motion_timeout")
        self.assertTrue(executor._holding)
        self.assertNotIn(0.08, env.gripper_targets)

    def test_insertion_force_and_speed_limits_are_validated(self):
        for force in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(force=force), self.assertRaises(ValueError):
                _insertion_plan(force_n=force)

        plan = _insertion_plan()
        with self.assertRaisesRegex(ValueError, "insertion_speed_mps"):
            BoltMotionPlan(
                pick_waypoints=plan.pick_waypoints,
                transport_waypoints=plan.transport_waypoints,
                open_gripper_width_m=plan.open_gripper_width_m,
                grasp_gripper_width_m=plan.grasp_gripper_width_m,
                max_gripper_force_n=plan.max_gripper_force_n,
                seat_waypoint=plan.seat_waypoint,
                retract_waypoint=plan.retract_waypoint,
                max_insertion_force_n=plan.max_insertion_force_n,
                max_insertion_travel_m=plan.max_insertion_travel_m,
                insertion_speed_mps=0.006,
            )
        with self.assertRaisesRegex(ValueError, "held_seat_dwell_s"):
            BoltMotionPlan(
                pick_waypoints=plan.pick_waypoints,
                transport_waypoints=plan.transport_waypoints,
                open_gripper_width_m=plan.open_gripper_width_m,
                grasp_gripper_width_m=plan.grasp_gripper_width_m,
                max_gripper_force_n=plan.max_gripper_force_n,
                seat_waypoint=plan.seat_waypoint,
                retract_waypoint=plan.retract_waypoint,
                max_insertion_force_n=plan.max_insertion_force_n,
                max_insertion_travel_m=plan.max_insertion_travel_m,
                held_seat_dwell_s=0.1,
            )


if __name__ == "__main__":
    unittest.main()
