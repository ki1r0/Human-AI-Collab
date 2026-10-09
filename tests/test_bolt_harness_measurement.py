"""Synthetic CAD pose/contact fixtures; no P3 physics success is asserted."""

import unittest
from dataclasses import replace
from math import cos, pi, sin
from types import SimpleNamespace

from runtime.bolt_harness.contacts import (
    BOLT_ACTOR_PATH,
    CAP_COLLIDER_PATH,
    FINGER_ACTOR_PATHS,
    BoltFingerContact,
)
from runtime.bolt_harness.measurement import (
    CAP_OUTER_RADIUS_M,
    CAP_BEARING_RADIUS_M,
    CAP_BEARING_RADIUS_SOURCE_UNITS,
    CAP_SUPPORT_PLANE_Z_M,
    CAP_UNDERSIDE_LOCAL_Y_M,
    CASING_ENTRY_PLANE_Z_M,
    CASING_INTERIOR_FLOOR_Z_M,
    CASING_MIN_RADIUS_M,
    CASING_RADIAL_CLEARANCE_M,
    BOLT_TIP_LOCAL_Y_M,
    COVER_MIN_RADIUS_M,
    SHANK_RADIUS_M,
    SHANK_RADIUS_SOURCE_UNITS,
    SELECTED_HOLE_XY_M,
    BoltSeatSamplingAdapter,
    BoltSeatSamplingCriteria,
    diagnose_native_pose_window,
    CapSupportContactCriteria,
    ContactPointReport,
    cap_support_contact,
    measure_bolt_pose_geometry,
    read_filtered_contact_reports,
    _cap_grasp_contact,
    _exact_cap_grasp_contact,
    _exact_cap_grasp_contact_batches,
)
from runtime.bolt_harness.evaluator import (
    BoltNativePoseSample,
    BoltPoseWindowTolerances,
    BoltSeatMeasurement,
    BoltSeatStatus,
    BoltSeatTolerances,
    IndependentBoltSeatingEvaluator,
)
from tools.calibrate_bolt_contact import (
    PROBE_RESET_OFFSETS_M,
    PROBE_RESET_QUATERNIONS_WXYZ,
    _validate_probe,
)


ROOT_POS = (0.58, -0.15, 0.794105702)
ROOT_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)
ENV_ORIGIN = (0.0, 0.0, 0.0)
ZERO_VELOCITY = (0.0, 0.0, 0.0)
SYNTHETIC_CONTACT_LIMITS = CapSupportContactCriteria(
    max_cap_patch_error_m=0.0001,
    max_contact_separation_m=0.0001,
    min_normal_alignment_cosine=0.95,
    normal_axis_sign=1,
    min_contact_force_n=0.01,
)
SYNTHETIC_ADAPTER_LIMITS = BoltSeatSamplingCriteria(
    cap_support=SYNTHETIC_CONTACT_LIMITS,
    min_cap_grasp_local_y_m=0.019,
    max_cap_grasp_patch_error_m=0.001,
    max_cap_grasp_contact_separation_m=0.0001,
    min_cap_grasp_force_n=0.05,
    max_cap_grasp_normal_alignment_cosine=0.5,
    max_robot_contact_separation_m=0.0001,
    min_robot_contact_force_n=0.01,
    max_controller_tcp_linear_speed_mps=0.001,
    max_controller_tcp_angular_speed_radps=0.01,
    max_controller_arm_joint_speed_radps=0.01,
    max_tcp_tracking_error_m=0.001,
    max_tcp_orientation_tracking_error_rad=0.01,
    release_gripper_width_m=0.05,
    min_pickup_root_lift_m=0.01,
)
SYNTHETIC_EVALUATOR_LIMITS = BoltSeatTolerances(
    max_cap_support_gap_m=0.002,
    min_casing_entry_depth_m=0.01,
    max_shaft_radial_error_m=0.001,
    max_penetration_m=0.0005,
    max_axial_motion_since_release_m=0.001,
    max_relative_linear_speed_mps=0.001,
    max_relative_angular_speed_radps=0.01,
    held_seat_dwell_s=0.25,
    stable_dwell_s=1.0,
    max_sample_gap_s=0.25,
)


def pose(pos=ROOT_POS, quat=ROOT_QUAT):
    return measure_bolt_pose_geometry(
        time_s=0.0,
        bolt_root_pos_w_m=pos,
        bolt_root_quat_wxyz=quat,
        bolt_root_lin_vel_w_mps=ZERO_VELOCITY,
        bolt_root_ang_vel_w_radps=ZERO_VELOCITY,
        env_origin_w_m=ENV_ORIGIN,
    )


def evaluator_sample(geometry, time_s, penetration_m, physically_held=True):
    return BoltSeatMeasurement(
        time_s=time_s,
        through_selected_cover_hole=geometry.through_selected_cover_hole,
        entered_selected_casing_opening=geometry.entered_selected_casing_opening,
        casing_entry_depth_m=geometry.casing_entry_depth_m,
        shaft_radial_error_m=geometry.shaft_radial_error_m,
        cap_support_gap_m=geometry.cap_support_gap_m,
        cap_support_contact=True,
        penetration_m=penetration_m,
        premature_bottoming=False,
        physically_held=physically_held,
        insertion_provenance_valid=True,
        controller_stopped=True,
        robot_contact_or_support=physically_held,
        axial_position_m=geometry.axial_position_m,
        relative_linear_speed_mps=geometry.relative_linear_speed_mps,
        relative_angular_speed_radps=geometry.relative_angular_speed_radps,
    )


class FakeContactView:
    def __init__(self):
        self.dt = None

    def get_contact_data(self, dt):
        self.dt = dt
        return (
            [[5.0], [1.5], [0.2]],
            [[9.0, 9.0, 9.0], [0.58, -0.142, CAP_SUPPORT_PLANE_Z_M], [0.58, -0.15, 0.80]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0], [0.0], [-0.001]],
            [[1, 2]],
            [[0, 1]],
        )


class RawContactBufferView:
    def __init__(self, forces, points, normals, separations, counts, starts):
        self.buffers = (forces, points, normals, separations, counts, starts)

    def get_contact_data(self, dt):
        return self.buffers


class SyntheticFilteredContactView:
    """Small raw-buffer fixture matching RigidContactView's six-buffer shape."""

    def __init__(self, filter_paths):
        self.filter_paths = tuple(filter_paths)
        self.filter_count = len(self.filter_paths)
        self.max_contact_data_count = 64
        self.reports = {index: () for index in range(self.filter_count)}
        self.dt = None

    def set_reports(self, reports):
        self.reports = {
            index: tuple(reports.get(index, ())) for index in range(self.filter_count)
        }

    def get_contact_data(self, dt):
        self.dt = dt
        forces, points, normals, separations = [], [], [], []
        counts, starts = [], []
        for index in range(self.filter_count):
            current = self.reports[index]
            starts.append(len(forces))
            counts.append(len(current))
            for report in current:
                forces.append([report.force_n])
                points.append(list(report.point_w_m))
                normals.append(list(report.normal_w))
                separations.append([report.separation_m])
        return (
            forces,
            points,
            normals,
            separations,
            [counts],
            [starts],
        )


class SyntheticBoltEnv:
    """Mutable, synthetic one-environment source for adapter unit histories."""

    def __init__(self):
        self.num_envs = 1
        self.physics_dt = 1.0 / 120.0
        self.scene = SimpleNamespace(env_origins=[[0.0, 0.0, 0.0]])
        self._bolt = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=[list((0.30, 0.25, 0.80))],
                root_quat_w=[list(ROOT_QUAT)],
                root_lin_vel_w=[list(ZERO_VELOCITY)],
                root_ang_vel_w=[list(ZERO_VELOCITY)],
            )
        )
        self._robot = SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=0.0))
        self.bilateral = False
        self.left_force = 0.0
        self.right_force = 0.0
        self.public_state = {
            "tcp_velocity": [[0.0] * 6],
            "joint_velocities": [[0.0] * 9],
            "tcp_tracking_error_m": [0.0],
            "tcp_orientation_tracking_error_rad": [0.0],
            "motion_safety_armed": True,
            "gripper_width_m": 0.01,
        }

    def get_bilateral_grasp_contact(self):
        return {
            "left_contact": [self.bilateral],
            "right_contact": [self.bilateral],
            "bilateral_contact": [self.bilateral],
            "left_force_n": [self.left_force],
            "right_force_n": [self.right_force],
        }

    def get_public_executor_state(self):
        return self.public_state


def support_report(env=None, *, force_n=1.0, normal=(0.0, 0.0, 1.0), separation=0.0):
    # The synthetic cover record is fixed at the nominal seated bolt pose.
    root = ROOT_POS
    return ContactPointReport(
        force_n,
        (root[0] + 0.008, root[1], root[2] + CAP_UNDERSIDE_LOCAL_Y_M),
        normal,
        separation,
    )


def grasp_report(env):
    return grasp_report_at(env)


def grasp_report_at(env, *, local_y=0.022, normal=(1.0, 0.0, 0.0)):
    root = env._bolt.data.root_pos_w[0]
    return ContactPointReport(
        -0.1,
        (root[0] + 0.008, root[1], root[2] + local_y),
        normal,
        0.0,
    )


def make_synthetic_adapter(
    *,
    contact_capacity=None,
    exact_contact_provider=None,
    native_pose_sample_provider=None,
    pose_window_tolerances=None,
    evaluator_limits=SYNTHETIC_EVALUATOR_LIMITS,
):
    env = SyntheticBoltEnv()
    bolt_view = SyntheticFilteredContactView(
        (
            "/World/envs/env_0/Cover/node/mesh_0",
            "/World/envs/env_0/Cover/node/mesh_1",
            "/World/envs/env_0/Casing/node/mesh_0",
            "/World/envs/env_0/Robot/panda_leftfinger",
            "/World/envs/env_0/Robot/panda_rightfinger",
            "/World/envs/env_0/Robot/panda_hand",
        )
    )
    bolt_view.filter_paths = (bolt_view.filter_paths,)
    views = {"bolt": bolt_view}
    source = {
        "contact_physx_view": bolt_view,
        # Filter metadata is flat even when the underlying view paths are grouped.
        "filter_map": [
            {
                "filter_index": 0,
                "filter_prim_path": "/World/envs/env_0/Cover/node/mesh_0",
                "category": "cover",
                "source_prim_path": "/World/envs/env_0/Cover/node/mesh_0",
            },
            {
                "filter_index": 1,
                "filter_prim_path": "/World/envs/env_0/Cover/node/mesh_1",
                "category": "cover",
                "source_prim_path": "/World/envs/env_0/Cover/node/mesh_1",
            },
            {
                "filter_index": 2,
                "filter_prim_path": "/World/envs/env_0/Casing/node/mesh_0",
                "category": "casing",
                "source_prim_path": "/World/envs/env_0/Casing/node/mesh_0",
            },
            {
                "filter_index": 3,
                "filter_prim_path": "/World/envs/env_0/Robot/panda_leftfinger",
                "category": "robot",
                "source_prim_path": "/World/envs/env_0/Robot/panda_leftfinger",
            },
            {
                "filter_index": 4,
                "filter_prim_path": "/World/envs/env_0/Robot/panda_rightfinger",
                "category": "robot",
                "source_prim_path": "/World/envs/env_0/Robot/panda_rightfinger",
            },
            {
                "filter_index": 5,
                "filter_prim_path": "/World/envs/env_0/Robot/panda_hand",
                "category": "robot",
                "source_prim_path": "/World/envs/env_0/Robot/panda_hand",
            },
        ],
        "dt_s": env.physics_dt,
        "contact_data_capacity": (
            bolt_view.max_contact_data_count
            if contact_capacity is None
            else contact_capacity
        ),
    }
    if exact_contact_provider is not None:
        source["exact_cap_finger_contact_provider"] = exact_contact_provider
    if native_pose_sample_provider is not None:
        source["native_pose_sample_provider"] = native_pose_sample_provider
        source["native_pose_dt_s"] = env.physics_dt
    if pose_window_tolerances is not None:
        source["pose_window_tolerances"] = {
            "max_position_excursion_m": pose_window_tolerances.max_position_excursion_m,
            "max_orientation_excursion_rad": pose_window_tolerances.max_orientation_excursion_rad,
        }
    adapter = BoltSeatSamplingAdapter(
        env,
        evaluator_limits,
        SYNTHETIC_ADAPTER_LIMITS,
        contact_source=source,
    )
    return env, views, adapter


def exact_cap_contact(
    side,
    *,
    force_on_bolt=(0.0, 0.2, 0.0),
    normal_on_bolt=(0.0, 1.0, 0.0),
    separation=-0.00001,
    cap_collider=True,
    finger_actor=None,
    position=(ROOT_POS[0] + 0.012, ROOT_POS[1], ROOT_POS[2] + 0.022),
):
    finger_actor = finger_actor or FINGER_ACTOR_PATHS[side]
    if side == "left":
        actor0, actor1 = BOLT_ACTOR_PATH, finger_actor
        collider0 = CAP_COLLIDER_PATH if cap_collider else "/World/envs/env_0/Bolt/collision_shaft"
        collider1 = finger_actor + "/collision"
    else:
        actor0, actor1 = finger_actor, BOLT_ACTOR_PATH
        collider0 = finger_actor + "/collision"
        collider1 = CAP_COLLIDER_PATH if cap_collider else "/World/envs/env_0/Bolt/collision_shaft"
    impulse = tuple(value * 0.01 for value in force_on_bolt)
    return BoltFingerContact(
        finger=side,
        event_type="CONTACT_PERSIST",
        header_index=0,
        contact_data_index=0,
        contact_data_offset=0,
        contact_data_count=1,
        actor0_path=actor0,
        actor1_path=actor1,
        collider0_path=collider0,
        collider1_path=collider1,
        raw_position_w_m=position,
        raw_normal_w=normal_on_bolt,
        raw_impulse_w_ns=impulse,
        raw_separation_m=separation,
        normal_on_bolt_w=normal_on_bolt,
        impulse_on_bolt_w_ns=impulse,
        force_on_bolt_w_n=force_on_bolt,
        normal_compression_n=sum(
            force_on_bolt[i] * normal_on_bolt[i] for i in range(3)
        ),
    )


def sample_adapter(
    env,
    views,
    time_s,
    *,
    root_pos=ROOT_POS,
    held=False,
    cover=(),
    casing=(),
    robot=(),
    finger_contacts=False,
    left_finger_contacts=None,
    right_finger_contacts=None,
    grasp_local_y=0.022,
    grasp_normal=(1.0, 0.0, 0.0),
    gripper_width=0.01,
    tcp_speed=0.0,
):
    env._robot._data._sim_timestamp = time_s
    env._bolt.data.root_pos_w = [list(root_pos)]
    env._bolt.data.root_quat_w = [list(ROOT_QUAT)]
    env._bolt.data.root_lin_vel_w = [list(ZERO_VELOCITY)]
    env._bolt.data.root_ang_vel_w = [list(ZERO_VELOCITY)]
    env.bilateral = held
    env.left_force = 0.2 if held else 0.0
    env.right_force = 0.2 if held else 0.0
    env.public_state["gripper_width_m"] = gripper_width
    env.public_state["tcp_velocity"] = [[tcp_speed, 0.0, 0.0, 0.0, 0.0, 0.0]]
    left_contacts_enabled = finger_contacts if left_finger_contacts is None else left_finger_contacts
    right_contacts_enabled = finger_contacts if right_finger_contacts is None else right_finger_contacts
    views["bolt"].set_reports(
        {
            0: cover[:1],
            1: cover[1:],
            2: casing,
            3: (grasp_report_at(env, local_y=grasp_local_y, normal=grasp_normal),)
            if left_contacts_enabled
            else (),
            4: (grasp_report_at(env, local_y=grasp_local_y, normal=grasp_normal),)
            if right_contacts_enabled
            else (),
            5: robot,
        }
    )
    return views


class BoltHarnessMeasurementTests(unittest.TestCase):
    def test_adapter_feeds_completed_native_pose_samples_and_retains_velocity_vectors(self):
        raw_batch = (
            {
                "physics_step_index": 11,
                "sim_timestamp_s": 1.0 / 120.0,
                "native_root_transform_raw": [
                    *ROOT_POS,
                    ROOT_QUAT[1],
                    ROOT_QUAT[2],
                    ROOT_QUAT[3],
                    ROOT_QUAT[0],
                ],
                "native_root_velocity_raw": [0.02, 0.0, 0.0, 0.0, 0.0, 1.32],
            },
            {
                "physics_step_index": 12,
                "sim_timestamp_s": 2.0 / 120.0,
                "native_root_transform_raw": [
                    *ROOT_POS,
                    ROOT_QUAT[1],
                    ROOT_QUAT[2],
                    ROOT_QUAT[3],
                    ROOT_QUAT[0],
                ],
                "native_root_velocity_raw": [0.02, 0.0, 0.0, 0.0, 0.0, 1.32],
            },
        )
        pose_limits = BoltPoseWindowTolerances(
            max_position_excursion_m=0.0001,
            max_orientation_excursion_rad=0.001,
        )
        env, views, adapter = make_synthetic_adapter(
            native_pose_sample_provider=lambda: raw_batch,
            pose_window_tolerances=pose_limits,
        )
        sample_adapter(env, views, 2.0 / 120.0)
        env._bolt.data.root_lin_vel_w = [[0.02, 0.0, 0.0]]
        env._bolt.data.root_ang_vel_w = [[0.0, 0.0, 1.32]]

        status = adapter.observe()
        measured = adapter._trace[-1].measurement
        self.assertEqual(status, BoltSeatStatus(False, False))
        self.assertEqual(len(measured.native_pose_samples), 2)
        self.assertEqual(measured.native_pose_samples[-1].physics_step_index, 12)
        self.assertEqual(measured.native_pose_samples[-1].native_angular_velocity_w_radps, (0.0, 0.0, 1.32))
        self.assertAlmostEqual(measured.relative_linear_speed_mps, 0.02)
        self.assertAlmostEqual(measured.relative_angular_speed_radps, 1.32)

    def test_adapter_requires_pose_provider_and_pose_tolerances_as_a_pair(self):
        provider = lambda: ()
        pose_limits = BoltPoseWindowTolerances(
            max_position_excursion_m=0.0001,
            max_orientation_excursion_rad=0.001,
        )
        with self.assertRaisesRegex(ValueError, "configured together"):
            make_synthetic_adapter(native_pose_sample_provider=provider)
        with self.assertRaisesRegex(ValueError, "configured together"):
            make_synthetic_adapter(pose_window_tolerances=pose_limits)

    def test_cover_away_probe_reset_lies_horizontal_clear_of_hole_and_above_cover(self):
        offset = PROBE_RESET_OFFSETS_M["cover-away"]
        self.assertGreater(offset[0], COVER_MIN_RADIUS_M + SHANK_RADIUS_M)
        self.assertEqual(PROBE_RESET_QUATERNIONS_WXYZ["cover-away"], (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            ROOT_POS[2] + offset[2] - CAP_OUTER_RADIUS_M - CAP_SUPPORT_PLANE_Z_M,
            0.002,
            places=7,
        )

    def test_cover_away_probe_accepts_real_cover_support_without_socket_alignment(self):
        row = {
            "geometry": {
                "through_selected_cover_hole": False,
                "entered_selected_casing_opening": False,
                "shaft_radial_error_m": 0.015,
            },
            "measurement": {
                "physically_held": False,
                "insertion_provenance_valid": False,
                "robot_contact_or_support": False,
                "cap_support_contact": True,
                "shaft_radial_error_m": 0.015,
                "relative_linear_speed_mps": 0.0,
                "relative_angular_speed_radps": 0.0,
            },
            "private_contact_evidence": {
                "cover_contact_count": 1,
                "casing_contact_count": 0,
            },
            "status": {"seat_ready": False, "task_success": False},
        }
        result = _validate_probe(
            "cover-away",
            [row, row],
            stable_window_samples=2,
            tolerances=SYNTHETIC_EVALUATOR_LIMITS,
        )
        self.assertTrue(result["probe_validation_passed"])
        self.assertTrue(result["checks"]["cover_contact_away_from_selected_socket"])

    def test_nominal_seated_pose_uses_source_local_positive_y_axis(self):
        measured = pose()
        self.assertAlmostEqual(measured.bolt_axis_w[0], 0.0, places=7)
        self.assertAlmostEqual(measured.bolt_axis_w[1], 0.0, places=7)
        self.assertAlmostEqual(measured.bolt_axis_w[2], 1.0, places=7)
        self.assertTrue(measured.through_selected_cover_hole)
        self.assertTrue(measured.entered_selected_casing_opening)
        self.assertAlmostEqual(measured.casing_entry_depth_m, 0.0238376, places=6)
        self.assertAlmostEqual(measured.shaft_radial_error_m, 0.0, places=7)
        self.assertAlmostEqual(measured.cap_support_gap_m, 0.0, places=8)
        self.assertAlmostEqual(measured.axial_position_m, 0.05820905, places=7)
        self.assertAlmostEqual(
            measured.tip_to_interior_floor_clearance_m,
            ROOT_POS[2] + BOLT_TIP_LOCAL_Y_M - CASING_INTERIOR_FLOOR_Z_M,
            places=8,
        )
        self.assertFalse(measured.premature_bottoming)
        self.assertFalse(hasattr(measured, "penetration_m"))

    def test_static_negative_poses_cover_surface_wrong_hole_and_no_casing(self):
        cover_only = pose((0.58, -0.15, 0.826149998))
        self.assertTrue(cover_only.through_selected_cover_hole)
        self.assertFalse(cover_only.entered_selected_casing_opening)

        above_cover = pose((0.58, -0.15, 0.838149998))
        self.assertFalse(above_cover.through_selected_cover_hole)

        wrong_hole = pose((0.60, -0.15, ROOT_POS[2]))
        self.assertFalse(wrong_hole.through_selected_cover_hole)
        self.assertFalse(wrong_hole.entered_selected_casing_opening)
        self.assertFalse(wrong_hole.cover_shaft_penetration_applicable)
        self.assertFalse(wrong_hole.casing_shaft_penetration_applicable)
        self.assertFalse(wrong_hole.cap_bearing_patch_applicable)
        self.assertEqual(wrong_hole.geometric_penetration_bound_m, 0.0)
        self.assertGreater(wrong_hole.shaft_radial_error_m, CASING_RADIAL_CLEARANCE_M)

    def test_audited_apertures_and_shaft_envelope_report_intrusion_beyond_clearance(self):
        self.assertAlmostEqual(SHANK_RADIUS_M, 0.005884001993747715, places=12)
        self.assertAlmostEqual(CASING_MIN_RADIUS_M, 0.006221929427494521, places=12)
        self.assertAlmostEqual(COVER_MIN_RADIUS_M, 0.007295854101598516, places=12)
        self.assertAlmostEqual(CAP_BEARING_RADIUS_SOURCE_UNITS, 5.000001, places=9)
        self.assertAlmostEqual(CAP_BEARING_RADIUS_M, 0.010000002, places=10)
        self.assertAlmostEqual(
            SHANK_RADIUS_M, SHANK_RADIUS_SOURCE_UNITS * 0.002, places=14
        )

        excess_m = 0.0001
        beyond_casing_clearance = pose(
            (ROOT_POS[0] + CASING_RADIAL_CLEARANCE_M + excess_m, ROOT_POS[1], ROOT_POS[2])
        )
        self.assertGreater(beyond_casing_clearance.casing_shaft_penetration_bound_m, 0.0)
        self.assertAlmostEqual(
            beyond_casing_clearance.casing_shaft_penetration_bound_m, excess_m, places=8
        )
        self.assertAlmostEqual(
            beyond_casing_clearance.geometric_penetration_bound_m, excess_m, places=8
        )
        self.assertFalse(beyond_casing_clearance.entered_selected_casing_opening)

    def test_tilt_projects_the_full_shaft_envelope_at_the_casing_plane(self):
        angle_rad = 25.0 * pi / 180.0
        half_angle = (pi / 2.0 + angle_rad) / 2.0
        quaternion = (cos(half_angle), sin(half_angle), 0.0, 0.0)
        axis_y, axis_z = -sin(angle_rad), cos(angle_rad)
        axis_distance = (CASING_ENTRY_PLANE_Z_M - ROOT_POS[2]) / axis_z
        root_y = SELECTED_HOLE_XY_M[1] - axis_y * axis_distance
        tilted = pose((SELECTED_HOLE_XY_M[0], root_y, ROOT_POS[2]), quaternion)

        expected = max(0.0, SHANK_RADIUS_M / axis_z - CASING_MIN_RADIUS_M)
        self.assertAlmostEqual(tilted.casing_entry_radial_error_m, 0.0, places=8)
        self.assertAlmostEqual(tilted.casing_shaft_penetration_bound_m, expected, places=8)
        self.assertGreater(tilted.casing_shaft_penetration_bound_m, 0.0)
        self.assertFalse(tilted.entered_selected_casing_opening)

    def test_cap_bearing_bound_uses_lowest_tilted_face_point(self):
        angle_rad = 3.0 * pi / 180.0
        half_angle = (pi / 2.0 + angle_rad) / 2.0
        quaternion = (cos(half_angle), sin(half_angle), 0.0, 0.0)
        axis_z = cos(angle_rad)
        root_z = CAP_SUPPORT_PLANE_Z_M - axis_z * CAP_UNDERSIDE_LOCAL_Y_M
        tilted_cap = pose((ROOT_POS[0], ROOT_POS[1], root_z), quaternion)

        # Use the audited outer-head radius as a conservative bound for the
        # tilted underside bevel; contact classification remains on the flat face.
        expected = CAP_OUTER_RADIUS_M * sin(angle_rad)
        self.assertAlmostEqual(tilted_cap.cap_support_gap_m, 0.0, places=8)
        self.assertTrue(tilted_cap.cap_bearing_patch_applicable)
        self.assertAlmostEqual(tilted_cap.cap_head_penetration_bound_m, expected, places=8)
        self.assertAlmostEqual(tilted_cap.geometric_penetration_bound_m, expected, places=8)

        lowered = pose((ROOT_POS[0], ROOT_POS[1], ROOT_POS[2] - 0.0001))
        self.assertAlmostEqual(lowered.cap_head_penetration_bound_m, 0.0001, places=8)

    def test_hover_and_not_entered_poses_have_no_geometry_penetration(self):
        hover = pose((ROOT_POS[0], ROOT_POS[1], ROOT_POS[2] + 0.002))
        self.assertEqual(hover.geometric_penetration_bound_m, 0.0)
        self.assertFalse(cap_support_contact(hover, (), SYNTHETIC_CONTACT_LIMITS))

        root_z = CASING_ENTRY_PLANE_Z_M - BOLT_TIP_LOCAL_Y_M + 0.001
        not_entered = pose((ROOT_POS[0], ROOT_POS[1], root_z))
        self.assertFalse(not_entered.entered_selected_casing_opening)
        self.assertEqual(not_entered.geometric_penetration_bound_m, 0.0)

    def test_free_bolt_below_cover_plane_outside_socket_has_no_local_penetration(self):
        free_table_bolt = pose((0.30, 0.25, CAP_SUPPORT_PLANE_Z_M - 0.088))

        self.assertFalse(free_table_bolt.through_selected_cover_hole)
        self.assertFalse(free_table_bolt.entered_selected_casing_opening)
        self.assertFalse(free_table_bolt.cover_shaft_penetration_applicable)
        self.assertFalse(free_table_bolt.casing_shaft_penetration_applicable)
        self.assertFalse(free_table_bolt.cap_bearing_patch_applicable)
        self.assertEqual(free_table_bolt.cover_shaft_penetration_bound_m, 0.0)
        self.assertEqual(free_table_bolt.casing_shaft_penetration_bound_m, 0.0)
        self.assertEqual(free_table_bolt.cap_head_penetration_bound_m, 0.0)
        self.assertEqual(free_table_bolt.geometric_penetration_bound_m, 0.0)

    def test_native_pose_window_reports_pose_motion_separately_from_native_velocity(self):
        samples = []
        for index, (x_m, angle_rad) in enumerate(
            ((0.0, 0.0), (0.000002, 0.0002), (0.0, 0.0))
        ):
            samples.append(
                {
                    "sim_timestamp_s": 65.0 + index / 120.0,
                    "native_root_transform_raw": [
                        x_m,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        sin(angle_rad / 2.0),
                        cos(angle_rad / 2.0),
                    ],
                    "native_root_velocity_raw": [0.02, 0.0, 0.0, 0.0, 0.0, 1.32],
                }
            )

        diagnostic = diagnose_native_pose_window(samples)
        self.assertEqual(diagnostic.sample_count, 3)
        self.assertAlmostEqual(diagnostic.duration_s, 1.0 / 60.0, places=10)
        self.assertAlmostEqual(diagnostic.position_bbox_diagonal_span_m, 0.000002)
        self.assertAlmostEqual(diagnostic.max_orientation_deviation_from_first_rad, 0.0002, places=8)
        self.assertAlmostEqual(diagnostic.max_pose_fd_linear_speed_mps, 0.00024, places=8)
        self.assertAlmostEqual(diagnostic.max_pose_fd_angular_speed_radps, 0.024, places=7)
        self.assertEqual(diagnostic.max_native_linear_speed_mps, 0.02)
        self.assertEqual(diagnostic.max_native_angular_speed_radps, 1.32)

    def test_native_pose_window_rejects_missing_or_nonmonotonic_physics_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            diagnose_native_pose_window([])
        sample = {
            "sim_timestamp_s": 1.0,
            "native_root_transform_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "native_root_velocity_raw": [0.0] * 6,
        }
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            diagnose_native_pose_window([sample, dict(sample)])

    def test_interior_floor_crossing_is_marked_as_premature_bottoming(self):
        tip_below_floor = CASING_INTERIOR_FLOOR_Z_M - 0.00005
        root_z = tip_below_floor - BOLT_TIP_LOCAL_Y_M
        bottomed = pose((ROOT_POS[0], ROOT_POS[1], root_z))
        self.assertTrue(bottomed.entered_selected_casing_opening)
        self.assertAlmostEqual(bottomed.tip_to_interior_floor_clearance_m, -0.00005, places=8)
        self.assertTrue(bottomed.premature_bottoming)

    def test_small_negative_cap_gap_preserves_path_and_evaluator_checks_penetration(self):
        solver_contact_pose = pose((ROOT_POS[0], ROOT_POS[1], ROOT_POS[2] - 0.0001))
        self.assertTrue(solver_contact_pose.through_selected_cover_hole)
        self.assertTrue(solver_contact_pose.entered_selected_casing_opening)
        self.assertAlmostEqual(solver_contact_pose.cap_support_gap_m, -0.0001, places=7)

        admitted = IndependentBoltSeatingEvaluator(SYNTHETIC_EVALUATOR_LIMITS)
        admitted_status = [
            admitted.observe(evaluator_sample(solver_contact_pose, t, 0.0001))
            for t in (0.0, 0.125, 0.25)
        ]
        self.assertEqual(admitted_status[-1], BoltSeatStatus(True, False))

        penetrating = IndependentBoltSeatingEvaluator(SYNTHETIC_EVALUATOR_LIMITS)
        rejected_status = [
            penetrating.observe(evaluator_sample(solver_contact_pose, t, 0.0006))
            for t in (0.0, 0.125, 0.25)
        ]
        self.assertTrue(all(not status.seat_ready for status in rejected_status))
        self.assertEqual(
            penetrating.observe(
                evaluator_sample(solver_contact_pose, 0.375, 0.0006, physically_held=False)
            ),
            BoltSeatStatus(False, False),
        )

    def test_pose_gap_reports_hover_without_claiming_actual_contact(self):
        hover = pose((ROOT_POS[0], ROOT_POS[1], ROOT_POS[2] + 0.002))
        self.assertGreater(hover.cap_support_gap_m, 0.0019)
        self.assertTrue(hover.through_selected_cover_hole)
        self.assertEqual(hover.geometric_penetration_bound_m, 0.0)
        self.assertFalse(cap_support_contact(hover, (), SYNTHETIC_CONTACT_LIMITS))

    def test_cap_support_requires_filtered_contact_on_underside_patch_and_normal(self):
        measured = pose()
        point = (
            ROOT_POS[0] + 0.008,
            ROOT_POS[1],
            ROOT_POS[2] + CAP_UNDERSIDE_LOCAL_Y_M,
        )
        valid_report = ContactPointReport(1.0, point, (0.0, 0.0, 1.0), 0.0)
        self.assertTrue(cap_support_contact(measured, (valid_report,), SYNTHETIC_CONTACT_LIMITS))
        distributed_reports = (
            ContactPointReport(0.006, point, (0.0, 0.0, 1.0), 0.0),
            ContactPointReport(0.006, point, (0.0, 0.0, 1.0), 0.0),
        )
        self.assertTrue(
            cap_support_contact(measured, distributed_reports, SYNTHETIC_CONTACT_LIMITS)
        )

        side_report = ContactPointReport(1.0, point, (1.0, 0.0, 0.0), 0.0)
        opposite_axial_report = ContactPointReport(1.0, point, (0.0, 0.0, -1.0), 0.0)
        weak_report = ContactPointReport(0.001, point, (0.0, 0.0, 1.0), 0.0)
        separated_report = ContactPointReport(1.0, point, (0.0, 0.0, 1.0), 0.001)
        wrong_patch = ContactPointReport(
            1.0,
            (ROOT_POS[0] + CAP_OUTER_RADIUS_M + 0.002, ROOT_POS[1], point[2]),
            (0.0, 0.0, 1.0),
            0.0,
        )
        outside_bearing_face = ContactPointReport(
            1.0,
            (ROOT_POS[0] + CAP_BEARING_RADIUS_M + 0.0002, ROOT_POS[1], point[2]),
            (0.0, 0.0, 1.0),
            0.0,
        )
        self.assertFalse(cap_support_contact(measured, (side_report,), SYNTHETIC_CONTACT_LIMITS))
        self.assertFalse(
            cap_support_contact(measured, (opposite_axial_report,), SYNTHETIC_CONTACT_LIMITS)
        )
        signed_support = ContactPointReport(-1.0, point, (0.0, 0.0, -1.0), 0.0)
        signed_wrong_direction = ContactPointReport(-1.0, point, (0.0, 0.0, 1.0), 0.0)
        self.assertTrue(
            cap_support_contact(measured, (signed_support,), SYNTHETIC_CONTACT_LIMITS)
        )
        self.assertFalse(
            cap_support_contact(measured, (signed_wrong_direction,), SYNTHETIC_CONTACT_LIMITS)
        )
        self.assertFalse(cap_support_contact(measured, (weak_report,), SYNTHETIC_CONTACT_LIMITS))
        self.assertLess(
            CAP_BEARING_RADIUS_M + 0.0002, CAP_OUTER_RADIUS_M,
            "fixture should be on the head bevel/side, not outside the whole head",
        )
        self.assertFalse(
            cap_support_contact(measured, (outside_bearing_face,), SYNTHETIC_CONTACT_LIMITS)
        )
        self.assertFalse(
            cap_support_contact(measured, (separated_report,), SYNTHETIC_CONTACT_LIMITS)
        )
        self.assertFalse(cap_support_contact(measured, (wrong_patch,), SYNTHETIC_CONTACT_LIMITS))

    def test_signed_scalar_times_normal_identifies_compression_on_both_finger_sides(self):
        geometry = pose()
        point_z = ROOT_POS[2] + 0.022
        right = ContactPointReport(
            -0.03761463239789009,
            (ROOT_POS[0], ROOT_POS[1] + 0.008, point_z),
            (0.0, 1.0, 0.0),
            -2.640888624227955e-06,
        )
        left = ContactPointReport(
            -1.0700167417526245,
            (ROOT_POS[0], ROOT_POS[1] - 0.008, point_z),
            (0.0, -1.0, 0.0),
            -1.956904452526942e-05,
        )

        self.assertEqual(right.force_vector_w_n, (0.0, -0.03761463239789009, 0.0))
        self.assertEqual(left.force_vector_w_n, (0.0, 1.0700167417526245, 0.0))
        self.assertLess(right.force_vector_w_n[1] * (right.point_w_m[1] - ROOT_POS[1]), 0.0)
        self.assertLess(left.force_vector_w_n[1] * (left.point_w_m[1] - ROOT_POS[1]), 0.0)
        signed_grasp_limits = replace(SYNTHETIC_ADAPTER_LIMITS, min_cap_grasp_force_n=0.01)
        self.assertTrue(_cap_grasp_contact(geometry, (right,), signed_grasp_limits))
        self.assertTrue(_cap_grasp_contact(geometry, (left,), signed_grasp_limits))

        reversed_right = ContactPointReport(
            -right.force_n, right.point_w_m, right.normal_w, right.separation_m
        )
        self.assertFalse(
            _cap_grasp_contact(geometry, (reversed_right,), signed_grasp_limits)
        )

    def test_exact_cap_pair_uses_ids_and_signed_compression_not_current_pose_patch(self):
        geometry = pose()
        limits = replace(SYNTHETIC_ADAPTER_LIMITS, min_cap_grasp_force_n=0.05)
        radial_point = exact_cap_contact("left").raw_position_w_m
        self.assertGreater(
            ((radial_point[0] - ROOT_POS[0]) ** 2 + (radial_point[1] - ROOT_POS[1]) ** 2)
            ** 0.5,
            CAP_OUTER_RADIUS_M,
        )

        for side in ("left", "right"):
            with self.subTest(side=side):
                report = exact_cap_contact(side)
                self.assertTrue(
                    _exact_cap_grasp_contact(
                        geometry, (report,), limits, expected_finger=side
                    )
                )
                self.assertFalse(
                    _exact_cap_grasp_contact(
                        geometry,
                        (report,),
                        limits,
                        expected_finger="right" if side == "left" else "left",
                    )
                )

        invalid_contacts = (
            exact_cap_contact("left", force_on_bolt=(0.0, -0.2, 0.0)),
            exact_cap_contact("left", cap_collider=False),
            exact_cap_contact("left", finger_actor=FINGER_ACTOR_PATHS["right"]),
            exact_cap_contact("left", normal_on_bolt=(0.0, 0.0, 1.0)),
            exact_cap_contact("left", separation=0.001),
        )
        for report in invalid_contacts:
            with self.subTest(report=report):
                self.assertFalse(
                    _exact_cap_grasp_contact(
                        geometry, (report,), limits, expected_finger="left"
                    )
                )

    def test_exact_contact_provider_overrides_moving_root_patch_and_is_traced(self):
        contacts = {
            "left": ((exact_cap_contact("left"),),),
            "right": ((exact_cap_contact("right"),),),
        }
        env, views, adapter = make_synthetic_adapter(
            exact_contact_provider=lambda: contacts
        )
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()
        for time_s in (0.125, 0.25):
            sample_adapter(
                env,
                views,
                time_s,
                root_pos=(0.58, -0.15, 0.85),
                held=True,
                finger_contacts=False,
            )
            adapter.observe()

        sample = adapter._trace[-1]
        self.assertTrue(sample.left_cap_grasp)
        self.assertTrue(sample.right_cap_grasp)
        self.assertTrue(sample.measurement.physically_held)
        self.assertTrue(sample.pickup_proven)
        self.assertEqual(sample.exact_left_cap_grasp_contacts, contacts["left"][0])
        self.assertEqual(sample.exact_right_cap_grasp_contacts, contacts["right"][0])
        self.assertEqual(sample.exact_left_cap_grasp_contact_batches, contacts["left"])
        self.assertEqual(sample.exact_right_cap_grasp_contact_batches, contacts["right"])

    def test_exact_contact_provider_requires_both_per_finger_sequences(self):
        provider_value = {
            "left": ((exact_cap_contact("left"),),),
            "right": ((exact_cap_contact("right"),),),
        }
        env, views, adapter = make_synthetic_adapter(
            exact_contact_provider=lambda: provider_value
        )
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()
        provider_value.pop("right")
        sample_adapter(
            env,
            views,
            0.125,
            root_pos=(0.58, -0.15, 0.85),
            held=True,
            finger_contacts=False,
        )
        with self.assertRaisesRegex(TypeError, "batches for right"):
            adapter.observe()

    def test_exact_contact_provider_rejects_unbatched_multi_tick_force_records(self):
        env, views, adapter = make_synthetic_adapter(
            exact_contact_provider=lambda: {
                "left": (exact_cap_contact("left"),),
                "right": (exact_cap_contact("right"),),
            }
        )
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        with self.assertRaisesRegex(TypeError, "per-physics-tick batches"):
            adapter.observe()

    def test_exact_grasp_force_sums_manifold_points_but_not_separate_physics_ticks(self):
        geometry = pose()
        limits = replace(SYNTHETIC_ADAPTER_LIMITS, min_cap_grasp_force_n=0.05)
        first = exact_cap_contact("left", force_on_bolt=(0.0, 0.03, 0.0))
        second = exact_cap_contact("left", force_on_bolt=(0.0, 0.03, 0.0))

        self.assertTrue(
            _exact_cap_grasp_contact(
                geometry, (first, second), limits, expected_finger="left"
            )
        )
        self.assertFalse(
            _exact_cap_grasp_contact_batches(
                geometry,
                ((first,), (second,)),
                limits,
                expected_finger="left",
            )
        )

    def test_raw_filtered_contact_view_records_preserve_per_point_normals(self):
        view = FakeContactView()
        reports = read_filtered_contact_reports(
            view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=1
        )
        self.assertAlmostEqual(view.dt, 1.0 / 120.0)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0].point_w_m, (0.58, -0.142, CAP_SUPPORT_PLANE_Z_M))
        self.assertEqual(reports[0].normal_w, (0.0, 0.0, 1.0))
        self.assertEqual(reports[1].separation_m, -0.001)

    def test_raw_filtered_contact_reader_preserves_signed_force_and_rejects_nonfinite(self):
        forces = [[0.0] for _ in range(12)]
        points = [[0.0, 0.0, 0.0] for _ in range(12)]
        normals = [[1.0, 0.0, 0.0] for _ in range(12)]
        separations = [[0.0] for _ in range(12)]
        forces[0] = [-0.03761463239789009]
        points[0] = [0.30577850341796875, 0.25996333360671997, 0.7262306809425354]
        normals[0] = [-0.0003099033492617309, 0.9999995827674866, 0.0007943024393171072]
        separations[0] = [-2.640888624227955e-06]
        forces[6] = [-1.0700167417526245]
        points[6] = [0.29423388838768005, 0.23997114598751068, 0.7265112400054932]
        normals[6] = [-0.00044799144961871207, -0.9999998211860657, -9.817798127187416e-05]
        separations[6] = [-1.956904452526942e-05]
        view = RawContactBufferView(
            forces,
            points,
            normals,
            separations,
            [[0, 0, 0, 1, 1]],
            [[0, 0, 0, 0, 6]],
        )
        right = read_filtered_contact_reports(
            view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=3
        )[0]
        left = read_filtered_contact_reports(
            view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=4
        )[0]
        self.assertEqual(right.force_n, -0.03761463239789009)
        self.assertEqual(left.force_n, -1.0700167417526245)
        self.assertLess(right.force_vector_w_n[1], 0.0)
        self.assertGreater(left.force_vector_w_n[1], 0.0)

        for raw_force in (float("nan"), float("inf")):
            with self.subTest(raw_force=raw_force):
                view = RawContactBufferView(
                    [[raw_force]],
                    [[0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0]],
                    [[0.0]],
                    [[0, 0, 0, 1]],
                    [[0, 0, 0, 0]],
                )
                with self.assertRaises(ValueError) as raised:
                    read_filtered_contact_reports(
                        view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=3
                    )
                message = str(raised.exception)
                self.assertIn("filter_index=3", message)
                self.assertIn("point_index=0", message)
                self.assertIn(f"force_n_raw={raw_force!r}", message)

    def test_raw_filtered_contact_reader_accepts_empty_filter_and_rejects_bad_counts(self):
        empty_view = RawContactBufferView([], [], [], [], [[0]], [[0]])
        self.assertEqual(
            read_filtered_contact_reports(
                empty_view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=0
            ),
            (),
        )

        truncated_view = RawContactBufferView(
            [[1.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0]],
            [[2]],
            [[0]],
        )
        with self.assertRaisesRegex(ValueError, "exceed reported data arrays"):
            read_filtered_contact_reports(
                truncated_view, dt_s=1.0 / 120.0, sensor_index=0, filter_index=0
            )

    def test_input_pose_and_contact_calibration_are_explicit_and_validated(self):
        with self.assertRaises(ValueError):
            pose(quat=(0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            CapSupportContactCriteria(0.0, 0.0, 0.0, 1, 0.01)
        with self.assertRaises(ValueError):
            CapSupportContactCriteria(0.001, 0.0, 0.95, 0, 0.01)
        with self.assertRaises(ValueError):
            ContactPointReport(float("nan"), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            ContactPointReport(float("inf"), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            ContactPointReport(1.0, (0.0, 0.0), (0.0, 0.0, 1.0), 0.0)

    def test_synthetic_adapter_exposes_only_boolean_status_and_accepts_unassisted_dwell(self):
        env, views, adapter = make_synthetic_adapter()
        statuses = []
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()

        for time_s in (0.125,):
            sample_adapter(
                env,
                views,
                time_s,
                root_pos=(0.58, -0.15, 0.85),
                held=True,
                finger_contacts=True,
                tcp_speed=0.02,
            )
            statuses.append(adapter.observe())

        # Two separate Cover mesh filters are read; the cap support is on mesh 1.
        support = (
            ContactPointReport(0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),
            support_report(),
        )
        sample_adapter(
            env, views, 0.25, held=True, cover=support, finger_contacts=True
        )
        statuses.append(adapter.observe())
        sample_adapter(
            env, views, 0.375, held=True, cover=support, finger_contacts=True
        )
        statuses.append(adapter.observe())
        sample_adapter(
            env, views, 0.5, held=True, cover=support, finger_contacts=True
        )
        statuses.append(adapter.observe())
        self.assertTrue(statuses[-1].seat_ready)
        self.assertFalse(statuses[-1].task_success)

        # Contact loss at partial opening starts release only after held-seat
        # certification. Residual contact from the other finger delays dwell.
        sample_adapter(
            env,
            views,
            0.625,
            held=False,
            cover=support,
            left_finger_contacts=True,
            right_finger_contacts=False,
            gripper_width=0.021,
        )
        lingering = adapter.observe()
        self.assertEqual(lingering, BoltSeatStatus(False, False))
        sample_adapter(
            env,
            views,
            0.75,
            cover=support,
            left_finger_contacts=True,
            right_finger_contacts=False,
            gripper_width=0.03,
        )
        self.assertEqual(adapter.observe(), BoltSeatStatus(False, False))
        for time_s in (1.0, 1.25, 1.5, 1.75):
            sample_adapter(env, views, time_s, cover=support)
            status = adapter.observe()
            if time_s < 1.75:
                self.assertEqual(status, BoltSeatStatus(False, False))
        sample_adapter(env, views, 2.0, cover=support)
        status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, True))

        self.assertEqual(len(BoltSeatStatus.__dataclass_fields__), 2)
        self.assertTrue(all(type(value) is bool for value in vars(adapter._trace[-1].status).values()))
        self.assertEqual(len(adapter._trace), 12)
        self.assertEqual(adapter._trace[2].held_inserted_sample_count, 1)
        self.assertTrue(adapter._trace[5].release_started)
        self.assertTrue(adapter._trace[5].held_seat_certified)
        self.assertTrue(adapter._trace[5].pickup_sequence_valid)
        self.assertTrue(adapter._trace[6].pickup_sequence_valid)
        self.assertTrue(adapter._trace[2].pickup_proven)
        self.assertAlmostEqual(adapter._trace[1].pickup_root_lift_m, 0.05)
        self.assertEqual(len(adapter._trace[2].left_grasp_contacts), 1)
        self.assertEqual(len(adapter._trace[2].right_grasp_contacts), 1)
        self.assertEqual(len(adapter._trace[2].robot_contacts_by_source), 3)
        self.assertEqual(len(adapter._trace[2].cover_contacts), 2)
        self.assertAlmostEqual(views["bolt"].dt, env.physics_dt)

    def test_synthetic_adapter_rejects_missing_unheld_reset_provenance(self):
        env, views, adapter = make_synthetic_adapter()
        support = support_report(env)
        for time_s in (0.0, 0.125, 0.25, 0.375):
            sample_adapter(
                env,
                views,
                time_s,
                held=True,
                cover=(support_report(env),),
                finger_contacts=True,
            )
            status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, False))
        self.assertFalse(adapter._trace[-1].measurement.insertion_provenance_valid)

    def test_synthetic_adapter_requires_per_side_cap_contact_not_bilateral_force_or_width(self):
        invalid_grasp_evidence = (
            {"grasp_local_y": CAP_UNDERSIDE_LOCAL_Y_M - 0.001},
            {"grasp_normal": (0.0, 0.0, 1.0)},
            {"right_finger_contacts": False},
            {"finger_contacts": False},
        )
        for evidence in invalid_grasp_evidence:
            env, views, adapter = make_synthetic_adapter()
            sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
            adapter.observe()
            sample_adapter(
                env,
                views,
                0.125,
                root_pos=(0.58, -0.15, 0.85),
                held=True,
                **evidence,
            )
            adapter.observe()
            sample = adapter._trace[-1]
            self.assertTrue(sample.bilateral_contact)
            self.assertFalse(sample.measurement.physically_held)
            self.assertFalse(sample.pickup_proven)

    def test_synthetic_adapter_requires_root_lift_and_contiguous_held_pickup_sequence(self):
        env, views, adapter = make_synthetic_adapter()
        sample_adapter(env, views, 0.0, root_pos=(0.58, -0.15, 0.85))
        adapter.observe()
        sample_adapter(
            env,
            views,
            0.125,
            root_pos=(0.58, -0.15, 0.85),
            held=True,
            finger_contacts=True,
        )
        adapter.observe()
        sample_adapter(env, views, 0.25, held=True, finger_contacts=True)
        adapter.observe()
        self.assertFalse(adapter._trace[-1].pickup_proven)
        self.assertFalse(adapter._trace[-1].measurement.insertion_provenance_valid)

        env, views, adapter = make_synthetic_adapter()
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()
        for time_s in (0.125, 0.25):
            sample_adapter(
                env,
                views,
                time_s,
                root_pos=(0.58, -0.15, 0.85),
                held=True,
                finger_contacts=True,
            )
            status = adapter.observe()
        self.assertTrue(adapter._trace[-1].pickup_proven)
        sample_adapter(env, views, 0.375, root_pos=(0.58, -0.15, 0.85))
        status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, False))
        self.assertFalse(adapter._trace[-1].pickup_sequence_valid)
        sample_adapter(env, views, 0.5, held=True, finger_contacts=True)
        status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, False))
        self.assertFalse(adapter._trace[-1].measurement.insertion_provenance_valid)

    def test_synthetic_adapter_refuses_contact_buffer_capacity_saturation(self):
        env, views, adapter = make_synthetic_adapter(contact_capacity=2)
        sample_adapter(
            env,
            views,
            0.0,
            cover=(support_report(), support_report()),
        )
        with self.assertRaisesRegex(RuntimeError, "reached configured capacity"):
            adapter.observe()

    def test_synthetic_adapter_keeps_raw_separation_diagnostic_out_of_penetration(self):
        for bad_contact in (
            ContactPointReport(
                1.0,
                (ROOT_POS[0] + 0.008, ROOT_POS[1], ROOT_POS[2] + CAP_UNDERSIDE_LOCAL_Y_M),
                (0.0, 0.0, -1.0),
                0.0,
            ),
        ):
            env, views, adapter = make_synthetic_adapter()
            for time_s in (0.0, 0.125, 0.25, 0.375):
                root_pos = (0.30, 0.25, 0.80) if time_s == 0.0 else (
                    (0.58, -0.15, 0.85) if time_s == 0.125 else ROOT_POS
                )
                report = bad_contact if time_s >= 0.25 else ()
                sample_adapter(
                    env,
                    views,
                    time_s,
                    root_pos=root_pos,
                    held=time_s > 0.0,
                    cover=(report,) if report else (),
                    finger_contacts=time_s > 0.0,
                )
                status = adapter.observe()
            self.assertEqual(status, BoltSeatStatus(False, False))
            self.assertFalse(adapter._trace[-1].measurement.cap_support_contact)

        env, views, adapter = make_synthetic_adapter()
        geometry = pose()
        for time_s in (0.0, 0.125, 0.25, 0.375):
            root_pos = (0.30, 0.25, 0.80) if time_s == 0.0 else (
                (0.58, -0.15, 0.85) if time_s == 0.125 else ROOT_POS
            )
            contacts = ()
            if time_s >= 0.25:
                contacts = (support_report(env),)
                if time_s == 0.375:
                    contacts += (ContactPointReport(0.1, (0, 0, 0), (1, 0, 0), -0.0006),)
            sample_adapter(
                env,
                views,
                time_s,
                root_pos=root_pos,
                held=time_s > 0.0,
                cover=contacts,
                finger_contacts=time_s > 0.0,
            )
            status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, False))
        self.assertAlmostEqual(
            adapter._trace[-1].measurement.penetration_m,
            geometry.geometric_penetration_bound_m,
        )
        self.assertTrue(
            any(
                report.separation_m == -0.0006
                for report in adapter._trace[-1].cover_contacts
            )
        )

    def test_synthetic_adapter_rejects_early_release_and_continuing_gripper_support(self):
        env, views, adapter = make_synthetic_adapter()
        support = support_report(env)
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()
        sample_adapter(
            env,
            views,
            0.125,
            root_pos=(0.58, -0.15, 0.85),
            held=True,
            finger_contacts=True,
        )
        adapter.observe()
        sample_adapter(
            env,
            views,
            0.25,
            root_pos=(0.58, -0.15, 0.85),
            held=True,
            finger_contacts=True,
        )
        adapter.observe()
        sample_adapter(
            env,
            views,
            0.375,
            held=False,
            cover=(support,),
            gripper_width=0.021,
        )
        early = adapter.observe()
        self.assertEqual(early, BoltSeatStatus(False, False))
        self.assertFalse(adapter._trace[-1].release_started)
        self.assertFalse(adapter._trace[-1].pickup_sequence_valid)

        env, views, adapter = make_synthetic_adapter()
        sample_adapter(env, views, 0.0, root_pos=(0.30, 0.25, 0.80))
        adapter.observe()
        sample_adapter(
            env,
            views,
            0.125,
            root_pos=(0.58, -0.15, 0.85),
            held=True,
            finger_contacts=True,
        )
        adapter.observe()
        for time_s in (0.25, 0.375, 0.5):
            sample_adapter(
                env, views, time_s, held=True, cover=(support_report(env),), finger_contacts=True
            )
            status = adapter.observe()
        self.assertTrue(status.seat_ready)
        sample_adapter(
            env,
            views,
            0.625,
            held=True,
            cover=(support_report(env),),
            robot=(grasp_report(env),),
            finger_contacts=True,
            gripper_width=0.05,
        )
        adapter.observe()
        self.assertTrue(adapter._trace[-1].release_started)
        self.assertTrue(adapter._trace[-1].pickup_sequence_valid)
        self.assertTrue(adapter._trace[-1].measurement.insertion_provenance_valid)
        for time_s in (0.75, 1.0, 1.25):
            sample_adapter(
                env,
                views,
                time_s,
                cover=(support_report(env),),
                robot=(grasp_report(env),),
            )
            status = adapter.observe()
        self.assertEqual(status, BoltSeatStatus(False, False))


if __name__ == "__main__":
    unittest.main()
