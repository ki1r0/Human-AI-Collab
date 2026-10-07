"""RoCo/Isaac Lab adapter for the M1 single-step task.

Isaac imports are intentionally kept in this module.  Importing ``hrc_m1`` on
the host therefore remains possible without an Isaac installation; the module
is loaded only by the simulator runner.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence


def _root() -> Path:
    return Path(os.environ.get("HRC_M1_ROOT", Path(__file__).resolve().parents[1]))


def make_env_classes() -> tuple[type, type]:
    """Create the config and environment classes inside Isaac Lab."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sim.spawners.materials import physics_materials_cfg, spawn_rigid_body_material
    from isaaclab.utils.math import subtract_frame_transforms
    from Galaxea_Lab_External.robots.robot_bundles import GALAXEA_R1_BUNDLE
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    root = _root()
    hub_path = str(root / "assets" / "parts" / "Hub Cover Output.usd")
    casing_path = str(root / "assets" / "parts" / "Casing Top.usd")
    blocked_scene = os.environ.get("HRC_M1_BLOCKED", "0") == "1"
    blocker_profile = os.environ.get("HRC_M1_BLOCKER_PROFILE", "default")

    def rigid_cfg(
        path: str,
        prim: str,
        *,
        pos: tuple[float, float, float],
        kinematic: bool,
        mass_kg: float,
        scale: float = 0.002,
        disable_gravity: bool | None = None,
    ) -> RigidObjectCfg:
        if disable_gravity is None:
            disable_gravity = kinematic
        return RigidObjectCfg(
            prim_path=prim,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=path,
                    scale=(scale, scale, scale),
                    # Keep mass explicit.  The source CAD volume is useful for
                    # calibration, but assigning the same mass to every USD
                    # (including the kinematic casing) hides physical errors.
                    # The Hub default remains the existing 5.7 kg estimate;
                    # the runner can override it for material calibration.
                    mass_props=sim_utils.MassPropertiesCfg(mass=float(mass_kg)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=disable_gravity,
                        kinematic_enabled=kinematic,
                        max_depenetration_velocity=0.2,
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=16,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=-0.00005),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        )

    @configclass
    class M1RocoEnvCfg(DirectRLEnvCfg):
        sim_dt = 0.01
        decimation = 5
        episode_length_s = 60.0
        action_space = 14
        observation_space = 0
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=sim_dt, render_interval=decimation)
        viewer = ViewerCfg(eye=(1.6, 1.4, 1.65), lookat=(0.55, 0.0, 0.95))
        scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
        robot_bundle = GALAXEA_R1_BUNDLE
        robot_cfg = GALAXEA_R1_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")
        # The canonical diagnostic reset is explicit so a clearance probe can
        # move the free Hub without writing its pose after the episode begins.
        # The default preserves the existing M1 fixture; the debug probe may
        # override this before scene construction.
        # The collision-on grasp regression is calibrated at y=0.45 m; the
        # earlier 0.40 m reset placed the jaw pair too close to the fixed
        # casing approach corridor.
        hub_reset_pos = (0.30, 0.45, 1.08)
        # The scatter reset is an independent full-physics fixture.  Its
        # actual Z values are derived from the USD bounds in
        # ``prepare_scatter_reset``; these fields only describe the support
        # surface and the desired XY layout.
        casing_reset_pos = (0.55, 0.0, 1.0)
        scatter_reset = False
        table_top_z = -0.05
        scatter_spawn_margin_m = 0.002
        table_size_xy = (1.5, 1.2)
        table_center_xy = (0.55, 0.0)
        scatter_support_top_z = -0.05
        scatter_hub_support_top_z = -0.05
        # A simple kinematic cuboid is sufficient as the M1 support fixture and
        # avoids importing RoCo's detailed desk mesh/collision cooking into the
        # seating measurement.
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 1.2, 0.10),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
            ),
            # The R1 base is authored with its bottom at z=0 and torso_link1
            # spanning z≈0.238..0.763 m.  A top surface at z=0.35 therefore
            # penetrates the base and torso at reset; the earlier comment
            # claiming that it sat below the torso was incorrect.  Keep the
            # table as a floor/support collision, but place its top at z=-0.05
            # so it cannot overlap the fixed robot base.  The calibrated
            # Casing remains a separate fixture at z=1.0.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, -0.10), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        # The strict physics rollout uses explicit kinematic support fixtures
        # rather than starting either CAD part in mid-air.  The four narrow
        # Hub pads support the annular material while leaving the bore open;
        # the Casing pad supports the fixture's authored bottom face.  They
        # are opt-in so the older diagnostic runs remain byte-for-byte
        # comparable, but the strict M1 runner enables them.
        _support_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, kinematic_enabled=True
        )
        _support_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0001, rest_offset=0.0
        )
        hub_support_n_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Hub_Staging_Support_N",
            spawn=sim_utils.CuboidCfg(
                size=(0.060, 0.020, 0.010),
                rigid_props=_support_rigid_props,
                collision_props=_support_collision_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.55, 1.060), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        hub_support_s_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Hub_Staging_Support_S",
            spawn=sim_utils.CuboidCfg(
                size=(0.060, 0.020, 0.010),
                rigid_props=_support_rigid_props,
                collision_props=_support_collision_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.35, 1.060), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        hub_support_e_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Hub_Staging_Support_E",
            spawn=sim_utils.CuboidCfg(
                size=(0.020, 0.060, 0.010),
                rigid_props=_support_rigid_props,
                collision_props=_support_collision_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, 0.45, 1.060), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        hub_support_w_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Hub_Staging_Support_W",
            spawn=sim_utils.CuboidCfg(
                size=(0.020, 0.060, 0.010),
                rigid_props=_support_rigid_props,
                collision_props=_support_collision_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.45, 1.060), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        casing_support_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Casing_Support",
            spawn=sim_utils.CuboidCfg(
                size=(0.50, 0.60, 0.020),
                rigid_props=_support_rigid_props,
                collision_props=_support_collision_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.934), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        # Optional real collision blocker for the REPAIR M0 blocked scenario.
        # It is absent from nominal runs.  The blocker is a separate fixed
        # cuboid in the target region; it is never a visual-only label or a
        # planner-visible scenario flag.
        if blocker_profile == "pin":
            # A small pin inside the socket bore is a removable obstruction,
            # but does not overlap the Hub's annular material.
            blocker_size = (0.010, 0.010, 0.050)
            blocker_pos = (0.55, 0.08683718, 1.145)
        elif blocker_profile == "arm_edge":
            # Keep the obstacle in the outside-finger corridor.  Its lower
            # edge is just outside the Hub annulus, so the robot can make a
            # real blocked descent without using the Hub itself as a bumper.
            blocker_size = (0.05, 0.025, 0.05)
            blocker_pos = (0.55, 0.238, 1.145)
        elif blocker_profile == "gentle":
            blocker_size = (0.16, 0.16, 0.02)
            blocker_pos = (0.55, 0.08683718, 1.13)
        else:
            blocker_size = (0.16, 0.16, 0.06)
            blocker_pos = (0.55, 0.08683718, 1.12)
        target_blocker_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/M1_Target_Blocker",
            spawn=sim_utils.CuboidCfg(
                size=blocker_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=blocker_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        )
        casing_cfg = rigid_cfg(
            casing_path,
            "/World/envs/env_.*/Casing_Top",
            pos=casing_reset_pos,
            kinematic=True,
            # Manifest estimate for the casing; it is kinematic in this M1
            # fixture, so this does not change the controller path.
            mass_kg=55.5,
        )
        # MagicAssembly's canonical M1 fit uses XYZ=(-90, 180, 0) degrees.
        # USD's XYZ composition maps to this wxyz quaternion:
        # (0, 0, +sqrt(1/2), +sqrt(1/2)).
        # Keeping the same orientation in reset and controller targets avoids
        # starting the free cover already intersecting the casing.
        hub_cfg = rigid_cfg(
            hub_path,
            "/World/envs/env_.*/Hub_Cover_Output_Top",
            # Keep the reset cover outside the casing footprint and above the
            # support table.  The earlier (0.40, 0.18, 1.03) reset overlapped
            # the kinematic casing before the calibration pose was written,
            # leaving a queued PhysX impulse that later ejected the cover.
            pos=hub_reset_pos,
            kinematic=False,
            # The legacy diagnostic keeps gravity disabled until its explicit
            # release boundary.  The strict runner changes this authored
            # property to False before scene construction and supplies the
            # staging pads above, so the same dynamic body is under gravity
            # for the entire rollout.
            disable_gravity=True,
            mass_kg=5.7,
        )
        hub_cfg.init_state.rot = (0.0, 0.0, 0.7071067812, 0.7071067812)
        head_camera_cfg = GALAXEA_R1_BUNDLE.head_camera_cfg
        left_hand_camera_cfg = GALAXEA_R1_BUNDLE.left_hand_camera_cfg
        right_hand_camera_cfg = GALAXEA_R1_BUNDLE.right_hand_camera_cfg
        overhead_camera_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/M1_Workcell_Overhead_Camera",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=100.0,
                horizontal_aperture=6.055,
                clipping_range=(0.01, 100.0),
            ),
            # Fixed world camera; ROS optical +Z points down after this
            # 180-degree X rotation. Center the casing in the top-down view.
            offset=CameraCfg.OffsetCfg(
                pos=(0.55, 0.08683718, 2.05),
                rot=(0.0, 1.0, 0.0, 0.0),
                convention="ros",
            ),
        )
        hub_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Hub_Cover_Output_Top",
            filter_prim_paths_expr=["/World/envs/env_.*/Casing_Top"],
            # The dense CAD pair can produce >200 contacts at the seat.  The
            # default four-point GPU buffer overflowed during calibration, so
            # allocate an explicit bounded buffer and retain distances for the
            # independent penetration check.
            track_contact_points=True,
            max_contact_data_count_per_prim=512,
        )
        # These are deliberately separate sensors because Isaac Lab only
        # applies filtered-contact reporting reliably when each sensor maps to
        # one primitive.  They are diagnostic/public instrumentation for the
        # inner-wall grasp; they do not imply a held verdict by themselves.
        left_gripper_link1_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/left_gripper_link1",
            filter_prim_paths_expr=["/World/envs/env_.*/Hub_Cover_Output_Top"],
            # Legacy runs use force-only reporting.  The strict runner opts in
            # to point tracking and records the actual radial contact points
            # needed to prove one-inside/one-outside topology.
            track_contact_points=False,
        )
        left_gripper_link2_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/left_gripper_link2",
            filter_prim_paths_expr=["/World/envs/env_.*/Hub_Cover_Output_Top"],
            track_contact_points=False,
        )
        right_gripper_link1_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_gripper_link1",
            filter_prim_paths_expr=["/World/envs/env_.*/Hub_Cover_Output_Top"],
            track_contact_points=False,
        )
        right_gripper_link2_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_gripper_link2",
            filter_prim_paths_expr=["/World/envs/env_.*/Hub_Cover_Output_Top"],
            track_contact_points=False,
        )
        left_arm_joint_pattern = GALAXEA_R1_BUNDLE.left_arm_joint_pattern
        right_arm_joint_pattern = GALAXEA_R1_BUNDLE.right_arm_joint_pattern
        left_gripper_dof_name = GALAXEA_R1_BUNDLE.left_gripper_dof_name
        right_gripper_dof_name = GALAXEA_R1_BUNDLE.right_gripper_dof_name
        torso_joint_pattern = GALAXEA_R1_BUNDLE.torso_joint_pattern
        initial_torso_pos = GALAXEA_R1_BUNDLE.initial_torso_pos
        # Diagnostic-only switch for isolating translational reachability. The
        # production M1 config leaves full pose IK enabled.
        ik_position_only = False
        # The R1 export has zero-width torso limits.  Zero-width limits are
        # numerically unstable in the Isaac 5.1 TGS solver, so keep the
        # RoCo reset posture inside a small bounded controller range until a
        # robot-specific limit calibration is available.
        torso_limit_half_range = 0.05
        # Isaac Sim 5.1 cannot reliably change the zero-width torso limits in
        # this R1 USD at runtime (PhysX lets the joints diverge).  The safe
        # default is therefore to keep the authored zero torso posture; a
        # future robot-specific USD/limit calibration may opt into the RoCo
        # nonzero posture explicitly.
        torso_runtime_override = False
        disable_fixture_collisions = False
        disable_table_collisions = False
        disable_casing_collisions = False
        filter_robot_casing_collisions = False
        filter_robot_casing_arm = "both"
        filter_robot_table_collisions = False
        recompute_pose_ik_each_step = True
        # Camera sensors must still be spawned when Isaac is launched with
        # ``--enable_cameras``, but long physics-only diagnostics can skip
        # their per-tick render/update work when no video is requested.
        update_cameras = True
        # A physics-only diagnostic can omit RTX camera prims entirely.  This
        # is distinct from ``update_cameras``: Isaac Sim refuses to initialize
        # a Camera unless the app was launched with ``--enable_cameras``.  The
        # public/video configuration keeps this enabled; short controller and
        # contact probes may disable it to measure dynamics without paying for
        # rendering or requiring an RTX sensor launch flag.
        spawn_cameras = True
        # Update RTX camera sensors sparsely for long diagnostic movies while
        # keeping the simulated controller and physics steps unchanged.
        camera_update_stride = 1
        # Optional physical grasp abstraction for the direct-control baseline.
        # The joint is authored disabled and can only be enabled by a runner
        # after filtered two-finger contact has been observed.  Nominal runs
        # leave this false; it is not part of the learned ACT interface.
        spawn_grasp_constraint = False
        spawn_physical_supports = False
        spawn_target_blocker = blocked_scene
        enable_right_contact_sensors = True
        # Contact material used only on the two authored R1 gripper links.
        # The strict physical rollout may raise this to model compliant/rubber
        # tips; it remains a real Coulomb friction coefficient, not an object
        # attachment or pose constraint.
        gripper_contact_static_friction = 1.5
        gripper_contact_dynamic_friction = 1.5
        # The CAD interface material is provisional.  Keep its default fixed
        # for the nominal baseline, but expose a run-scoped override for
        # sensitivity tests instead of silently retuning the physics.
        m1_contact_static_friction = 0.45
        m1_contact_dynamic_friction = 0.35
        # When enabled, 14-D actions are interpreted as absolute R1 joint
        # targets in environment order [L arm6, R arm6, L grip, R grip].
        # Direct waypoint diagnostics leave this disabled and use pose IK.
        action_control = False

    class M1RocoEnv(DirectRLEnv):
        cfg: M1RocoEnvCfg

        def __init__(self, cfg: M1RocoEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
            super().__init__(cfg, render_mode, **kwargs)
            self._active_arm = "left"
            self._pose_target: tuple[torch.Tensor, torch.Tensor] | None = None
            self._pose_joint_target: tuple[str, torch.Tensor] | None = None
            self._joint_target: tuple[torch.Tensor, torch.Tensor] | None = None
            self._torso_target: torch.Tensor | None = None
            self._gripper_target = 0.04
            self._gripper_targets: torch.Tensor | None = None
            self._last_obs: dict[str, Any] = {}
            self._camera_tick = 0
            # Public, calibrated grasp verification state.  The reference is
            # captured by the adapter immediately before the commanded lift;
            # no evaluator pose or post-reset object write is used.
            self._grasp_reference: dict[str, torch.Tensor] | None = None
            self.left_arm_cfg = SceneEntityCfg("robot", joint_names=["left_arm_joint.*"], body_names=["left_arm_link6"])
            self.right_arm_cfg = SceneEntityCfg("robot", joint_names=["right_arm_joint.*"], body_names=["right_arm_link6"])
            self.left_gripper_cfg = SceneEntityCfg("robot", joint_names=["left_gripper_axis1"])
            self.right_gripper_cfg = SceneEntityCfg("robot", joint_names=["right_gripper_axis1"])
            self.torso_cfg = SceneEntityCfg("robot", joint_names=[self.cfg.torso_joint_pattern])
            for entity in (self.left_arm_cfg, self.right_arm_cfg, self.left_gripper_cfg, self.right_gripper_cfg, self.torso_cfg):
                entity.resolve(self.scene)
            # DirectRLEnv does not restore an articulation's non-root joints
            # for a custom reset.  Keep the RoCo reset contract explicit: the
            # two arm chains and gripper DOFs start at the bundle's authored
            # default pose and hold that target until a policy action arrives.
            self._reset_joint_ids = (
                list(self.left_arm_cfg.joint_ids)
                + list(self.right_arm_cfg.joint_ids)
                + list(self.left_gripper_cfg.joint_ids)
                + list(self.right_gripper_cfg.joint_ids)
            )
            self.diff_ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="position" if self.cfg.ik_position_only else "pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                num_envs=self.scene.num_envs,
                device=self.device,
            )
            # Keep a position-only controller available for a narrowly scoped
            # final-seat diagnostic.  The normal lift/transport path remains
            # pose IK; a separate controller avoids changing that grasp
            # behavior merely to prevent a pose-DLS branch jump at the socket.
            self.position_diff_ik = (
                self.diff_ik
                if self.cfg.ik_position_only
                else DifferentialIKController(
                    DifferentialIKControllerCfg(
                        command_type="position",
                        use_relative_mode=False,
                        ik_method="dls",
                    ),
                    num_envs=self.scene.num_envs,
                    device=self.device,
                )
            )
            self._ik_position_only_override: bool | None = None

        def _setup_scene(self) -> None:
            self.robot = Articulation(self.cfg.robot_cfg)
            self.table = RigidObject(self.cfg.table_cfg)
            self.casing = RigidObject(self.cfg.casing_cfg)
            self.hub = RigidObject(self.cfg.hub_cfg)
            if self.cfg.spawn_physical_supports:
                self.hub_support_n = RigidObject(self.cfg.hub_support_n_cfg)
                self.hub_support_s = RigidObject(self.cfg.hub_support_s_cfg)
                self.hub_support_e = RigidObject(self.cfg.hub_support_e_cfg)
                self.hub_support_w = RigidObject(self.cfg.hub_support_w_cfg)
                self.casing_support = RigidObject(self.cfg.casing_support_cfg)
            else:
                self.hub_support_n = None
                self.hub_support_s = None
                self.hub_support_e = None
                self.hub_support_w = None
                self.casing_support = None
            if self.cfg.spawn_target_blocker:
                self.target_blocker = RigidObject(self.cfg.target_blocker_cfg)
            else:
                self.target_blocker = None
            if self.cfg.spawn_grasp_constraint:
                # A disabled FixedJoint is a physical grasp abstraction for a
                # control-policy calibration.  It is deliberately created in
                # the authored scene before PhysX starts; the runner fills in
                # the measured local frames and enables it only after contact.
                grasp_joint = UsdPhysics.FixedJoint.Define(
                    self.sim.get_initial_stage(), "/World/envs/env_0/M1_GraspConstraint"
                )
                grasp_joint.CreateBody0Rel().AddTarget(Sdf.Path("/World/envs/env_0/Robot/left_gripper_link1"))
                grasp_joint.CreateBody1Rel().AddTarget(Sdf.Path("/World/envs/env_0/Hub_Cover_Output_Top"))
                grasp_joint.CreateJointEnabledAttr(False)
                grasp_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                grasp_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                grasp_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                grasp_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                self.grasp_constraint = grasp_joint
            else:
                self.grasp_constraint = None
            self.hub_contact = ContactSensor(self.cfg.hub_contact_cfg)
            self.left_gripper_link1_contact = ContactSensor(self.cfg.left_gripper_link1_contact_cfg)
            self.left_gripper_link2_contact = ContactSensor(self.cfg.left_gripper_link2_contact_cfg)
            if self.cfg.enable_right_contact_sensors:
                self.right_gripper_link1_contact = ContactSensor(self.cfg.right_gripper_link1_contact_cfg)
                self.right_gripper_link2_contact = ContactSensor(self.cfg.right_gripper_link2_contact_cfg)
            else:
                self.right_gripper_link1_contact = None
                self.right_gripper_link2_contact = None
            if self.cfg.spawn_cameras:
                self.head_camera = Camera(self.cfg.head_camera_cfg)
                self.overhead_camera = Camera(self.cfg.overhead_camera_cfg)
                self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
                self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)
            else:
                self.head_camera = None
                self.overhead_camera = None
                self.left_hand_camera = None
                self.right_hand_camera = None
            # The table is the M1 support surface; omitting a second ground
            # plane avoids an extra collision layer under the CAD fixture.
            self._configure_collision_approximations()
            self.scene.clone_environments(copy_from_source=False)
            if (self.cfg.disable_fixture_collisions or self.cfg.disable_table_collisions
                    or self.cfg.disable_casing_collisions):
                # Calibration-only diagnostics: isolate one fixture at a time
                # without disabling Hub/robot contacts. These switches are
                # never enabled by the M1 task config or evaluated rollouts.
                stage = self.sim.get_initial_stage()
                roots = []
                if self.cfg.disable_fixture_collisions or self.cfg.disable_table_collisions:
                    roots.append("/World/envs/env_0/Table")
                if self.cfg.disable_fixture_collisions or self.cfg.disable_casing_collisions:
                    roots.append("/World/envs/env_0/Casing_Top")
                for prim_root in roots:
                    prim = stage.GetPrimAtPath(prim_root)
                    if prim:
                        for child in Usd.PrimRange(prim):
                            if child.HasAPI(UsdPhysics.CollisionAPI):
                                UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
            if self.cfg.filter_robot_casing_collisions:
                # Calibration-only pair filter: retain Casing↔Hub contacts,
                # but remove accidental Robot↔Casing contacts from the robot's
                # approach path. This is not enabled by nominal M1 config.
                # Use both explicit filtered pairs and USD collision groups:
                # imported articulated colliders may be represented as an
                # aggregate in PhysX, in which case a per-prim relation alone
                # is not sufficient to suppress the pair.
                stage = self.sim.get_initial_stage()
                robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
                casing_root = stage.GetPrimAtPath("/World/envs/env_0/Casing_Top")
                robot_colliders = [
                    prim for prim in Usd.PrimRange(robot_root)
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                if self.cfg.filter_robot_casing_arm in {"left", "right"}:
                    side_token = f"/{self.cfg.filter_robot_casing_arm}_"
                    robot_colliders = [
                        prim for prim in robot_colliders
                        if side_token in prim.GetPath().pathString
                    ]
                casing_colliders = [
                    prim for prim in Usd.PrimRange(casing_root)
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                for robot_prim in robot_colliders:
                    relation = UsdPhysics.FilteredPairsAPI.Apply(robot_prim).GetFilteredPairsRel()
                    for casing_prim in casing_colliders:
                        relation.AddTarget(casing_prim.GetPath())
                for casing_prim in casing_colliders:
                    relation = UsdPhysics.FilteredPairsAPI.Apply(casing_prim).GetFilteredPairsRel()
                    for robot_prim in robot_colliders:
                        relation.AddTarget(robot_prim.GetPath())
                robot_group_path = "/World/collisionGroups/HRC_M1_Robot"
                casing_group_path = "/World/collisionGroups/HRC_M1_Casing"
                robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)
                casing_group = UsdPhysics.CollisionGroup.Define(stage, casing_group_path)
                robot_collection = Usd.CollectionAPI.Apply(robot_group.GetPrim(), "colliders")
                casing_collection = Usd.CollectionAPI.Apply(casing_group.GetPrim(), "colliders")
                robot_includes = robot_collection.CreateIncludesRel()
                casing_includes = casing_collection.CreateIncludesRel()
                for robot_prim in robot_colliders:
                    robot_includes.AddTarget(robot_prim.GetPath())
                for casing_prim in casing_colliders:
                    casing_includes.AddTarget(casing_prim.GetPath())
                robot_group.CreateFilteredGroupsRel().AddTarget(casing_group.GetPath())
                casing_group.CreateFilteredGroupsRel().AddTarget(robot_group.GetPath())
                print(
                    f"[HRC_M1] diagnostic Robot-Casing filtered pairs: "
                    f"arm={self.cfg.filter_robot_casing_arm} "
                    f"robot_colliders={len(robot_colliders)} casing_colliders={len(casing_colliders)} "
                    f"collision_groups={robot_group_path},{casing_group_path}",
                    flush=True,
                )
            if self.cfg.filter_robot_table_collisions:
                # Calibration-only diagnostic for the analytic support table.
                # Keep Hub↔Casing and gripper↔Hub collision enabled while
                # removing Robot↔Table pairs.  The nominal M1 scene leaves
                # this disabled; if it resolves the failure, the table
                # height/extent or the authored R1 lower-body collider needs
                # correction rather than a task-level collision bypass.
                stage = self.sim.get_initial_stage()
                robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
                table_root = stage.GetPrimAtPath("/World/envs/env_0/Table")
                robot_colliders = [
                    prim for prim in Usd.PrimRange(robot_root)
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                table_colliders = [
                    prim for prim in Usd.PrimRange(table_root)
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                robot_group_path = "/World/collisionGroups/HRC_M1_RobotTable_Robot"
                table_group_path = "/World/collisionGroups/HRC_M1_RobotTable_Table"
                robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)
                table_group = UsdPhysics.CollisionGroup.Define(stage, table_group_path)
                robot_collection = Usd.CollectionAPI.Apply(robot_group.GetPrim(), "colliders")
                table_collection = Usd.CollectionAPI.Apply(table_group.GetPrim(), "colliders")
                robot_includes = robot_collection.CreateIncludesRel()
                table_includes = table_collection.CreateIncludesRel()
                for robot_prim in robot_colliders:
                    robot_includes.AddTarget(robot_prim.GetPath())
                for table_prim in table_colliders:
                    table_includes.AddTarget(table_prim.GetPath())
                robot_group.CreateFilteredGroupsRel().AddTarget(table_group.GetPath())
                table_group.CreateFilteredGroupsRel().AddTarget(robot_group.GetPath())
                print(
                    f"[HRC_M1] diagnostic Robot-Table filtered pairs: "
                    f"robot_colliders={len(robot_colliders)} table_colliders={len(table_colliders)} "
                    f"collision_groups={robot_group_path},{table_group_path}",
                    flush=True,
                )
            # Explicit non-bouncy steel/plastic contact material.  The imported
            # CAD files carry a visual default material but no trustworthy
            # solver material; leaving PhysX's default restitution made a
            # millimetre-scale seating contact eject the cover.
            m1_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
                static_friction=float(self.cfg.m1_contact_static_friction),
                dynamic_friction=float(self.cfg.m1_contact_dynamic_friction),
                restitution=0.0,
                friction_combine_mode="average",
            )
            spawn_rigid_body_material("/World/Materials/m1_contact", m1_mat_cfg)
            gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
                static_friction=float(self.cfg.gripper_contact_static_friction),
                dynamic_friction=float(self.cfg.gripper_contact_dynamic_friction),
                restitution=0.0,
                friction_combine_mode="average",
            )
            spawn_rigid_body_material("/World/Materials/m1_gripper_contact", gripper_mat_cfg)
            for env_idx in range(self.scene.num_envs):
                sim_utils.bind_physics_material(
                    f"/World/envs/env_{env_idx}/Hub_Cover_Output_Top/node_/mesh_",
                    "/World/Materials/m1_contact",
                )
                sim_utils.bind_physics_material(
                    f"/World/envs/env_{env_idx}/Casing_Top/node_/mesh_",
                    "/World/Materials/m1_contact",
                )
                for link_name in self.cfg.robot_bundle.gripper_collision_link_names:
                    sim_utils.bind_physics_material(
                        f"/World/envs/env_{env_idx}/Robot/{link_name}/collisions",
                        "/World/Materials/m1_gripper_contact",
                    )
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[])
            self.scene.articulations["robot"] = self.robot
            if self.target_blocker is not None:
                self.scene.rigid_objects["target_blocker"] = self.target_blocker
            self.scene.sensors["hub_contact"] = self.hub_contact
            self.scene.sensors["left_gripper_link1_contact"] = self.left_gripper_link1_contact
            self.scene.sensors["left_gripper_link2_contact"] = self.left_gripper_link2_contact
            if self.cfg.enable_right_contact_sensors:
                self.scene.sensors["right_gripper_link1_contact"] = self.right_gripper_link1_contact
                self.scene.sensors["right_gripper_link2_contact"] = self.right_gripper_link2_contact
            sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)))

        def _configure_collision_approximations(self) -> None:
            """Apply task-scoped collider approximations before physics starts."""
            stage = self.sim.get_initial_stage()
            counts = {"sdf": 0, "none": 0}
            # The table is an analytic CuboidCfg, so it already has a simple
            # solver-friendly collider.  Applying MeshCollisionAPI to the
            # generated cube mesh is invalid in Isaac Sim 5.1 (the generated
            # mesh has an empty physics:approximation typeName).  Keep the
            # authored CAD collision approximation unchanged until a
            # collision-specific calibration freezes a replacement; silently
            # swapping in a coarse hull would destroy the socket interface.
            # The legacy Casing is kinematic and can retain its authored
            # triangle mesh, preserving the socket opening.  In the scatter
            # fixture the Casing is intentionally dynamic, so PhysX rejects a
            # raw triangle mesh and silently falls back to a convex hull (which
            # seals the socket).  Use an explicit SDF there instead; it is a
            # supported dynamic approximation and keeps the concave interface
            # visible to the solver.
            casing_approximation = "sdf" if self.cfg.scatter_reset else "none"
            for root, approximation in ((self.hub, "sdf"), (self.casing, casing_approximation)):
                prim = stage.GetPrimAtPath(root.cfg.prim_path.replace(".*", "0"))
                if not prim:
                    continue
                for child in Usd.PrimRange(prim):
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Set(approximation)
                    if approximation == "sdf":
                        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(child)
                        sdf.GetSdfResolutionAttr().Set(256)
                        # The authored Hub mesh has a 28 mm thin axis.  Do not
                        # let PhysX's default SDF margin inflate that thin
                        # cover during grasp/seat calibration; the explicit
                        # contact/rest offsets above remain the only solver
                        # tolerance.  This is a scene-calibration setting, not
                        # a policy-time collision bypass.
                        sdf.GetSdfMarginAttr().Set(0.0)
                        sdf.GetSdfNarrowBandThicknessAttr().Set(0.0)
                    counts[approximation] += 1

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self._action = actions.to(self.device)
            # DirectRLEnv applies one action for ``decimation`` simulator
            # ticks.  Compute differential IK once per environment step and
            # hold that joint target across the substeps; recomputing from a
            # lagging articulated state at every tick over-stepped the R1
            # high-PD joints and made an otherwise reachable waypoint diverge.
            if (self._pose_joint_target is None or self.cfg.recompute_pose_ik_each_step) and self._pose_target is not None:
                arm_cfg = self.left_arm_cfg if self._active_arm == "left" else self.right_arm_cfg
                self._pose_joint_target = (self._active_arm, self._ik_target(self._pose_target[0], self._pose_target[1], arm_cfg))

        def _ik_target(self, target_position: torch.Tensor, target_orientation: torch.Tensor, arm_cfg: SceneEntityCfg) -> torch.Tensor:
            robot = self.robot
            body_id = arm_cfg.body_ids[0]
            jacobian_id = body_id - 1 if robot.is_fixed_base else body_id
            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_id, :, arm_cfg.joint_ids]
            ee_pose = robot.data.body_state_w[:, body_id, :7]
            root_pose = robot.data.root_state_w[:, :7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pose[:, :3], root_pose[:, 3:7], ee_pose[:, :3], ee_pose[:, 3:7])
            # DifferentialIKController consumes a pose in the articulation
            # root frame.  The task adapter stores waypoints in world metres;
            # converting here avoids silently mixing world and base frames on
            # the mobile R1 platform.
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], target_position, target_orientation
            )
            position_only = (
                self._ik_position_only_override
                if self._ik_position_only_override is not None
                else self.cfg.ik_position_only
            )
            controller = self.position_diff_ik if position_only else self.diff_ik
            if position_only:
                controller.set_command(target_pos_b, ee_quat=ee_quat_b)
            else:
                controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
            return controller.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, arm_cfg.joint_ids])

        def set_ik_position_only_override(self, enabled: bool | None) -> None:
            """Temporarily select position-only DLS for a local waypoint."""
            if enabled is not None and not isinstance(enabled, bool):
                raise TypeError("enabled must be bool or None")
            self._ik_position_only_override = enabled

        def _apply_action(self) -> None:
            if self.cfg.action_control and self._action.shape[-1] >= 14:
                # RoCo ACT's wrapper maps its policy order
                # [L6,Lgrip,R6,Rgrip] to this environment order before
                # calling env.step. Keep learned actions absolute and do not
                # silently clip arm targets; the runner performs the same
                # gross-range safety check as the official adapter.
                self._joint_target = (
                    self._action[:, :6],
                    self._action[:, 6:12],
                )
                self._gripper_targets = self._action[:, 12:14]
                self.robot.set_joint_position_target(self._joint_target[0], joint_ids=self.left_arm_cfg.joint_ids)
                self.robot.set_joint_position_target(self._joint_target[1], joint_ids=self.right_arm_cfg.joint_ids)
            elif self._pose_joint_target is not None:
                active_arm, arm_target = self._pose_joint_target
                arm_cfg = self.left_arm_cfg if active_arm == "left" else self.right_arm_cfg
                self.robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
            elif self._action.numel():
                # A zero action is a hold command for this adapter.  Sending
                # literal all-zero joint targets made the R1 arms sweep across
                # the fixture during a passive physics trial and could hit the
                # cover before it reached the socket.  Non-zero action vectors
                # still retain the simple joint-target path for diagnostics.
                if self._joint_target is None:
                    self._joint_target = (
                        self.robot.data.joint_pos[:, self.left_arm_cfg.joint_ids].clone(),
                        self.robot.data.joint_pos[:, self.right_arm_cfg.joint_ids].clone(),
                    )
                if bool(torch.any(torch.abs(self._action[:, :12]) > 1.0e-8).item()):
                    self._joint_target = (self._action[:, :6], self._action[:, 6:12])
                self.robot.set_joint_position_target(self._joint_target[0], joint_ids=self.left_arm_cfg.joint_ids)
                self.robot.set_joint_position_target(self._joint_target[1], joint_ids=self.right_arm_cfg.joint_ids)
            # The R1 USD's torso limits are widened at reset and its target
            # must be held on every physics tick.  Leaving these DOFs out of
            # the custom action path lets the authored zero target pull the
            # torso down, moving the arm links even while their arm joints
            # appear stationary.
            if self._torso_target is not None:
                self.robot.set_joint_position_target(self._torso_target, joint_ids=self.torso_cfg.joint_ids)
            # An explicit dual-jaw target is also used by the scripted
            # physics controller during a hybrid handoff.  It must not be
            # gated on learned ``action_control`` mode; otherwise the right
            # arm becomes active and the left load-bearing jaw silently loses
            # its close command.
            if self._gripper_targets is not None:
                self.robot.set_joint_position_target(self._gripper_targets[:, 0:1], joint_ids=self.left_gripper_cfg.joint_ids)
                self.robot.set_joint_position_target(self._gripper_targets[:, 1:2], joint_ids=self.right_gripper_cfg.joint_ids)
            else:
                grip_cfg = self.left_gripper_cfg if self._active_arm == "left" else self.right_gripper_cfg
                self.robot.set_joint_position_target(torch.full((self.scene.num_envs, 1), float(self._gripper_target), device=self.device), joint_ids=grip_cfg.joint_ids)
            dt = self.sim.get_physics_dt()
            objects = [self.table, self.casing, self.hub]
            if self.cfg.spawn_physical_supports:
                objects.extend(
                    [self.hub_support_n, self.hub_support_s, self.hub_support_e,
                     self.hub_support_w, self.casing_support]
                )
            for obj in objects:
                obj.update(dt)
            self._camera_tick += 1
            if (self.cfg.update_cameras and self.cfg.spawn_cameras
                    and self._camera_tick % max(1, int(self.cfg.camera_update_stride)) == 0):
                for camera in (
                    self.head_camera, self.overhead_camera,
                    self.left_hand_camera, self.right_hand_camera,
                ):
                    camera.update(dt)

        def _get_observations(self) -> dict[str, Any]:
            # Camera buffers are updated by Isaac Lab during the render step.
            def out(camera: Any, kind: str) -> Any:
                return camera.data.output.get(kind) if camera is not None else None
            obs = {
                "head_rgb": out(self.head_camera, "rgb"), "left_hand_rgb": out(self.left_hand_camera, "rgb"), "right_hand_rgb": out(self.right_hand_camera, "rgb"),
                "overhead_rgb": out(self.overhead_camera, "rgb"),
                "head_depth": out(self.head_camera, "distance_to_image_plane"), "left_hand_depth": out(self.left_hand_camera, "distance_to_image_plane"), "right_hand_depth": out(self.right_hand_camera, "distance_to_image_plane"),
                "hub_contact_force": self.hub_contact.data.net_forces_w,
                "hub_contact_force_matrix": self.hub_contact.data.force_matrix_w,
                "hub_contact_points": self.hub_contact.data.contact_pos_w,
                "left_gripper_link1_contact_force": self.left_gripper_link1_contact.data.net_forces_w,
                "left_gripper_link1_contact_force_matrix": self.left_gripper_link1_contact.data.force_matrix_w,
                "left_gripper_link2_contact_force": self.left_gripper_link2_contact.data.net_forces_w,
                "left_gripper_link2_contact_force_matrix": self.left_gripper_link2_contact.data.force_matrix_w,
                "left_arm_joint_pos": self.robot.data.joint_pos[:, self.left_arm_cfg.joint_ids], "right_arm_joint_pos": self.robot.data.joint_pos[:, self.right_arm_cfg.joint_ids],
                "left_gripper_joint_pos": self.robot.data.joint_pos[:, self.left_gripper_cfg.joint_ids], "right_gripper_joint_pos": self.robot.data.joint_pos[:, self.right_gripper_cfg.joint_ids],
                "left_arm_joint_vel": self.robot.data.joint_vel[:, self.left_arm_cfg.joint_ids], "right_arm_joint_vel": self.robot.data.joint_vel[:, self.right_arm_cfg.joint_ids],
            }
            self._last_obs = obs
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            return torch.zeros(self.scene.num_envs, device=self.device)

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            terminated = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
            timeout = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, timeout

        def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            super()._reset_idx(env_ids)
            # Initial placement is reset-only scene construction; no episode
            # action path writes poses.
            hub_state = self.hub.data.default_root_state.clone()
            self.hub.write_root_state_to_sim(hub_state)
            casing_state = self.casing.data.default_root_state.clone()
            self.casing.write_root_state_to_sim(casing_state)
            if self.cfg.spawn_physical_supports:
                # Support fixtures are kinematic, but reassert their authored
                # reset transforms so a repeated episode cannot inherit a
                # previous diagnostic's state.
                for support in (
                    self.hub_support_n, self.hub_support_s, self.hub_support_e,
                    self.hub_support_w, self.casing_support,
                ):
                    support.write_root_state_to_sim(support.data.default_root_state.clone())
            joint_ids = self._reset_joint_ids
            joint_values = self.robot.data.default_joint_pos[env_ids][:, joint_ids].clone()
            self.robot.write_joint_position_to_sim(joint_values, joint_ids, env_ids)
            self.robot.write_joint_velocity_to_sim(torch.zeros_like(joint_values), joint_ids, env_ids)
            self.robot.set_joint_position_target(joint_values, joint_ids, env_ids)
            # The pinned RoCo R1 bundle requires this torso posture for the
            # arm workspace.  Its USD has torso limits authored as [0, 0], so
            # widen each limit to the bundle's known initial value before
            # writing both the simulated joint state and its hold target.
            if self.cfg.torso_runtime_override:
                torso_values = torch.tensor(self.cfg.initial_torso_pos, device=self.device, dtype=torch.float32)
            else:
                torso_values = self.robot.data.default_joint_pos[env_ids][:, self.torso_cfg.joint_ids][0].clone()
            torso_values = torso_values.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.write_joint_position_to_sim(torso_values, self.torso_cfg.joint_ids, env_ids)
            self.robot.write_joint_velocity_to_sim(torch.zeros_like(torso_values), self.torso_cfg.joint_ids, env_ids)
            self.robot.set_joint_position_target(torso_values, joint_ids=self.torso_cfg.joint_ids, env_ids=env_ids)
            if self.cfg.torso_runtime_override:
                for local_idx, joint_id in enumerate(self.torso_cfg.joint_ids):
                    value = float(self.cfg.initial_torso_pos[local_idx])
                    self.robot.write_joint_position_limit_to_sim(
                        torch.tensor([value - self.cfg.torso_limit_half_range, value + self.cfg.torso_limit_half_range], device=self.device), [joint_id], env_ids
                    )
            # Re-assert the target after changing limits; Isaac Lab may clamp
            # the articulation's default/target buffers while applying the
            # new bounds.
            self._torso_target = torso_values.clone()
            self.robot.set_joint_position_target(self._torso_target, joint_ids=self.torso_cfg.joint_ids, env_ids=env_ids)
            self._pose_target = None
            self._pose_joint_target = None
            self._joint_target = None
            self._gripper_target = 0.04
            self._gripper_targets = None
            self._grasp_reference = None
            self._camera_tick = 0
            self._ik_position_only_override = None

        def set_pose_target(self, arm: str, position: torch.Tensor, orientation: torch.Tensor) -> None:
            if arm not in {"left", "right"}:
                raise ValueError("arm must be left or right")
            self._active_arm = arm
            self._pose_target = (position.to(self.device), orientation.to(self.device))
            self._pose_joint_target = None

        def clear_pose_target(self) -> None:
            self._pose_target = None
            self._pose_joint_target = None

        def set_gripper(self, opening: float) -> None:
            self._gripper_target = float(opening)
            self._gripper_targets = None

        def set_dual_gripper_targets(self, left_opening: float, right_opening: float) -> None:
            """Hold explicit left/right jaw targets on every physics tick.

            The single-target helper follows ``_active_arm``.  During a
            hybrid dual-arm handoff that is insufficient: opening the right
            jaw changes the active arm, while the load-bearing left jaw then
            stops receiving an explicit close target and can yield under the
            dynamic Hub's weight.  This helper still sends ordinary actuator
            targets (it does not add a constraint or modify the Hub state),
            but keeps both jaw commands observable and independent until the
            handoff is complete.
            """
            self._gripper_targets = torch.tensor(
                [[float(left_opening), float(right_opening)]],
                device=self.device,
                dtype=torch.float32,
            ).expand(self.scene.num_envs, -1).clone()
            self._gripper_target = float(left_opening)

        def root_states(self) -> dict[str, torch.Tensor]:
            return {"hub": self.hub.data.root_state_w.clone(), "casing": self.casing.data.root_state_w.clone()}

        def prepare_scatter_reset(self) -> dict[str, Any]:
            """Place both CAD parts on the table before the first action.

            This is reset-time scene initialization, not a controller action:
            no part pose is written after ``env.reset`` returns.  The root Z
            comes from each authored USD world bound, so a mesh extent change
            cannot silently reintroduce a hovering initial state.
            """
            if not self.cfg.scatter_reset:
                return {"enabled": False}
            stage = self.sim.get_initial_stage()
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                useExtentsHint=True,
            )

            def bounds(path: str) -> tuple[list[float], list[float]]:
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise RuntimeError(f"scatter reset cannot find USD prim: {path}")
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                lo, hi = box.GetMin(), box.GetMax()
                return [float(lo[i]) for i in range(3)], [float(hi[i]) for i in range(3)]

            table_pos = self.table.data.default_root_state[0, :3].detach().cpu()
            table_size = tuple(float(v) for v in self.cfg.table_cfg.spawn.size)
            table_min = [float(table_pos[0]) - table_size[0] / 2.0,
                         float(table_pos[1]) - table_size[1] / 2.0,
                         float(self.cfg.table_top_z) - table_size[2]]
            table_max = [float(table_pos[0]) + table_size[0] / 2.0,
                         float(table_pos[1]) + table_size[1] / 2.0,
                         float(self.cfg.table_top_z)]

            # In the scatter fixture the tabletop is intentionally below the
            # robot's finger path.  The small, kinematic support bodies are
            # the physical staging surface for the two dynamic CAD bodies;
            # they are not pose writes or a robot/table collision filter.
            casing_support_surface_z = float(
                self.cfg.scatter_support_top_z if self.cfg.spawn_physical_supports else self.cfg.table_top_z
            )
            hub_support_surface_z = float(
                self.cfg.scatter_hub_support_top_z if self.cfg.spawn_physical_supports else self.cfg.table_top_z
            )
            specs = (
                ("casing", self.casing, "/World/envs/env_0/Casing_Top", self.cfg.casing_reset_pos[:2], casing_support_surface_z),
                ("hub", self.hub, "/World/envs/env_0/Hub_Cover_Output_Top", self.cfg.hub_reset_pos[:2], hub_support_surface_z),
            )
            report: dict[str, Any] = {
                "enabled": True,
                "full_physics_required": True,
                "table_top_z_m": float(self.cfg.table_top_z),
                "support_surfaces_z_m": {"casing": casing_support_surface_z, "hub": hub_support_surface_z},
                "physical_supports": bool(self.cfg.spawn_physical_supports),
                "spawn_margin_m": float(self.cfg.scatter_spawn_margin_m),
                "table_aabb_m": {"min": table_min, "max": table_max},
                "objects": {},
            }
            for name, obj, path, xy, support_surface_z in specs:
                old_min, old_max = bounds(path)
                old_root = obj.data.default_root_state[0, :3].detach().cpu()
                z_offset = float(old_root[2]) - old_min[2]
                desired = (
                    float(xy[0]),
                    float(xy[1]),
                    support_surface_z + float(self.cfg.scatter_spawn_margin_m) + z_offset,
                )
                state = obj.data.default_root_state.clone()
                state[:, 0] = desired[0]
                state[:, 1] = desired[1]
                state[:, 2] = desired[2]
                state[:, 7:13] = 0.0
                # AssetData exposes default_root_state as a mutable tensor;
                # updating it makes DirectRLEnv's reset reassert the same
                # measured support pose for every episode.
                obj.data.default_root_state[:] = state
                obj.write_root_state_to_sim(state)
                delta = [desired[i] - float(old_root[i]) for i in range(3)]
                new_min = [old_min[i] + delta[i] for i in range(3)]
                new_max = [old_max[i] + delta[i] for i in range(3)]
                report["objects"][name] = {
                    "root_position_m": list(desired),
                    "source_root_position_m": [float(v) for v in old_root.tolist()],
                    "source_aabb_m": {"min": old_min, "max": old_max},
                    "predicted_aabb_after_reset_m": {"min": new_min, "max": new_max},
                    "bottom_to_table_top_m": float(new_min[2] - self.cfg.table_top_z),
                    "bottom_to_support_surface_m": float(new_min[2] - support_surface_z),
                    "xy_inside_table": bool(
                        new_min[0] >= table_min[0] and new_max[0] <= table_max[0]
                        and new_min[1] >= table_min[1] and new_max[1] <= table_max[1]
                    ),
                }
            if self.cfg.spawn_physical_supports:
                # Re-anchor the support fixtures to the measured reset XY and
                # USD-derived object bottoms.  This makes a repeated reset
                # deterministic even if a CAD asset's authored extent is
                # revised, while keeping all support contacts in PhysX.
                hub_x, hub_y = self.cfg.hub_reset_pos[:2]
                table_top_z = float(self.cfg.table_top_z)
                hub_support_top_z = float(self.cfg.scatter_hub_support_top_z) + float(self.cfg.scatter_spawn_margin_m)
                hub_height = hub_support_top_z - table_top_z
                support_z = table_top_z + hub_height / 2.0
                support_specs = (
                    (self.hub_support_n, (float(hub_x), float(hub_y) + 0.10, support_z)),
                    (self.hub_support_s, (float(hub_x), float(hub_y) - 0.10, support_z)),
                    (self.hub_support_e, (float(hub_x) + 0.10, float(hub_y), support_z)),
                    (self.hub_support_w, (float(hub_x) - 0.10, float(hub_y), support_z)),
                )
                support_report = []
                for support, pos in support_specs:
                    state = support.data.default_root_state.clone()
                    state[:, :3] = torch.tensor(pos, device=state.device, dtype=state.dtype)
                    state[:, 7:13] = 0.0
                    support.data.default_root_state[:] = state
                    support.write_root_state_to_sim(state)
                    support_report.append({
                        "prim_path": str(support.cfg.prim_path),
                        "root_position_m": list(pos),
                        "bottom_z_m": table_top_z,
                        "top_z_m": hub_support_top_z,
                        "size_m": [float(v) for v in support.cfg.spawn.size],
                    })
                casing_x, casing_y = self.cfg.casing_reset_pos[:2]
                casing_support_top_z = float(self.cfg.scatter_support_top_z) + float(self.cfg.scatter_spawn_margin_m)
                casing_height = casing_support_top_z - table_top_z
                casing_support_z = table_top_z + casing_height / 2.0
                state = self.casing_support.data.default_root_state.clone()
                state[:, :3] = torch.tensor(
                    (float(casing_x), float(casing_y), casing_support_z),
                    device=state.device,
                    dtype=state.dtype,
                )
                state[:, 7:13] = 0.0
                self.casing_support.data.default_root_state[:] = state
                self.casing_support.write_root_state_to_sim(state)
                support_report.append({
                    "prim_path": str(self.casing_support.cfg.prim_path),
                    "root_position_m": [float(casing_x), float(casing_y), float(casing_support_z)],
                    "bottom_z_m": table_top_z,
                    "top_z_m": casing_support_top_z,
                    "size_m": [float(v) for v in self.casing_support.cfg.spawn.size],
                })
                report["supports"] = support_report
            report["casing_dynamic"] = bool(
                not self.cfg.casing_cfg.spawn.rigid_props.kinematic_enabled
                and not self.cfg.casing_cfg.spawn.rigid_props.disable_gravity
            )
            report["hub_dynamic"] = bool(
                not self.cfg.hub_cfg.spawn.rigid_props.kinematic_enabled
                and not self.cfg.hub_cfg.spawn.rigid_props.disable_gravity
            )
            self._scatter_reset_report = report
            return report

        def scatter_reset_report(self) -> dict[str, Any]:
            return dict(getattr(self, "_scatter_reset_report", {"enabled": False}))

        def retract_hub_staging_supports(self, *, drop_m: float = 0.30, mode: str = "down") -> None:
            """Withdraw kinematic staging pads after a real grasp is closed.

            The pads are setup fixtures, not a grasp constraint.  Moving only
            these environment bodies lets a held dynamic Hub clear the staging
            surface before the robot performs the placement descent.  The
            lateral mode moves each pad radially away from the Hub at the same
            height.  ``disable_collision`` is a diagnostic ablation that
            removes only the pad collision APIs, avoiding any kinematic motion
            impulse while retaining the dynamic Hub and gripper contacts.
            """
            if not self.cfg.spawn_physical_supports:
                return
            supports = (
                (self.hub_support_n, (0.0, 1.0)),
                (self.hub_support_s, (0.0, -1.0)),
                (self.hub_support_e, (1.0, 0.0)),
                (self.hub_support_w, (-1.0, 0.0)),
            )
            for support, direction in supports:
                state = support.data.root_state_w.clone()
                if mode == "disable_collision":
                    from pxr import Usd, UsdPhysics

                    stage = self.sim.get_initial_stage()
                    root = stage.GetPrimAtPath(str(support.cfg.prim_path).replace(".*", "0"))
                    for prim in Usd.PrimRange(root):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                    continue
                if mode == "lateral":
                    state[:, 0] += float(drop_m) * direction[0]
                    state[:, 1] += float(drop_m) * direction[1]
                elif mode == "down":
                    state[:, 2] -= float(drop_m)
                else:
                    raise ValueError(f"unsupported staging-support withdrawal mode: {mode}")
                support.write_root_state_to_sim(state)

        def clear_target_blocker(self) -> bool:
            """Move the registered helper blocker out of the goal region.

            This is the only scene mutation reserved for the M0 helper.  It
            does not touch the Hub, Casing, robot, materials, or controller
            state.  The caller must establish a robot safe-hold first.
            """
            blocker = getattr(self, "target_blocker", None)
            if blocker is None:
                return False
            state = blocker.data.root_state_w.clone()
            state[:, :3] += torch.tensor([0.50, 0.50, 0.20], device=self.device)
            blocker.write_root_state_to_sim(state)
            self.sim.forward()
            return True

        def begin_grasp_verification(self) -> None:
            """Start the public object-follow check immediately before lift."""
            body_id = self.left_arm_cfg.body_ids[0]
            self._grasp_reference = {
                "hub": self.hub.data.root_state_w[0, :3].clone(),
                "link6": self.robot.data.body_state_w[0, body_id, :3].clone(),
            }

        @staticmethod
        def _sensor_force_norm(sensor: Any) -> float:
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if matrix is None or not matrix.numel():
                return 0.0
            return float(torch.linalg.vector_norm(matrix).detach().cpu().item())

        def grasp_verification(self) -> tuple[str, str]:
            """Return a public held verdict from filtered contacts + object follow.

            This intentionally exposes only the categorical status and a
            short reason to the action adapter.  The evaluator still retains
            the raw force/pose trace privately; a reaching TCP or commanded
            gripper value alone cannot produce HELD_CONFIRMED.
            """
            if self._grasp_reference is None:
                return "UNKNOWN", "grasp_reference_not_started"
            body_id = self.left_arm_cfg.body_ids[0]
            hub_delta = torch.linalg.vector_norm(
                self.hub.data.root_state_w[0, :3] - self._grasp_reference["hub"]
            )
            link6_delta = torch.linalg.vector_norm(
                self.robot.data.body_state_w[0, body_id, :3] - self._grasp_reference["link6"]
            )
            force1 = self._sensor_force_norm(self.left_gripper_link1_contact)
            force2 = self._sensor_force_norm(self.left_gripper_link2_contact)
            both_contact = force1 > 1.0e-3 and force2 > 1.0e-3
            follows = float(hub_delta.item()) >= 0.5 * max(float(link6_delta.item()), 1.0e-6)
            if both_contact and follows:
                return "HELD_CONFIRMED", (
                    f"filtered_pair_contact_and_object_follow;forces_N={force1:.3f},{force2:.3f};"
                    f"hub_follow_m={float(hub_delta.item()):.6f};link6_m={float(link6_delta.item()):.6f}"
                )
            if not both_contact:
                return "HELD_FAILED", f"missing_filtered_pair_contact;forces_N={force1:.3f},{force2:.3f}"
            return "HELD_FAILED", (
                f"contact_without_object_follow;hub_follow_m={float(hub_delta.item()):.6f};"
                f"link6_m={float(link6_delta.item()):.6f}"
            )

        def enable_grasp_constraint(self) -> bool:
            """Enable the optional measured-frame physical grasp constraint.

            A runner may call this only after filtered contact on both fingers
            has been observed. The local frames are computed from current
            simulated poses, so enabling the joint does not teleport either
            rigid body. Nominal contact-only runs leave this feature disabled.
            """
            if self.grasp_constraint is None:
                return False
            from pxr import Gf  # noqa: E402

            names = list(self.robot.body_names)
            link_id = names.index("left_gripper_link1")
            body = self.robot.data.body_state_w[0, link_id, :7]
            hub = self.hub.data.root_state_w[0, :7]

            def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                w, x, y, z = a
                W, X, Y, Z = b
                return torch.stack((w * W - x * X - y * Y - z * Z,
                                    w * X + x * W + y * Z - z * Y,
                                    w * Y - x * Z + y * W + z * X,
                                    w * Z + x * Y - y * X + z * W))

            def qinv(q: torch.Tensor) -> torch.Tensor:
                return torch.stack((q[0], -q[1], -q[2], -q[3])) / torch.dot(q, q).clamp_min(1.0e-8)

            def qrotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                pure = torch.stack((torch.zeros((), device=v.device), v[0], v[1], v[2]))
                return qmul(qmul(q, pure), qinv(q))[1:]

            body_q = body[3:7]
            hub_q = hub[3:7]
            anchor_pos = body[:3]
            local_pos1 = qrotate(qinv(hub_q), anchor_pos - hub[:3])
            local_q1 = qmul(qinv(hub_q), body_q)

            self.grasp_constraint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            self.grasp_constraint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            self.grasp_constraint.GetLocalPos1Attr().Set(
                Gf.Vec3f(*[float(v) for v in local_pos1.detach().cpu()])
            )
            self.grasp_constraint.GetLocalRot1Attr().Set(
                Gf.Quatf(float(local_q1[0]), Gf.Vec3f(float(local_q1[1]), float(local_q1[2]), float(local_q1[3])))
            )
            self.grasp_constraint.GetJointEnabledAttr().Set(True)
            self.sim.forward()
            return True

        def disable_grasp_constraint(self) -> None:
            if self.grasp_constraint is not None:
                self.grasp_constraint.GetJointEnabledAttr().Set(False)
                self.sim.forward()

    return M1RocoEnvCfg, M1RocoEnv


__all__ = ["make_env_classes"]
