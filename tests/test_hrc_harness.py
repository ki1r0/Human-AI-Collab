"""Harness invariants and a contract-only closed loop; no physical success claims."""

import copy
import json
import tempfile
import time
import unittest
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from hrc_harness.calibration import fit
from hrc_harness.contract_world import ContractWorld
from hrc_harness.contracts import Channel, MeasuredCost, Resources, ToolCall, ToolSpec, public_only
from hrc_harness.decision import Estimator, Selector, state_key
from hrc_harness.evidence import InsertionCheck, Ledger, features, monitor, verify_after_help
from hrc_harness.evaluation import judge, rotate
from hrc_harness.run import load_config, run_episode
from hrc_harness.proposer import validate_proposal
from hrc_harness.runtime import Interrupted, Runtime


ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/harness_contract.yaml")


def call(runtime, name, identifier=None):
    return ToolCall(identifier or f"call_{len(runtime.calls)}", name, runtime.epoch,
                    runtime.observation.observation_id)


def state(stalls=1):
    return {"stage": "prealigned", "holding": "yes", "progress_band": "low",
            "force_band": "middle", "tracking_band": "low", "consecutive_stalls": stalls,
            "probe_history": []}


def estimate(p, cost, outcomes=None):
    return {"success_probability": p, "support_count": 10,
            "cost": {"wall_s": cost, "contact_exposure_Ns": 0, "helper_effort": 0, "model_money": 0},
            "outcomes": outcomes or {}}


class HarnessTests(unittest.TestCase):
    def test_private_release_requires_both_grippers_to_release(self):
        import numpy as np
        from unittest.mock import patch
        from hrc_harness.evaluation import record_isaac_sample
        class Tensor(np.ndarray):
            def abs(self):
                return np.abs(self)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numel(self):
                return self.size
        joints = np.array([[.04, .04]]).view(Tensor)
        root = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.]], dtype=float).view(Tensor)
        contact = lambda: SimpleNamespace(data=SimpleNamespace(force_matrix_w=np.zeros((1, 3)).view(Tensor)))
        env = SimpleNamespace(root_states=lambda: {"hub": root, "casing": root},
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=joints)), hub_contact=contact(),
            left_gripper_cfg=SimpleNamespace(joint_ids=[0]), right_gripper_cfg=SimpleNamespace(joint_ids=[1]),
            cfg=SimpleNamespace(sim_dt=.01))
        for side in ("left", "right"):
            for index in (1, 2):
                setattr(env, f"{side}_gripper_link{index}_contact", contact())
        task = {"sensors": {"released_opening_threshold_m": .035}}
        torch = SimpleNamespace(linalg=SimpleNamespace(vector_norm=np.linalg.norm))
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertTrue(record_isaac_sample(env, task)["released"])
            joints[0, 1] = .02
            self.assertFalse(record_isaac_sample(env, task)["released"])
            joints[0, 1] = .04
            env.right_gripper_link2_contact.data.force_matrix_w[0, 0] = 1
            self.assertFalse(record_isaac_sample(env, task)["released"])

    def test_dual_waypoint_commands_both_arms_and_guards_right_tracking(self):
        import numpy as np
        from hrc_harness.isaac import IsaacSceneAdapter
        class Tensor(np.ndarray):
            def detach(self):
                return self
            def cpu(self):
                return self
            def clone(self):
                return self.copy()
        for jam in (False, True):
            bodies = np.array([[[0, 0, 0, 1, 0, 0, 0],
                                [0, -.17, 0, 1, 0, 0, 0]]], dtype=float).view(Tensor)
            targets, positions = [], []
            adapter = IsaacSceneAdapter.__new__(IsaacSceneAdapter)
            adapter.config = {"dual_arm_tcp_offset_m": [0, -.17, 0],
                "safety": {"tcp_speed_limit_mps": 1, "tracking_limit_m": .001,
                           "workspace_xyz_m": [[-1, 1]]*3}}
            adapter.task = {"frames": {"robot_tool_offset_m": [0, 0, 0]}}
            adapter.torch = SimpleNamespace(tensor=lambda values, **kwargs: values)
            def pose(position, quaternion):
                adapter.env._pose_target = (position, quaternion)
            def ik(position, quaternion, arm):
                targets.append(arm.body_ids[0])
                positions.append(position[0])
                return position[0]
            adapter.env = SimpleNamespace(device="cpu", cfg=SimpleNamespace(sim_dt=.1),
                robot=SimpleNamespace(data=SimpleNamespace(body_state_w=bodies,
                    joint_pos=np.zeros((1, 2)).view(Tensor))),
                left_arm_cfg=SimpleNamespace(body_ids=[0], joint_ids=[0]),
                right_arm_cfg=SimpleNamespace(body_ids=[1], joint_ids=[1]),
                _gripper_target=.02, set_pose_target=lambda arm, p, q: pose(p, q),
                clear_pose_target=lambda: None, _ik_target=ik,
                set_dual_gripper_targets=lambda left, right: None)
            adapter._tcp = lambda: (bodies[0, 0, :3].tolist(), [1, 0, 0, 0])
            def step():
                bodies[0, 0, :3] = adapter.env._joint_target[0]
                if not jam:
                    bodies[0, 1, :3] = adapter.env._joint_target[1]
            adapter.idle_step = step
            point = {"tcp_xyz_m": [.02, 0, 0], "quat_wxyz": [1, 0, 0, 0], "ticks": 2}
            if jam:
                with self.assertRaisesRegex(Interrupted, "right_tracking_limit"):
                    adapter._waypoint(point, lambda: None)
            else:
                adapter._waypoint(point, lambda: None)
                self.assertEqual(targets, [0, 1, 0, 1])
                self.assertEqual(bodies[0, 1, :3].tolist(), [.02, -.17, 0])
                bodies[0, :, 2] -= .01
                adapter._waypoint({**point, "tcp_xyz_m": [.02, 0, .02]}, lambda: None)
                self.assertAlmostEqual(positions[4][2], .01)
                self.assertAlmostEqual(positions[5][2], .01)
                adapter.safe_hold()
                self.assertIsNone(adapter.target)
                self.assertIsNone(adapter.target_quat)

    def test_contact_place_preserves_requested_target_and_prioritizes_force_stop(self):
        import numpy as np
        from hrc_harness.isaac import load_task
        from hrc_harness.pilot import run_pilot
        class Tensor(np.ndarray):
            def detach(self):
                return self
            def cpu(self):
                return self
            def abs(self):
                return np.abs(self)
        plan = load_task(ROOT / "configs/harness_pilot.yaml")
        plan["release_ramp_s"] = .25
        plan["dual_arm_tcp_offset_m"] = None
        task = load_task(ROOT / plan["task_config"])
        task["scene"].update(plan["scene"])
        for contact in (50., 130.):
            joints = np.zeros((1, 2)).view(Tensor)
            bodies = np.zeros((1, 4, 7)).view(Tensor)
            bodies[:, :, 3] = 1
            env = SimpleNamespace(cfg=SimpleNamespace(sim_dt=.1, spawn_cameras=False),
                robot=SimpleNamespace(find_bodies=lambda pattern: (range(4), [f"finger{i}" for i in range(4)]),
                    data=SimpleNamespace(joint_pos=joints, joint_vel=joints.copy(), body_state_w=bodies,
                                         joint_pos_target=joints.copy(), applied_torque=joints.copy())),
                left_gripper_cfg=SimpleNamespace(joint_ids=[0]), right_arm_cfg=SimpleNamespace(body_ids=[1]),
                _gripper_target=.02, scatter_reset_report=lambda: {})
            for side in ("left", "right"):
                for index in (1, 2):
                    setattr(env, f"{side}_gripper_link{index}_contact", SimpleNamespace(
                        data=SimpleNamespace(net_forces_w=np.zeros((1, 3)))))
            adapter = SimpleNamespace(env=env, task=task, tick=0, target=None, hold_tcp=[.3, 0, 1.1],
                torch=SimpleNamespace(linalg=SimpleNamespace(vector_norm=np.linalg.norm)),
                private_samples=[], insertion_report={}, insertion_check=SimpleNamespace(samples=[]),
                safe_hold=lambda: None, evaluate=lambda: {"seat_gt": "UNKNOWN"})
            safe_holds = []
            def safe_hold():
                safe_holds.append(list(xyz))
                adapter.target = None
            adapter.safe_hold = safe_hold
            xyz, force = [.3, 0, 1.1], 0.
            adapter._tcp = lambda: (xyz, [0, 0, 1, 0])
            adapter._force = lambda name: force if name == "hub_contact" else 20.
            adapter.protection = lambda safety, **kw: "force_limit" if force > safety["force_limit_N"] else None
            def idle():
                adapter.tick += 1
                adapter.private_samples.append({"hub_xyz": list(xyz), "hub_quat": [1, 0, 0, 0]})
            adapter.idle_step = idle
            def waypoint(point, check, **kw):
                nonlocal xyz, force
                adapter.target = list(point["tcp_xyz_m"])
                if kw.get("insertion"):
                    adapter.target[2] += .02
                    force = contact
                    xyz = [adapter.target[0]+.001, adapter.target[1], adapter.target[2]]
                else:
                    xyz = list(adapter.target)
                env._gripper_target = point.get("gripper_opening_m", env._gripper_target)
                joints[:] = env._gripper_target
                adapter.idle_step()
                check()
            adapter._waypoint = waypoint
            with tempfile.TemporaryDirectory() as temp:
                result = run_pilot(adapter, plan, temp, stage="place")
                sample_phases = [json.loads(line)["phase"] for line in
                                 (Path(temp)/"public_samples.jsonl").read_text().splitlines()]
            commands = {row["phase"]: row for row in result["commands"]}
            if contact == 130:
                self.assertEqual(result["exit_reason"], "force_limit")
                self.assertNotIn("release", commands)
            else:
                self.assertEqual(result["exit_reason"], "pilot_complete")
                stopped = commands["insert"]["stopped_command_tcp_xyz_m"]
                self.assertAlmostEqual(stopped[2]-commands["insert"]["tcp_xyz_m"][2], .02)
                release_command = commands["release"]
                release = release_command["tcp_xyz_m"]
                self.assertAlmostEqual(release[0]-stopped[0], .001)
                self.assertEqual(release_command["ticks"], 3)
                self.assertEqual(sample_phases.count("insert"), 1)
                self.assertGreaterEqual(len(safe_holds), 2)
                self.assertAlmostEqual(commands["retract"]["tcp_xyz_m"][2]-stopped[2], plan["retract_clearance_m"])

    def test_scripted_request_smoke_does_not_fit_request_as_completion(self):
        plan = {"prefix": ["inspect"], "candidate_id": "help", "continuation": [],
                "record_calibration": False}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"smoke"
            result = run_episode(ContractWorld("nominal"), config(), output, branch_plan=plan)
            self.assertTrue(result["help_request_sent"])
            self.assertTrue(json.loads((output/"help_request.json").read_text())["evidence_refs"])
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["model_mode"], "scripted_actions_no_model")
            self.assertEqual(manifest["run_role"], "scripted_smoke")
            self.assertFalse((output/"calibration_record.json").exists())
            plan["record_calibration"] = True
            with self.assertRaisesRegex(ValueError, "help request has no helper-completion label"):
                run_episode(ContractWorld("nominal"), config(), Path(temp)/"calibration", branch_plan=plan)

    def test_model_history_keeps_event_refs_and_tool_results_without_duplicate_sensor_packets(self):
        ledger = Ledger()
        ref = ledger.append_public("observation", {"observation_id": "obs", "control_epoch": 0,
            "physics_tick": 12, "stage": "initial", "channels": {"contact_scalar_N": 3}})
        ledger.append_public("tool_result", {"status": "STALLED"})
        compact = ledger.public_context(compact_observations=True)
        self.assertEqual(compact["events"][0]["event_id"], ref)
        self.assertEqual(compact["events"][0]["payload"]["physics_tick"], 12)
        self.assertNotIn("channels", compact["events"][0]["payload"])
        self.assertEqual(compact["events"][1]["payload"], {"status": "STALLED"})
        self.assertIn("channels", ledger.public_context()["events"][0]["payload"])

    def test_candidate_evidence_refs_are_optional_but_validated(self):
        ledger = Ledger()
        ref = ledger.append_public("observation", {"observation_id": "obs"})
        context = {"candidates": [{"candidate_id": "observe", "is_probe": False}]}
        raw = {"observed_facts": [], "hypotheses": [], "rejected_candidates": [],
               "candidates": [{"candidate_id": "observe", "prediction": "new image",
                               "decision_link": "check scene", "evidence_refs": [ref]}],
               "suggested_action": "observe", "evidence_refs": [ref]}
        self.assertEqual(validate_proposal(raw, context, ledger, 3), raw)
        raw["candidates"][0]["evidence_refs"] = ["unknown_event"]
        with self.assertRaises(ValueError):
            validate_proposal(raw, context, ledger, 3)
        raw["candidates"][0]["evidence_refs"] = [ref]
        raw["candidates"][0]["unregistered_field"] = True
        with self.assertRaises(ValueError):
            validate_proposal(raw, context, ledger, 3)

    def test_http_proposer_records_public_response_and_decodes_complete_json_fences(self):
        import io
        from unittest.mock import patch
        from hrc_harness.proposer import HttpProposer
        trace = []
        proposer = HttpProposer({"endpoint": "http://localhost/model", "model": "test", "timeout_s": 1}, trace.append)
        context = {"candidates": [], "ledger": {"events": []}, "observation": {"observation_id": "obs"}}
        observation = SimpleNamespace(frames={})
        expected = {"suggested_action": "observe"}
        content = "```json\n" + json.dumps(expected) + "\n```"
        response = {"choices": [{"message": {"content": content}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            self.assertEqual(proposer(context, observation), expected)
        self.assertEqual(trace, [{"context": context, "response": content}])
        response["choices"][0]["message"]["content"] = '```json\n{"suggested_action":'
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(json.JSONDecodeError):
                proposer(context, observation)

    def test_gripper_target_changes_are_ramped_each_physics_tick(self):
        from hrc_harness.isaac import IsaacSceneAdapter
        adapter = IsaacSceneAdapter.__new__(IsaacSceneAdapter)
        targets = []
        adapter.env = SimpleNamespace(cfg=SimpleNamespace(sim_dt=.01), device="cpu", _gripper_target=.02,
            set_pose_target=lambda *args: None, set_gripper=targets.append)
        adapter.config = {"safety": {"tcp_speed_limit_mps": 1, "workspace_xyz_m": [[-1, 1]]*3}}
        adapter.task = {"frames": {"robot_tool_offset_m": [0, 0, 0]}}
        adapter.torch = SimpleNamespace(tensor=lambda value, **kwargs: value)
        adapter._tcp = lambda: ([0, 0, 0], [1, 0, 0, 0])
        adapter.idle_step = lambda: None
        adapter._waypoint(dict(tcp_xyz_m=[0, 0, 0], quat_wxyz=[1, 0, 0, 0], ticks=4,
                               gripper_opening_m=.04), lambda: None)
        self.assertEqual(len(targets), 4)
        for actual, expected in zip(targets, (.025, .03, .035, .04)):
            self.assertAlmostEqual(actual, expected)

    def test_offline_grasp_measurement_exposes_slip_without_certifying_it(self):
        from hrc_harness.pilot import measure_grasp
        self.assertEqual(measure_grasp([])["status"], "insufficient_lift_data")
        rows = [{"phase": phase, "private": {"hub_xyz": [0, 0, hub], "hub_quat": [1, 0, 0, 0]},
                 "public": {"tcp_xyz_m": [0, 0, tcp], "tcp_quat_wxyz": [1, 0, 0, 0], "pair_contact_N": [10, 12]}}
                for phase, hub, tcp in (("close", 1., 1.), ("held_dwell", 1.08, 1.10))]
        result = measure_grasp(rows)
        self.assertAlmostEqual(result["cover_lift_m"], .08)
        self.assertAlmostEqual(result["relative_translation_drift_m"], .02)
        self.assertAlmostEqual(result["relative_rotation_drift_deg"], 0)
        self.assertEqual(result["status"], "measured_not_certified")

    def test_reset_includes_mimic_slaves_before_the_first_physics_tick(self):
        from hrc_harness.isaac import reset_scene
        calls = []
        env = SimpleNamespace(_reset_joint_ids=[1, 2], cfg=SimpleNamespace(harness_cover_support_offsets={}),
            robot=SimpleNamespace(find_joints=lambda pattern: ([2, 3], ["left_gripper_axis2", "right_gripper_axis2"])),
            prepare_scatter_reset=lambda: calls.append("prepare"),
            reset=lambda seed: calls.append((seed, list(env._reset_joint_ids))))
        reset_scene(env, 7)
        self.assertEqual(calls, ["prepare", (7, [1, 2, 3])])

    def test_scene_task_height_matches_the_actual_table_and_supports(self):
        from hrc_harness.isaac import configure_scene, load_task
        from hrc_harness.pilot import nominal_points
        task = load_task(ROOT / "configs/task_hub_cover.yaml")
        rigid = lambda: SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace()), init_state=SimpleNamespace())
        cfg = SimpleNamespace(sim=SimpleNamespace(), hub_cfg=rigid(), casing_cfg=rigid(),
            robot_cfg=SimpleNamespace(spawn=SimpleNamespace(), init_state=SimpleNamespace(
                joint_pos={"left_gripper_axis1": .04, "right_gripper_axis1": .04})),
            hub_reset_pos=(0, 0, 1), casing_reset_pos=(0, 0, 1), table_center_xy=(1.6, 0),
            scatter_spawn_margin_m=.002,
            table_cfg=SimpleNamespace(init_state=SimpleNamespace(), spawn=SimpleNamespace(size=(3.2, 2, .1))))
        for name in ("hub_support_n_cfg", "hub_support_s_cfg", "hub_support_e_cfg", "hub_support_w_cfg", "casing_support_cfg"):
            setattr(cfg, name, SimpleNamespace(spawn=SimpleNamespace(size=(.06, .02, .01))))
        configure_scene(cfg, task)
        self.assertAlmostEqual(cfg.table_cfg.init_state.pos[2]+.05, task["scene"]["table_top_z_m"])
        self.assertAlmostEqual(cfg.table_cfg.init_state.pos[0]-1.6, .15)
        self.assertFalse(cfg.hub_cfg.spawn.rigid_props.disable_gravity)
        self.assertFalse(cfg.hub_cfg.spawn.rigid_props.kinematic_enabled)
        self.assertFalse(cfg.spawn_grasp_constraint)
        self.assertEqual(cfg.decimation, 1)
        self.assertFalse(cfg.torso_runtime_override)
        plan = load_task(ROOT / "configs/harness_pilot.yaml")
        task["scene"].update(plan["scene"])
        task["scene"]["fixed_bundle_torso_posture"] = True
        cfg.robot_cfg.spawn.collision_props = SimpleNamespace()
        configure_scene(cfg, task)
        self.assertTrue(cfg.torso_runtime_override)
        self.assertEqual(cfg.torso_limit_half_range, 0.0)
        self.assertEqual(cfg.casing_cfg.init_state.rot, (0, 0, 0, 1))
        self.assertEqual(cfg.hub_cfg.init_state.rot, tuple(plan["scene"]["cover_initial_quat_wxyz"]))
        self.assertEqual(cfg.scatter_hub_support_top_z, 1.066)
        self.assertAlmostEqual(cfg.hub_support_n_cfg.spawn.size[2], .234)
        self.assertAlmostEqual(cfg.casing_support_cfg.spawn.size[2], .102)
        self.assertEqual(cfg.robot_cfg.init_state.joint_pos["left_gripper_axis2"], .04)
        close, seat = nominal_points(task, plan)
        tool = rotate(plan["grasp_quat_wxyz"], task["frames"]["robot_tool_offset_m"])
        self.assertAlmostEqual(seat[1]-plan["grasp_pair_offset_m"]-tool[1], .04928584)
        self.assertAlmostEqual(close[1]-plan["grasp_pair_offset_m"]-tool[1], -.17)
        task["scene"].update(cover_initial_xy=[.4, .45], casing_initial_xy=[.55, 0])
        plan = {**plan, "grasp_quat_wxyz": [0, 1, 0, 0],
                "grasp_link6_x_offset_m": .05, "grasp_pair_offset_m": .02}
        close, seat = nominal_points(task, plan)
        self.assertAlmostEqual(close[0], .4579)
        self.assertAlmostEqual(close[1], .47)
        self.assertAlmostEqual(seat[0]-close[0], .15)
        self.assertGreater(seat[2], task["scene"]["table_top_z_m"])
        trim = [0.01, -0.02, 0.003]
        _, trimmed_seat = nominal_points(task, {**plan, "seat_target_trim_casing_local_m": trim})
        _, untrimmed_seat = nominal_points(task, plan)
        world_trim = rotate(task["scene"]["casing_initial_quat_wxyz"], trim)
        for actual, original, delta in zip(trimmed_seat, untrimmed_seat, world_trim):
            self.assertAlmostEqual(actual-original, delta)

    def test_rule_insertion_check_uses_contact_progress_and_duration(self):
        settings = dict(window_s=.2, min_progress_mm=.1, contact_threshold_N=5,
                        target_tolerance_mm=.1, timeout_s=1)
        for label, positions, forces, expected in (
            ("persistent_contact", [0, 0, 0], [8, 8, 8], "BLOCKED"),
            ("transient_force", [0, 0, 0], [0, 8, 0], "IN_PROGRESS"),
            ("moving_under_load", [0, 1, 2], [8, 8, 8], "IN_PROGRESS"),
            ("moving_wrong_way", [0, -1, -2], [8, 8, 8], "BLOCKED"),
        ):
            with self.subTest(label=label):
                checker = InsertionCheck(settings)
                for index, (position, force) in enumerate(zip(positions, forces)):
                    result = checker.update(index*.1, position, 5-position, force, "yes")
                self.assertEqual(result["state"], expected)
                public_only(result)
        checker = InsertionCheck(settings)
        self.assertEqual(checker.update(1, 0, 5, 0, "yes")["reason"], "insertion_timeout")
        result = checker.update(1.1, 5, 0, 8, "yes")
        self.assertEqual(result["reason"], "tcp_target_reached_not_seating_success")
        self.assertNotIn(result["state"], {"SUCCESS", "CANDIDATE_COMPLETE"})

    def test_rule_insertion_check_missing_data_breaks_the_window(self):
        settings = dict(window_s=.2, min_progress_mm=.1, contact_threshold_N=5,
                        target_tolerance_mm=.1, timeout_s=1)
        checker = InsertionCheck(settings)
        self.assertEqual(InsertionCheck({}).update(0, 0, 5, 8, "yes")["state"], "UNKNOWN")
        for progress, remaining, force, holding in ((None, 5, 8, "yes"), (0, None, 8, "yes"),
                                                    (0, 5, None, "yes"), (0, 5, 8, "unknown")):
            with self.subTest(progress=progress, remaining=remaining, force=force, holding=holding):
                checker.samples.clear()
                checker.update(0, 0, 5, 8, "yes")
                self.assertEqual(checker.update(.1, progress, remaining, force, holding)["state"], "UNKNOWN")
                self.assertEqual(checker.update(.2, 0, 5, 8, "yes")["state"], "IN_PROGRESS")
        for name, value in (("window_s", 0), ("timeout_s", -.1), ("contact_threshold_N", float("nan"))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                InsertionCheck({**settings, name: value})

    def test_monitor_completion_is_public_fresh_and_not_a_force_threshold(self):
        world = ContractWorld()
        now = time.monotonic()
        obs = world.observe(0)
        channels = dict(obs.channels)
        channels["contact_scalar_N"] = Channel(100, "N", now, "test_sensor")
        self.assertFalse(monitor(replace(obs, channels=channels), now, 5)["reported_success"])
        for name in ("visual_seated", "released", "stable_observed"):
            channels[name] = Channel(True, "state", now, "test_public_estimate")
        obs = replace(obs, channels=channels)
        self.assertEqual(monitor(obs, now, 5)["state"], "CANDIDATE_COMPLETE")
        for packet, clock in ((replace(obs, valid=False), now), (obs, now+6)):
            self.assertEqual(monitor(packet, clock, 5)["state"], "UNKNOWN")
            self.assertFalse(monitor(packet, clock, 5)["reported_success"])
        channels["insertion_state"] = Channel("BLOCKED", "state", now, "rule_force_tcp_time")
        channels["insertion_reason"] = Channel("sustained_contact_without_progress", "state", now, "rule_force_tcp_time")
        self.assertEqual(monitor(replace(obs, channels=channels), now, 5)["state"], "BLOCKED")

    def test_isaac_rule_check_stops_before_release_and_reaches_request_boundary(self):
        from hrc_harness.isaac import IsaacSceneAdapter
        for mode, expected_reason in (("jam", "sustained_contact_without_progress"),
                                      ("low_force", "insertion_profile_ended_before_target"),
                                      ("normal", "bounded_profile_complete"),
                                      ("opposite_axis", "bounded_profile_complete")):
            with self.subTest(mode=mode):
                # Synthetic robot/sensors only. Exercise the actual Isaac waypoint loop, not part GT.
                adapter = IsaacSceneAdapter.__new__(IsaacSceneAdapter)
                adapter.config = {"safety": {"tcp_speed_limit_mps": 1, "workspace_xyz_m": [[-1, 1]]*3}}
                points = [dict(phase="insert", tcp_xyz_m=[0, 0, .002], quat_wxyz=[1, 0, 0, 0], ticks=10),
                          dict(phase="release", tcp_xyz_m=[0, 0, .002], quat_wxyz=[1, 0, 0, 0],
                               ticks=1, gripper_opening_m=.04)]
                adapter.task = {"frames": {"certified": True, "socket_axis_world": [0, 0, -1 if mode == "opposite_axis" else 1],
                                           "robot_tool_offset_m": [0, 0, 0]},
                                "profiles": {"test_seat": {"certified": True, "waypoints": points}}}
                adapter.insertion_check = InsertionCheck(dict(window_s=.02, min_progress_mm=.1,
                    contact_threshold_N=5, target_tolerance_mm=.1, timeout_s=.2))
                adapter.insertion_report = {}
                adapter.tick, adapter.force_exposure, adapter.hold_certified = 0, 0, True
                adapter.stage, adapter.target = "prealigned", None
                adapter.torch = SimpleNamespace(tensor=lambda value, **kwargs: value)
                released = []
                xyz = [0, 0, 0]
                adapter.env = SimpleNamespace(cfg=SimpleNamespace(sim_dt=.01), device="cpu",
                    _gripper_target=.02, set_pose_target=lambda *args: None, set_gripper=released.append)
                adapter._tcp = lambda: (list(xyz), [1, 0, 0, 0])
                adapter._holding = lambda: "yes"
                adapter._force = lambda name: 0 if mode == "low_force" else 8
                adapter.safe_hold = lambda: None
                def tick():
                    adapter.tick += 1
                    adapter.last_robot_stamp = time.monotonic()
                    if mode in {"normal", "opposite_axis"}:
                        xyz[:] = adapter.target
                adapter.idle_step = tick
                spec = ToolSpec("seat", "seat_once", profile="test_seat", requires_holding=True)
                status, reason, actual, _ = adapter.execute(spec, lambda: None)
                self.assertEqual(reason, expected_reason)
                if mode in {"normal", "opposite_axis"}:
                    self.assertEqual(status, "COMPLETED")
                    self.assertEqual(released, [.04])
                    self.assertEqual(adapter.tick, 11)
                else:
                    self.assertEqual(status, "STALLED")
                    self.assertFalse(released)
                    self.assertLessEqual(adapter.tick, 10)
                    # The shared policy sees measured tool status and emits a request without a helper.
                    world = ContractWorld()
                    original = world.execute
                    def execute(registered, check):
                        if registered.tool == "seat_once":
                            return status, reason, actual, MeasuredCost()
                        return original(registered, check)
                    world.execute = execute
                    with tempfile.TemporaryDirectory() as temp:
                        summary = run_episode(world, config(), Path(temp) / "run")
                    self.assertEqual(summary["terminal"], "help_requested")
                    self.assertFalse(world._cleared)

    def test_nominal_and_blocked_closed_loop(self):
        for condition, contacts, helps, terminal in (("nominal", 1, 0, "finish"),
                                                   ("blocked", 1, 1, "help_requested")):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "run"
                result = run_episode(ContractWorld(condition), config(), out)
                self.assertEqual(result["terminal"], terminal)
                self.assertEqual(result["resources"]["contacts"], contacts)
                self.assertEqual(result["resources"]["helps"], helps)
                self.assertFalse(result["physical_validation"])
                public = (out / "public_trace.jsonl").read_text()
                self.assertNotIn('"condition"', public)
                self.assertNotIn('"seat_gt"', public)
                self.assertIn('"condition"', (out / "private_gt.jsonl").read_text())

    def test_all_methods_share_contract_environment(self):
        for method in ("generic", "repair_adapted", "random_safe", "information", "auto", "fixed_retry"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                result = run_episode(ContractWorld("blocked"), config(), Path(temp) / "run", method=method)
                self.assertLessEqual(result["resources"]["contacts"], 4)
                self.assertEqual(result["terminal"], "stop" if method == "auto" else "help_requested")

    def test_no_private_keys_or_frame_paths(self):
        for key in ("ground_truth", "scenario_family", "blocker_present", "metrics_path", "reward"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                public_only({"nested": [{key: True}]})
        world = ContractWorld()
        obs = replace(world.observe(0), frames={"front": "/tmp/private_condition/front.png"},
                      frame_timestamps={"front": time.monotonic()})
        self.assertNotIn("private_condition", json.dumps(obs.public_dict(time.monotonic(), 5)))
        self.assertEqual(features(world.observe(0), time.monotonic(), 5, Ledger()),
                         features(ContractWorld("blocked").observe(0), time.monotonic(), 5, Ledger()))

    def test_ledger_is_append_only_and_refs_checked(self):
        ledger = Ledger()
        raw = {"measurement": [1]}
        ref = ledger.append_public("measurement", raw)
        raw["measurement"][0] = 9
        exposed = ledger.public_context()
        exposed["events"].clear()
        self.assertEqual(ledger.public_context()["events"][0]["payload"]["measurement"], [1])
        ledger.check_refs([ref])
        with self.assertRaises(ValueError):
            ledger.update_hypotheses([{"support_refs": ["nonexistent"]}])

    def test_stale_epoch_observation_and_duplicates(self):
        runtime = Runtime(ContractWorld(), config())
        pick = call(runtime, "pick", "pick1")
        self.assertEqual(runtime.execute(pick).status, "COMPLETED")
        self.assertEqual(runtime.execute(pick).exit_reason, "duplicate_call_id")
        stale = replace(call(runtime, "seat"), control_epoch=99)
        self.assertEqual(runtime.execute(stale).exit_reason, "stale_epoch")
        stale = replace(call(runtime, "seat"), based_on_observation="old")
        self.assertEqual(runtime.execute(stale).exit_reason, "stale_observation")
        runtime.observation = replace(runtime.observation, wall_timestamp=runtime.clock()-10)
        self.assertEqual(runtime.execute(call(runtime, "seat")).exit_reason, "stale_observation")

    def test_helper_ownership_and_unknown_verification(self):
        world = ContractWorld("blocked")
        cfg = config()
        cfg["help_mode"] = "intervene_resume"
        runtime = Runtime(world, cfg)
        runtime.execute(call(runtime, "pick"))
        original = world.help
        def help_with_attempt(request, check):
            self.assertEqual(runtime.owner, "helper")
            self.assertEqual(runtime.execute(call(runtime, "seat")).exit_reason, "ownership")
            return original(request, check)
        world.help = help_with_attempt
        world.invalid_after_help = True
        runtime.execute(call(runtime, "help"))
        self.assertEqual(runtime.owner, "robot")
        self.assertEqual(runtime.epoch, 2)
        self.assertEqual(runtime.verification, "UNKNOWN")
        self.assertNotIn("seat", [item.candidate_id for item in runtime.candidates()])

    def test_verifier_uses_fresh_measurements_not_report(self):
        world = ContractWorld()
        world._holding = "yes"
        obs = replace(world.observe(2), physics_tick=10)
        params = dict(epoch=2, after_tick=9, now=time.monotonic(), freshness_s=5, tracking_limit_m=.01)
        self.assertEqual(verify_after_help(obs, **params), "PASS")
        self.assertEqual(verify_after_help(replace(obs, physics_tick=9), **params), "UNKNOWN")
        self.assertEqual(verify_after_help(replace(obs, control_epoch=1), **params), "UNKNOWN")
        channels = dict(obs.channels)
        channels["holding"] = replace(channels["holding"], value="no")
        self.assertEqual(verify_after_help(replace(obs, channels=channels), **params), "FAIL")
        world = ContractWorld("blocked")
        world.partial_help = True
        cfg = config()
        cfg["help_mode"] = "intervene_resume"
        runtime = Runtime(world, cfg)
        runtime.execute(call(runtime, "pick"))
        runtime.execute(call(runtime, "help"))
        self.assertEqual(runtime.execute(call(runtime, "seat")).status, "STALLED")

    def test_every_probe_charges_contact_and_help_reserves_resume(self):
        cfg = config()
        cfg["help_mode"] = "intervene_resume"
        runtime = Runtime(ContractWorld("blocked"), cfg)
        runtime.execute(call(runtime, "pick"))
        runtime.execute(call(runtime, "xy"))
        self.assertEqual((runtime.used.contacts, runtime.used.probes), (1, 1))
        runtime.used = Resources(contacts=4)
        self.assertEqual(runtime.execute(call(runtime, "help")).exit_reason, "help_resume_budget")
        self.assertEqual(runtime.used.helps, 0)
        invalid = config()
        invalid["tools"][2]["resources"]["contacts"] = 0
        with self.assertRaises(ValueError):
            Runtime(ContractWorld(), invalid)

    def test_cancel_interrupts_mid_tool(self):
        world = ContractWorld()
        runtime = Runtime(world, config())
        original = world.idle_step
        def tick():
            original()
            runtime.cancel.set()
        world.idle_step = tick
        result = runtime.execute(call(runtime, "pick"))
        self.assertEqual(result.status, "ABORTED")
        self.assertEqual(world.tick, 1)
        self.assertEqual(runtime.terminal, "stop")
        self.assertTrue(world.stopped)

    def test_model_wait_steps_physics_and_times_out(self):
        runtime = Runtime(ContractWorld(), config())
        runtime.config["model_timeout_s"] = .01
        def slow(context, obs):
            time.sleep(.1)
        with self.assertRaises(Interrupted):
            runtime.reason(slow, {})
        self.assertGreater(runtime.adapter.tick, 0)
        self.assertGreater(runtime.cost.model_s, 0)
        self.assertEqual(runtime.terminal, "stop")

    def test_missing_sensor_is_not_zero(self):
        obs = ContractWorld().observe(0)
        channels = dict(obs.channels)
        channels["contact_scalar_N"] = Channel(None, "N", time.monotonic(), "unavailable", False)
        value = features(replace(obs, channels=channels), time.monotonic(), 5, Ledger())
        self.assertEqual(value["force_band"], "missing")

    def test_depth_two_example_and_single_success_reward(self):
        initial, progress, stalled = state(1), state(0), state(2)
        outcomes = {"SUCCESS": {"probability": .2, "terminal_success_probability": 1},
                    "PROGRESS": {"probability": .4, "next_state": progress},
                    "STILL_STALLED": {"probability": .4, "next_state": stalled}}
        table = {"version": 1, "provenance": {"backend": "contract_only"}, "states": {
            state_key(initial): {"seat": estimate(.35, .12), "help": estimate(.9, .4), "xy": estimate(.2, .1, outcomes)},
            state_key(progress): {"seat": estimate(.8, .12), "help": estimate(.9, .4)},
            state_key(stalled): {"seat": estimate(.35, .12), "help": estimate(.9, .4)}}}
        selector = Selector(Estimator(table, allow_contract_data=True), cost_weights={"wall_s": 1},
                            help_mode="intervene_resume")
        candidates = [ToolSpec("seat", "seat_once", resources=Resources(contacts=1)),
                      ToolSpec("help", "ask_act", resources=Resources(helps=1)),
                      ToolSpec("xy", "probe_xy", resources=Resources(contacts=1, probes=1)), ToolSpec("stop", "stop")]
        rows = selector.rank(initial, candidates, lambda resource: resource.contacts <= 2, 0)
        values = {row["candidate_id"]: row["Q_hat"] for row in rows}
        self.assertAlmostEqual(values["seat"], .23)
        self.assertAlmostEqual(values["help"], .50)
        self.assertAlmostEqual(values["xy"], .572)
        self.assertEqual(selector.choose(rows, candidates, method="proposed", suggested="help", fallback="stop")[0], "xy")
        constrained = selector.rank(initial, candidates, lambda resource: resource.contacts <= 1, 0)
        self.assertLess(next(row["Q_hat"] for row in constrained if row["candidate_id"] == "xy"), .572)

    def test_unsupported_and_synthetic_estimates_are_explicit(self):
        with self.assertRaises(ValueError):
            Estimator({"version": 1, "provenance": {"backend": "contract_only"}, "states": {}})
        self.assertIsNone(Estimator().predict(state(), "seat"))
        with self.assertRaises(ValueError):
            state_key({**state(), "true_offset": .01})
        with self.assertRaises(ValueError):
            Selector(Estimator(), cost_weights={"wall_s": 1, "model_s": 1})

    def test_calibration_group_split_and_no_test_fitting(self):
        row = {"snapshot_id": "a", "split": "train", "backend": "isaac", "public_state": state(),
               "candidate_id": "seat", "outcome": "STILL_STALLED", "terminal_success": False,
               "cost": estimate(0, 0)["cost"]}
        records = [row, {**row, "snapshot_id": "b"},
                   {**row, "snapshot_id": "c", "split": "test", "terminal_success": True},
                   {**row, "snapshot_id": "d", "split": "validation", "terminal_success": False}]
        table, report = fit(records)
        self.assertEqual(Estimator(table).predict(state(), "seat").success_probability, 0)
        self.assertEqual(report["test_records_not_fitted"], 1)
        self.assertEqual(report["brier"], 0)
        with self.assertRaises(ValueError):
            fit([row, {**row, "split": "test"}])
        with self.assertRaises(ValueError):
            fit([{**row, "candidate_id": "help"}])

    def test_unverified_geometry_never_passes_gt(self):
        import yaml
        task = yaml.safe_load((ROOT / "configs/task_hub_cover.yaml").read_text())
        self.assertEqual(judge([], task)["seat_gt"], "UNKNOWN")
        self.assertEqual(judge([{}], task)["seat_gt"], "UNKNOWN")

    def test_seat_tilt_is_axis_based_and_registration_separate(self):
        import yaml
        task = yaml.safe_load((ROOT / "configs/task_hub_cover.yaml").read_text())
        task["frames"].update(certified=True, socket_axis_local=[0, 0, 1], plug_axis_local=[0, 0, 1],
                              seated_relative_position_m=[0, 0, 0], seated_relative_quat_wxyz=[1, 0, 0, 0],
                              registration_symmetry_quats=[[1, 0, 0, 0]])
        task["evaluation"].update(certified=True, registration_certified=True, registration_tolerance_deg=2)
        sample = {"hub_xyz": [0, 0, 0], "casing_xyz": [0, 0, 0],
                  "hub_quat": [math.sqrt(.5), 0, 0, math.sqrt(.5)], "casing_quat": [1, 0, 0, 0],
                  "released": True, "support_contact": True, "penetration_m": 0, "speed_mps": 0, "dt_s": .1}
        result = judge([sample]*5, task)
        self.assertEqual(result["seat_gt"], "PASS")
        self.assertEqual(result["registration_gt"], "FAIL")
        self.assertAlmostEqual(result["geometry"]["tilt_deg"], 0)
        self.assertEqual(judge([{**sample, "penetration_m": None}], task)["seat_gt"], "UNKNOWN")
        offset = {**sample, "hub_xyz": [0.003, 0.004, 0.001]}
        diagnostic = judge([offset]*5, task)["geometry"]
        self.assertEqual(diagnostic["position_error_casing_local_m"], [0.003, 0.004, 0.001])
        self.assertEqual(diagnostic["radial_error_vector_casing_local_m"], [0.003, 0.004, 0.0])
        self.assertAlmostEqual(diagnostic["full_orientation_error_deg"], 90)
        self.assertAlmostEqual(diagnostic["orientation_error_quat_casing_local_wxyz"][0], math.sqrt(.5))

    def test_visual_facts_are_derived_and_single_frame_is_not_stability(self):
        world = ContractWorld()
        raw_observe = world.observe
        def camera_observe(epoch):
            packet = raw_observe(epoch)
            return replace(packet, frames={"front": "/tmp/front.png"},
                           frame_timestamps={"front": time.monotonic()})
        world.observe = camera_observe
        runtime = Runtime(world, config())
        original = runtime.ledger.public_context()["events"][0]
        fact = {"name": "stable_observed", "value": True,
                "observation_id": runtime.observation.observation_id, "evidence_refs": [original["event_id"]]}
        runtime.apply_visual_facts([fact])
        self.assertNotIn("stable_observed", runtime.visual_facts)
        fact["name"] = "target_visible"
        runtime.apply_visual_facts([fact])
        self.assertIn("target_visible", runtime.visual_facts)
        self.assertEqual(runtime.ledger.public_context()["events"][0], original)

    def test_calibration_branch_is_marked_synthetic_and_prefix_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            plan = {"snapshot_id": "test_snapshot", "split": "train", "prefix": ["pick", "seat"],
                    "candidate_id": "help", "continuation": ["seat", "finish"]}
            cfg = config()
            cfg["help_mode"] = "intervene_resume"
            run_episode(ContractWorld("blocked"), cfg, out, branch_plan=plan)
            record = json.loads((out / "calibration_record.json").read_text())
            self.assertEqual(record["completion_scope"], "help_verify_robot_resume")
            self.assertEqual(record["backend"], "contract_only")
            self.assertTrue(record["terminal_success"])
            with self.assertRaises(ValueError):
                fit([record])

    def test_false_finish_is_scored_only_after_online_finish(self):
        with tempfile.TemporaryDirectory() as temp:
            world = ContractWorld()
            out = Path(temp) / "run"
            def evaluator():
                events = [json.loads(line) for line in (out / "public_trace.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["payload"]["exit_reason"], "finish")
                return {"seat_gt": "FAIL", "registration_gt": "UNKNOWN", "physical_validation": False}
            world.evaluate = evaluator
            result = run_episode(world, config(), out)
            self.assertTrue(result["false_finish"])
            self.assertEqual(result["terminal"], "finish")

    def test_isaac_online_methods_do_not_query_part_truth_or_write_part_pose(self):
        import inspect
        from hrc_harness.isaac import IsaacSceneAdapter
        for method in ("observe", "execute", "protection", "_waypoint", "safe_hold", "_tcp", "_holding"):
            source = inspect.getsource(getattr(IsaacSceneAdapter, method))
            for forbidden in ("root_states", "hub.data.root", "casing.data.root", "write_root_state", "write_root_pose"):
                self.assertNotIn(forbidden, source)

    def test_request_only_stops_without_helper_mutation_or_resume_reservation(self):
        world = ContractWorld("blocked")
        def forbidden(*args):
            self.fail("request-only must not intervene, quiesce for handover, or resume")
        world.help = world.quiesce = forbidden
        runtime = Runtime(world, config())
        runtime.used = Resources(contacts=4)
        result = runtime.execute(call(runtime, "help"))
        self.assertEqual(result.exit_reason, "help_request_emitted")
        self.assertEqual(runtime.terminal, "help_requested")
        self.assertEqual(runtime.owner, "robot")
        self.assertFalse(world._cleared)
        self.assertEqual((runtime.used.contacts, runtime.used.helps), (4, 1))
        self.assertEqual(runtime.cost.helper_s, 0)
        self.assertEqual(runtime.execute(call(runtime, "seat")).exit_reason, "episode_ended")
        self.assertIsNotNone(runtime.help_request)

    def test_request_only_never_reuses_helper_completion_estimate(self):
        table = {"version": 1, "provenance": {"backend": "contract_only"},
                 "states": {state_key(state()): {"help": estimate(.99, 0), "seat": estimate(.3, .1)}}}
        selector = Selector(Estimator(table, allow_contract_data=True))
        candidates = [ToolSpec("help", "ask_act", resources=Resources(helps=1)),
                      ToolSpec("seat", "seat_once", resources=Resources(contacts=1)), ToolSpec("stop", "stop")]
        rows = selector.rank(state(), candidates, lambda need: need.contacts == 0, 60)
        help_row = next(row for row in rows if row["candidate_id"] == "help")
        self.assertFalse(help_row["supported"])
        self.assertIsNone(help_row["Q_hat"])
        self.assertEqual(help_row["completion_scope"], "help_request_only")
        self.assertEqual(selector.choose(rows, candidates, method="proposed", suggested="help", fallback="help"),
                         ("help", "request_only_help_utility_not_defined_fallback"))

    def test_request_outbox_is_persisted_but_not_calibration_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            world = ContractWorld("blocked")
            summary = run_episode(world, config(), out)
            self.assertEqual(summary["help_delivery"], "local_outbox")
            self.assertEqual(summary["primary_gt"], "FAIL")
            request = json.loads((out / "help_request.json").read_text())
            self.assertEqual(request["target"], "socket_hub_output")
            self.assertFalse(world._cleared)
            from hrc_harness.calibration import record_branch
            with self.assertRaisesRegex(ValueError, "help request has no"):
                record_branch(out, {"prefix": ["pick", "seat"], "candidate_id": "help"})


if __name__ == "__main__":
    unittest.main()
