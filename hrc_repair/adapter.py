"""Skill adapters.

``IsaacSubprocessAdapter`` deliberately reuses the already validated strict
Isaac launcher.  Each call owns one simulator process and one output folder;
there is never more than one stage writer.  This is the smallest safe bridge
until the strict controller is exposed as an in-process skill API.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .contracts import HelpReport, Observation, SkillResult
from .evaluator import placement_success


def _public_contact_summary(metrics: dict[str, Any]) -> dict[str, bool]:
    """Extract only permitted contact/robot-state evidence for the planner.

    This deliberately ignores all pose, score, verdict, scenario, and
    evaluator fields.  The private metrics artifact remains available to the
    independent evaluator and is never placed in the public observation.
    """
    records = metrics.get("records", []) if isinstance(metrics, dict) else []
    if not isinstance(records, list):
        records = []
    by_label: dict[str, dict[str, Any]] = {
        str(item.get("label")): item for item in records if isinstance(item, dict) and item.get("label")
    }
    final = by_label.get("retract") or by_label.get("release_clearance") or (records[-1] if records else {})
    if not isinstance(final, dict):
        final = {}
    gripper = final.get("gripper_to_hub_contact_force_norm_N", {})
    if not isinstance(gripper, dict):
        gripper = {}
    finite_forces = []
    for value in gripper.values():
        try:
            finite_forces.append(float(value))
        except (TypeError, ValueError):
            continue
    hub_force = final.get("hub_casing_force_norm", 0.0)
    gripper_open = final.get("gripper_joint", [0.0])
    try:
        gripper_open = float(gripper_open[0]) if isinstance(gripper_open, list) else float(gripper_open)
    except (TypeError, ValueError, IndexError):
        gripper_open = 0.0
    try:
        support_contact = float(hub_force) > 1.0
    except (TypeError, ValueError):
        support_contact = False
    release_contact_free = bool(finite_forces) and max(finite_forces) <= 1.0
    released = gripper_open >= 0.04
    return {
        "support_contact": support_contact,
        "release_contact_free": release_contact_free,
        "released": released,
        "blocked_guard": False,
    }


def _public_mode(config: dict[str, Any]) -> bool:
    return str(config.get("experiment", {}).get("feedback_mode", "")).startswith("public_")


class ContractAdapter:
    backend_name = "contract_debug"

    def __init__(self, *, scenario: str) -> None:
        self.scenario = scenario
        self.place_calls = 0
        self.help_calls = 0
        self.blocker_present = scenario == "blocked"
        self.episode_id = ""

    def reset(self, episode_id: str, seed: int, out_dir: Path) -> Observation:
        self.episode_id = episode_id
        self.place_calls = 0
        self.help_calls = 0
        self.blocker_present = self.scenario == "blocked"
        return Observation(episode_id, "obs_0000", 0, time.time(), held="yes", target_visible="yes", evidence="contract-only preplace reset")

    def place(self, observation: Observation, out_dir: Path) -> SkillResult:
        self.place_calls += 1
        blocked = self.blocker_present
        return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "place", "guarded_abort" if blocked else "completed", "failed" if blocked else "succeeded", "contact_guard" if blocked else None, feedback={"public_evidence": "registered target area blocked" if blocked else "placement skill completed in contract backend", "public_sensor": {"blocked_guard": blocked, "support_contact": not blocked, "release_contact_free": not blocked, "released": not blocked}})

    def retract(self, observation: Observation, out_dir: Path) -> SkillResult:
        return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "retract", "completed", "succeeded", feedback={"public_evidence": "safe hold reached"})

    def help(self, observation: Observation, out_dir: Path) -> HelpReport:
        self.help_calls += 1
        before = 0
        after = 1
        if self.scenario == "blocked":
            self.blocker_present = False
            status = "succeeded"
            report = "Contract helper removed the registered blocker; cover remains held and must still be placed."
        else:
            status = "unsupported"
            report = "No registered blocker was present in nominal contract scenario."
        return HelpReport(f"help_{self.help_calls:03d}", "clear_target_area", "hub", status, report, time.time(), time.time(), before, after)

    def close(self) -> None:
        return None


class IsaacSubprocessAdapter:
    backend_name = "isaac_strict_subprocess"

    def __init__(self, *, scenario: str, repo_root: Path, config: dict[str, Any], out_dir: Path) -> None:
        self.scenario = scenario
        self.repo_root = repo_root
        self.config = config
        self.out_dir = out_dir
        self.place_calls = 0
        self.episode_id = ""
        self.blocker_present = scenario == "blocked"
        self.blocker_removed = False
        self._last_result_dir: Path | None = None

    def reset(self, episode_id: str, seed: int, out_dir: Path) -> Observation:
        self.episode_id = episode_id
        self.place_calls = 0
        self.blocker_removed = False
        self.out_dir = out_dir
        scatter = str(self.config.get("scene", {}).get("initial_layout", "")) == "scatter"
        return Observation(episode_id, "obs_0000", 0, time.time(), held="no" if scatter else "yes", target_visible="yes", evidence=("Isaac scatter reset places both dynamic parts on the table before the strict skill; no part pose is written after action start." if scatter else "Isaac preplace reset is performed inside the strict skill process; no part pose is written after action start."), sensor_availability={"rgb": False, "robot_state": True, "contact_or_guard": True})

    def place(self, observation: Observation, out_dir: Path) -> SkillResult:
        self.place_calls += 1
        run_dir = out_dir / f"place_{self.place_calls:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._last_result_dir = run_dir
        script = self.repo_root / "tools" / "run_m1_isaac_strict_success.sh"
        env = os.environ.copy()
        controller = self.config.get("controller", {})
        scene = self.config.get("scene", {})
        scatter = str(scene.get("initial_layout", "")) == "scatter" or str(scene.get("start_stage", "")) == "scattered"
        scatter_supports = bool(scene.get("scatter_physical_supports", False))
        cover_xy = scene.get("cover_initial_xy", [0.55, 0.43])
        casing_xy = scene.get("casing_initial_xy", [0.55, 0.0])
        env.update({
            "M1_ISAAC_STRICT_OUTPUT_DIR": str(run_dir),
            "M1_SCATTER_RESET": "1" if scatter else "0",
            "M1_PREPLACE_CONTROLLED_PLACE": "0" if scatter else "1",
            "M1_RETRACT_STAGING_SUPPORTS": "1" if scatter and scatter_supports else ("0" if controller.get("m0_defer_staging_support_retraction", False) else "1"),
            "M1_SCATTER_PHYSICAL_SUPPORTS": "1" if scatter_supports else "0",
            "M1_SCATTER_SUPPORT_TOP_Z": str(float(scene.get("scatter_support_top_z", 0.934))),
            "M1_STAGING_SUPPORT_DROP_M": str(self.config.get("controller", {}).get("staging_support_drop_m", 0.15)),
            "M1_STAGING_SUPPORT_RETRACT_STEPS": str(self.config.get("controller", {}).get("staging_support_retract_steps", 10)),
            "M1_STAGING_SUPPORT_WITHDRAW_MODE": str(self.config.get("controller", {}).get("staging_support_withdraw_mode", "down")),
            "M1_EPISODE_LENGTH_S": str(self.config.get("budgets", {}).get("episode_wall_timeout_s", 180)),
            "M1_HUB_RESET_X": str(float(cover_xy[0])) if scatter else "0.55",
            "M1_HUB_RESET_Y": str(float(cover_xy[1])) if scatter else "0.08683718",
            "M1_HUB_RESET_Z": "" if scatter else "1.205",
            "M1_CASING_RESET_X": str(float(casing_xy[0])),
            "M1_CASING_RESET_Y": str(float(casing_xy[1])),
            "M1_TABLE_SIZE_X": str(float(scene.get("table_size_xy", [1.8, 1.8])[0])),
            "M1_TABLE_SIZE_Y": str(float(scene.get("table_size_xy", [1.8, 1.8])[1])),
            "M1_TABLE_TOP_Z": str(float(scene.get("table_top_z", 0.934))),
            "M1_TABLE_CENTER_X": str(float(scene.get("table_center_xy", [1.20, 0.0])[0])),
            "M1_TABLE_CENTER_Y": str(float(scene.get("table_center_xy", [1.20, 0.0])[1])),
            "M1_SCATTER_SPAWN_MARGIN": str(float(scene.get("scatter_spawn_margin_m", 0.002))),
            # Keep physical calibration explicit in the M0 config.  The
            # contract backend is unaffected, while Isaac runs remain
            # auditable instead of silently using launcher defaults.
            "M1_GRIP_OPENING": str(self.config.get("controller", {}).get("grip_opening_m", 0.005)),
            "M1_RELEASE_OPENING": str(self.config.get("controller", {}).get("release_opening_m", 0.05)),
            "M1_GRASP_CORRECTION_X_DEG": os.environ.get("M1_GRASP_CORRECTION_X_DEG", str(controller.get("grasp_correction_x_deg", 0))),
            "M1_GRASP_CORRECTION_Y_DEG": os.environ.get("M1_GRASP_CORRECTION_Y_DEG", str(controller.get("grasp_correction_y_deg", 0))),
            "M1_PREPLACE_RELEASE_CLEARANCE_Y": str(self.config.get("controller", {}).get("preplace_release_clearance_y_m", 0.03)),
            "M1_PREPLACE_SEAT_EXTRA_DEPTH_M": str(self.config.get("controller", {}).get("preplace_seat_extra_depth_m", 0.0)),
            "M1_PREPLACE_CORRECT_ORIENTATION": "1" if self.config.get("controller", {}).get("preplace_correct_orientation", False) else "0",
            "M1_CORRECT_ORIENTATION_AT_INSERTION_ABOVE": "1" if controller.get("m0_correct_orientation_at_insertion_above", False) else "0",
            "M1_RELEASE_RIGHT_AT_INSERTION_ABOVE": "1" if controller.get("m0_release_right_at_insertion_above", False) else "0",
            "M1_GRIPPER_STATIC_FRICTION": str(self.config.get("controller", {}).get("gripper_static_friction", 4.0)),
            "M1_GRIPPER_DYNAMIC_FRICTION": str(self.config.get("controller", {}).get("gripper_dynamic_friction", 3.0)),
            "M1_TRANSPORT_CLEARANCE_Z": str(self.config.get("controller", {}).get("transport_clearance_z_m", 0.55)),
            "M1_FOLLOW_LINK_ORIENTATION": "1" if self.config.get("controller", {}).get("follow_link_orientation", False) else "0",
            "M1_ROTATE_HELD_ORIENTATION": "1" if self.config.get("controller", {}).get("rotate_held_orientation", False) else "0",
            "M1_BLOCKED": "1" if self.scenario == "blocked" and not self.blocker_removed else "0",
            "M1_INSERTION_SEGMENTS": "8",
            "M1_INSERTION_SEGMENT_STEPS": "40",
            "M1_NO_VIDEO": "0" if self.config.get("record_video", True) else "1",
            "M1_HEADLESS": "1",
            "M1_RENDER_INTERVAL": str(self.config.get("controller", {}).get("render_interval", 20)),
            "M1_CAMERA_UPDATE_STRIDE": str(self.config.get("controller", {}).get("camera_update_stride", 20)),
            "M1_INSERT_SAFE_WAYPOINT": "1" if self.config.get("controller", {}).get("insert_safe_waypoint", False) else "0",
            "M1_STRICT_ACCEPTANCE": "0",
            "M1_PLACE_ACCEPTANCE": "1",
            "M1_TRACK_TIP_CONTACT": "0",
        })
        started = time.time()
        try:
            completed = subprocess.run([str(script)], cwd=self.repo_root, env=env, text=True, capture_output=True, timeout=float(self.config.get("budgets", {}).get("episode_wall_timeout_s", 180)) + 60)
            (run_dir / "launcher.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "launcher.stderr.log").write_text(completed.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired:
            return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "place", "system_error", "unknown", "system_error", started_at=started, ended_at=time.time(), safe_exit_complete=False, feedback={"public_evidence": "Isaac subprocess timed out"})
        metrics_path = run_dir / "metrics.json"
        episode_video = run_dir / "episode.mp4"
        # The strict runner writes the live camera stream as
        # ``inner_wall_probe.mp4``; normalize that name for the M0 artifact
        # contract without copying or rewriting the video.
        if not episode_video.exists() and (run_dir / "inner_wall_probe.mp4").exists():
            episode_video = run_dir / "inner_wall_probe.mp4"
        if episode_video.exists() and not (run_dir / "rollout.mp4").exists():
            try:
                (run_dir / "rollout.mp4").symlink_to(episode_video.name)
            except OSError:
                pass
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if not metrics_path.exists():
            # A Kit process that exits after scene setup (most commonly an
            # RTX/camera startup failure) is not a physical placement
            # failure.  Keep this distinct so the planner cannot spend a
            # help budget on a renderer/control-system fault.
            return SkillResult(
                observation.episode_id,
                observation.observation_id,
                observation.control_epoch,
                "place",
                "system_error",
                "unknown",
                "no_metrics_artifact",
                started_at=started,
                ended_at=time.time(),
                safe_exit_complete=False,
                feedback={
                    "public_evidence": "Isaac subprocess exited without metrics.json; no physical outcome was assigned",
                    "metrics_path": str(metrics_path),
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                },
            )
        public = _public_mode(self.config)
        public_sensor = _public_contact_summary(metrics) if public else None
        if public:
            placement, success = {}, False
            score = {}
        elif self.config.get("experiment", {}).get("task_mode") == "placement":
            success, placement = placement_success(metrics, self.config)
            score = metrics.get("physical_place_score", {}) if isinstance(metrics, dict) else {}
        else:
            score = metrics.get("physical_place_score", {}) if isinstance(metrics, dict) else {}
            placement = {}
            success = int(score.get("total", 0)) >= int(score.get("max", 4)) if isinstance(score, dict) else False
            if "SUCCESS" in str(metrics.get("insertion_verdict", "")):
                success = True
        if self.scenario == "blocked" and not self.blocker_removed:
            # The first blocked skill is a real colliding Isaac cuboid.  The
            # subprocess bridge cannot keep the same Kit process alive across
            # an external helper handoff, so a post-help attempt starts a
            # fresh reset with the blocker omitted.  This limitation is
            # recorded rather than hidden; the in-process handoff remains a
            # required M0 follow-up.
            feedback = {"public_evidence": "registered target blocker stopped the physical descent", "metrics_path": str(metrics_path), "returncode": completed.returncode, "continuation_fidelity": "blocked_physical_reset_per_skill"}
            if public:
                feedback["public_sensor"] = {"blocked_guard": True, "support_contact": False, "release_contact_free": False, "released": False}
            return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "place", "guarded_abort" if public else "completed", "unknown" if public else "failed", "contact_guard" if public else "goal_not_reached", started_at=started, ended_at=time.time(), safe_exit_complete=completed.returncode == 0, feedback=feedback)
        if public:
            feedback = {
                "public_evidence": "physical place completed; public contact and gripper-state signals are available",
                "metrics_path": str(metrics_path),
                "returncode": completed.returncode,
                "wall_time_s": time.time() - started,
                "public_sensor": public_sensor or {},
            }
            return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "place", "completed" if completed.returncode == 0 else "system_error", "unknown", "system_error" if completed.returncode != 0 else None, started_at=started, ended_at=time.time(), safe_exit_complete=completed.returncode == 0, frames_after=[str(episode_video)] if episode_video.exists() else [], feedback=feedback)
        evidence = f"Isaac M0 placement {placement.get('success')}" if placement else f"Isaac physical-place score {score.get('total', 0)}/{score.get('max', 4)}"
        feedback = {"public_evidence": evidence, "metrics_path": str(metrics_path), "returncode": completed.returncode, "wall_time_s": time.time() - started}
        if placement and not public:
            feedback["placement_score"] = placement
        return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "place", "completed" if completed.returncode == 0 else "system_error", "succeeded" if success else "failed", None if success else "goal_not_reached", started_at=started, ended_at=time.time(), safe_exit_complete=completed.returncode == 0, frames_after=[str(episode_video)] if episode_video.exists() else [], feedback=feedback)

    def retract(self, observation: Observation, out_dir: Path) -> SkillResult:
        return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "retract", "completed", "succeeded", feedback={"public_evidence": "strict launcher completes retract inside place skill"})

    def help(self, observation: Observation, out_dir: Path) -> HelpReport:
        if self.scenario != "blocked" or self.blocker_removed:
            return HelpReport("help_001", "clear_target_area", "hub", "unsupported", "No registered blocker is available for this helper call.", time.time(), time.time(), observation.control_epoch, observation.control_epoch)
        self.blocker_removed = True
        now = time.time()
        return HelpReport("help_001", "clear_target_area", "hub", "succeeded", "The registered target blocker is removed for the next isolated reset; cover still must be placed by the robot. In-process handoff is not yet available in this subprocess adapter.", now, now, observation.control_epoch, observation.control_epoch + 1)

    def close(self) -> None:
        return None


class IsaacPersistentWorkerAdapter:
    """Single-Kit M0 adapter with a real blocked→help→continue handoff.

    The worker owns one Isaac stage for the whole episode.  A blocked place
    writes an event and waits in a physics safe-hold; ``help`` writes the one
    allowed command, and the worker moves only the registered blocker before
    resuming the same dynamic Hub/robot state.  This is intentionally a
    separate backend/config from ``IsaacSubprocessAdapter`` so the older
    reset-per-skill evidence remains reproducible.
    """

    backend_name = "isaac_persistent_worker"

    def __init__(self, *, scenario: str, repo_root: Path, config: dict[str, Any], out_dir: Path, vader_vision: bool = False, rgb_observer: bool = False, initial_rgb_observation: bool = False) -> None:
        self.scenario = scenario
        self.repo_root = repo_root
        self.config = config
        self.out_dir = out_dir
        self.rgb_observer = bool(vader_vision or rgb_observer or initial_rgb_observation)
        self.initial_rgb_observation = bool(initial_rgb_observation)
        self.vader_vision = bool(vader_vision)
        self.replan_after_help = bool(config.get("controller", {}).get("replan_after_help", False) or self.rgb_observer)
        self.place_calls = 0
        self.episode_id = ""
        self._docker_name: str | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._log_handle: Any = None
        self._run_dir: Path | None = None
        self._event_path: Path | None = None
        self._command_path: Path | None = None
        self._event_offset = 0
        self._seen_events: list[dict[str, Any]] = []
        self._blocked_reported = False
        self._helper_sent = False
        self._awaiting_action: str | None = None

    def reset(self, episode_id: str, seed: int, out_dir: Path) -> Observation:
        self.episode_id = episode_id
        self.out_dir = out_dir
        scatter = str(self.config.get("scene", {}).get("initial_layout", "")) == "scatter" or str(self.config.get("scene", {}).get("start_stage", "")) == "scattered"
        observation = Observation(
            episode_id,
            "obs_0000",
            0,
            time.time(),
            held="unknown" if self.rgb_observer else ("no" if scatter else "yes"),
            target_visible="unknown" if self.rgb_observer else "yes",
            evidence="Isaac reset observation is awaiting camera state recognition." if self.rgb_observer else ("Persistent Isaac worker will execute the scattered-table full-physics pick-and-place skill in one physical episode." if scatter else "Persistent Isaac worker will execute the preplace skill in one physical episode."),
            sensor_availability={"rgb": False, "robot_state": True, "contact_or_guard": True},
        )
        if not self.initial_rgb_observation:
            return observation
        self._start_worker()
        event, _events = self._wait_for_event(
            "initial_ready",
            timeout_s=float(self.config.get("budgets", {}).get("episode_wall_timeout_s", 900)) + 60.0,
        )
        frames = self._frame_map(event.get("camera_frame_names", []))
        if not frames:
            raise RuntimeError("Isaac initial RGB observation event did not include readable camera frames")
        self._awaiting_action = "pick" if scatter else "place"
        return Observation(
            episode_id,
            "obs_0000",
            0,
            time.time(),
            frames=frames,
            held="unknown",
            placed="unknown",
            target_visible="unknown",
            release_observed="unknown",
            evidence="Settled Isaac reset state captured before the first robot skill.",
            sensor_availability={"rgb": bool(frames), "robot_state": True, "contact_or_guard": True},
        )

    def _start_worker(self) -> None:
        if self._worker is not None:
            return
        self._run_dir = self.out_dir / "place_01"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._event_path = self._run_dir / "m0_events.jsonl"
        self._command_path = self._run_dir / "m0_command.json"
        script = self.repo_root / "tools" / "run_m1_isaac_strict_success.sh"
        env = os.environ.copy()
        controller = self.config.get("controller", {})
        budgets = self.config.get("budgets", {})
        scene = self.config.get("scene", {})
        scatter = str(scene.get("initial_layout", "")) == "scatter" or str(scene.get("start_stage", "")) == "scattered"
        scatter_supports = bool(scene.get("scatter_physical_supports", False))
        cover_xy = scene.get("cover_initial_xy", [0.55, 0.43])
        casing_xy = scene.get("casing_initial_xy", [0.55, 0.0])
        self._docker_name = f"hac-m0-{self.episode_id}"
        env.update({
            "M1_DOCKER_NAME": self._docker_name,
            "M1_ISAAC_STRICT_OUTPUT_DIR": str(self._run_dir),
            "M1_M0_SERVE": "1",
            "M1_M0_VADER_VQA": "1" if self.vader_vision else "0",
            "M1_M0_REPAIR_RGB": "1" if self.rgb_observer and not self.vader_vision else "0",
            "M1_M0_INITIAL_RGB_OBSERVE": "1" if self.initial_rgb_observation else "0",
            "M1_M0_REPLAN_AFTER_HELP": "1" if self.replan_after_help else "0",
            "M1_M0_STEPWISE_ACTIONS": "1" if controller.get("stepwise_actions", False) or (self.rgb_observer and scatter) else "0",
            "M1_M0_HELP_TIMEOUT_S": str(budgets.get("episode_wall_timeout_s", 900)),
            "M1_SCATTER_RESET": "1" if scatter else "0",
            "M1_PREPLACE_CONTROLLED_PLACE": "0" if scatter else "1",
            "M1_RETRACT_STAGING_SUPPORTS": "1" if scatter and scatter_supports else ("0" if controller.get("m0_defer_staging_support_retraction", False) else "1"),
            "M1_SCATTER_PHYSICAL_SUPPORTS": "1" if scatter_supports else "0",
            "M1_SCATTER_SUPPORT_TOP_Z": str(float(scene.get("scatter_support_top_z", 0.934))),
            "M1_STAGING_SUPPORT_DROP_M": str(controller.get("staging_support_drop_m", 0.15)),
            "M1_STAGING_SUPPORT_RETRACT_STEPS": str(controller.get("staging_support_retract_steps", 10)),
            "M1_STAGING_SUPPORT_WITHDRAW_MODE": str(controller.get("staging_support_withdraw_mode", "down")),
            "M1_EPISODE_LENGTH_S": str(budgets.get("episode_wall_timeout_s", 900)),
            "M1_HUB_RESET_X": str(float(cover_xy[0])) if scatter else "0.55",
            "M1_HUB_RESET_Y": str(float(cover_xy[1])) if scatter else "0.08683718",
            "M1_HUB_RESET_Z": "" if scatter else "1.205",
            "M1_CASING_RESET_X": str(float(casing_xy[0])),
            "M1_CASING_RESET_Y": str(float(casing_xy[1])),
            "M1_TABLE_SIZE_X": str(float(scene.get("table_size_xy", [1.8, 1.8])[0])),
            "M1_TABLE_SIZE_Y": str(float(scene.get("table_size_xy", [1.8, 1.8])[1])),
            "M1_TABLE_TOP_Z": str(float(scene.get("table_top_z", 0.934))),
            "M1_TABLE_CENTER_X": str(float(scene.get("table_center_xy", [1.20, 0.0])[0])),
            "M1_TABLE_CENTER_Y": str(float(scene.get("table_center_xy", [1.20, 0.0])[1])),
            "M1_SCATTER_SPAWN_MARGIN": str(float(scene.get("scatter_spawn_margin_m", 0.002))),
            "M1_GRIP_OPENING": str(controller.get("grip_opening_m", 0.018)),
            "M1_RELEASE_OPENING": str(controller.get("release_opening_m", 0.05)),
            "M1_GRASP_CORRECTION_X_DEG": os.environ.get("M1_GRASP_CORRECTION_X_DEG", "0"),
            "M1_GRASP_CORRECTION_Y_DEG": os.environ.get("M1_GRASP_CORRECTION_Y_DEG", "0"),
            "M1_PREPLACE_RELEASE_CLEARANCE_Y": str(controller.get("m0_release_clearance_y_m", controller.get("preplace_release_clearance_y_m", 0.03))),
            "M1_PREPLACE_SEAT_EXTRA_DEPTH_M": str(controller.get("preplace_seat_extra_depth_m", 0.0)),
            "M1_PREPLACE_CORRECT_ORIENTATION": "1" if controller.get("preplace_correct_orientation", False) else "0",
            "M1_GRIPPER_STATIC_FRICTION": str(controller.get("gripper_static_friction", 4.0)),
            "M1_GRIPPER_DYNAMIC_FRICTION": str(controller.get("gripper_dynamic_friction", 3.0)),
            "M1_TRANSPORT_CLEARANCE_Z": os.environ.get("M1_TRANSPORT_CLEARANCE_Z", str(controller.get("transport_clearance_z_m", 0.0))),
            "M1_TRANSPORT_DIRECT": "1" if controller.get("m0_transport_direct", False) else "0",
            "M1_REANCHOR_AT_INSERTION_ABOVE": "1" if controller.get("m0_reanchor_at_insertion_above", False) else "0",
            "M1_CORRECT_ORIENTATION_AT_INSERTION_ABOVE": "1" if controller.get("m0_correct_orientation_at_insertion_above", False) else "0",
            "M1_FOLLOW_LINK_ORIENTATION": "1" if controller.get("follow_link_orientation", False) else "0",
            "M1_ROTATE_HELD_ORIENTATION": "1" if controller.get("rotate_held_orientation", False) else "0",
            "M1_BLOCKED": "1" if self.scenario == "blocked" else "0",
            "HRC_M1_BLOCKER_PROFILE": os.environ.get("M1_M0_BLOCKER_PROFILE", str(controller.get("m0_blocker_profile", "arm_edge"))) if self.scenario == "blocked" else "default",
            "M1_M0_BLOCKED_ATTEMPT_STEPS": os.environ.get("M1_M0_BLOCKED_ATTEMPT_STEPS", str(controller.get("m0_blocked_attempt_steps", 8))),
            "M1_M0_PRECONTACT_GUARD": "1" if os.environ.get("M1_M0_PRECONTACT_GUARD", "1") == "1" else "0",
            "M1_M0_HELP_SETTLE_STEPS": os.environ.get("M1_M0_HELP_SETTLE_STEPS", str(controller.get("m0_help_settle_steps", 50))),
            "M1_M0_HELP_POLL_STEP_INTERVAL": os.environ.get("M1_M0_HELP_POLL_STEP_INTERVAL", str(controller.get("m0_help_poll_step_interval", 10))),
            "M1_M0_HOLD_OPENING": os.environ.get("M1_M0_HOLD_OPENING", str(controller.get("m0_hold_opening_m", 0.005))),
            "M1_DUAL_GRIPPER": "1" if controller.get("m0_dual_gripper", False) else "0",
            "M1_SYNCHRONOUS_DUAL_LIFT": "1" if controller.get("m0_synchronous_dual_lift", False) else "0",
            "M1_SYNCHRONOUS_DUAL_TRANSPORT": "1" if controller.get("m0_synchronous_dual_transport", False) else "0",
            "M1_RELEASE_RIGHT_AFTER_LIFT": "1" if controller.get("m0_release_right_after_lift", False) else "0",
            "M1_RELEASE_RIGHT_AT_INSERTION_ABOVE": "1" if controller.get("m0_release_right_at_insertion_above", False) else "0",
            "M1_M0_DEFER_STAGING_SUPPORT_RETRACTION": "1" if controller.get("m0_defer_staging_support_retraction", False) else "0",
            "M1_M0_REGRASP_STEPS": str(controller.get("m0_regrasp_steps", 60)),
            "M1_M0_FREEZE_RELEASE_JOINTS": "1" if controller.get("m0_freeze_release_joints", False) else "0",
            "M1_M0_RETRACT_SEGMENTS": str(controller.get("m0_retract_segments", 1)),
            "M1_M0_RETRACT_SEGMENT_STEPS": str(controller.get("m0_retract_segment_steps", 20)),
            "M1_M0_RETRACT_STEP_Z_M": str(controller.get("m0_retract_step_z_m", 0.02)),
            "M1_NO_VIDEO": "1" if os.environ.get("M1_M0_NO_VIDEO", "0") == "1" else ("0" if self.config.get("experiment", {}).get("record_video", True) else "1"),
            "M1_INSERTION_SEGMENTS": "8",
            "M1_INSERTION_SEGMENT_STEPS": "40",
            "M1_HEADLESS": "1",
            "M1_RENDER_INTERVAL": str(controller.get("render_interval", 20)),
            "M1_CAMERA_UPDATE_STRIDE": str(controller.get("camera_update_stride", 20)),
            "M1_INSERT_SAFE_WAYPOINT": "1" if controller.get("insert_safe_waypoint", False) else "0",
            "M1_STRICT_ACCEPTANCE": "0",
            "M1_PLACE_ACCEPTANCE": "1",
            "M1_TRACK_TIP_CONTACT": "0",
        })
        log_path = self._run_dir / "persistent_worker.log"
        self._log_handle = log_path.open("w", encoding="utf-8")
        self._worker = subprocess.Popen(
            [str(script)],
            cwd=self.repo_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _read_events(self) -> list[dict[str, Any]]:
        if self._event_path is None or not self._event_path.exists():
            return []
        lines = self._event_path.read_text(encoding="utf-8").splitlines()
        new: list[dict[str, Any]] = []
        for line in lines[self._event_offset:]:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                new.append(item)
        self._event_offset = len(lines)
        self._seen_events.extend(new)
        return new

    def _wait_for_event(self, name: str, *, timeout_s: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started = time.time()
        events: list[dict[str, Any]] = []
        while time.time() - started <= timeout_s:
            events.extend(self._read_events())
            match = next((event for event in events if event.get("event") == name), None)
            if match is not None:
                return match, events
            if self._worker is not None and self._worker.poll() is not None:
                raise RuntimeError(f"Isaac worker exited before {name}")
            time.sleep(0.2)
        raise TimeoutError(f"Isaac worker did not emit {name} within {timeout_s:.1f}s")

    def _frame_map(self, names: Any) -> dict[str, str]:
        paths = self._rgb_frame_paths(names)
        result: dict[str, str] = {}
        for path in paths:
            stem = Path(path).stem
            alias = next((name for name in ("head_rgb", "left_hand_rgb", "right_hand_rgb") if name in stem), None)
            if alias:
                result[alias] = path
        return result

    def _metrics_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "metrics.json"

    def _wait_for_result(self, *, wait_for_metrics: bool, timeout_s: float) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        started = time.time()
        new_events: list[dict[str, Any]] = []
        while time.time() - started <= timeout_s:
            new_events.extend(self._read_events())
            if any(event.get("event") == "blocked_waiting_help" for event in new_events):
                return "blocked", {}, new_events
            metrics_path = self._metrics_path()
            if wait_for_metrics and metrics_path.exists():
                try:
                    return "complete", json.loads(metrics_path.read_text(encoding="utf-8")), new_events
                except ValueError:
                    pass
            if self._worker is not None and self._worker.poll() is not None:
                if metrics_path.exists():
                    try:
                        return "complete", json.loads(metrics_path.read_text(encoding="utf-8")), new_events
                    except ValueError:
                        pass
                return "system_error", {}, new_events
            time.sleep(0.2)
        return "timeout", {}, new_events

    def _result_from_metrics(self, observation: Observation, metrics: dict[str, Any], *, started: float) -> SkillResult:
        metrics_path = self._metrics_path()
        public = _public_mode(self.config)
        if public:
            success, placement = False, {}
        else:
            success, placement = placement_success(metrics, self.config)
        video = self._run_dir / "inner_wall_probe.mp4" if self._run_dir else None
        if video is not None and video.exists() and not (self._run_dir / "rollout.mp4").exists():
            try:
                (self._run_dir / "rollout.mp4").symlink_to(video.name)
            except OSError:
                pass
        frame_names = (metrics.get("rgb_frame_names") or metrics.get("vader_frame_names", [])) if isinstance(metrics, dict) else []
        frame_paths = self._rgb_frame_paths(frame_names)
        if public:
            feedback = {
                "public_evidence": "physical place completed; public contact and gripper-state signals are available",
                "metrics_path": str(metrics_path),
                "persistent_worker": True,
                "wall_time_s": time.time() - started,
                "public_sensor": _public_contact_summary(metrics),
            }
            return SkillResult(
                observation.episode_id,
                observation.observation_id,
                observation.control_epoch,
                "place",
                "completed",
                "unknown",
                None,
                started_at=started,
                ended_at=time.time(),
                safe_exit_complete=True,
                frames_after=frame_paths if self.rgb_observer else ([str(video)] if video is not None and video.exists() else []),
                feedback=feedback,
            )
        score = metrics.get("physical_place_score", {}) if isinstance(metrics, dict) else {}
        evidence = f"persistent Isaac M0 placement {placement.get('success')}" if placement else f"persistent physical-place score {score.get('total', 0)}/{score.get('max', 4)}"
        return SkillResult(
            observation.episode_id,
            observation.observation_id,
            observation.control_epoch,
            "place",
            "completed" if success else "guarded_abort",
            "succeeded" if success else "failed",
            None if success else "goal_not_reached",
            started_at=started,
            ended_at=time.time(),
            safe_exit_complete=True,
            frames_after=frame_paths if self.rgb_observer else ([str(video)] if video is not None and video.exists() else []),
            feedback={"public_evidence": evidence, "metrics_path": str(metrics_path), "placement_score": placement, "persistent_worker": True, "wall_time_s": time.time() - started},
        )

    def _rgb_frame_paths(self, names: Any) -> list[str]:
        if not self.rgb_observer or self._run_dir is None or not isinstance(names, list):
            return []
        paths = []
        for name in names:
            filename = Path(str(name)).name
            if filename != str(name):
                continue
            frame_dir = "rgb_frames" if (self._run_dir / "rgb_frames").is_dir() else "vader_frames"
            path = self._run_dir / frame_dir / filename
            if path.is_file():
                paths.append(str(path))
        return paths

    def _send_action(self, action: str) -> None:
        if self._awaiting_action == action and self._command_path is not None:
            self._command_path.write_text(json.dumps({"action": action}), encoding="utf-8")
            self._awaiting_action = None

    def pick(self, observation: Observation, out_dir: Path) -> SkillResult:
        started = time.time()
        self._start_worker()
        if self._awaiting_action != "pick":
            return SkillResult(
                observation.episode_id, observation.observation_id, observation.control_epoch,
                "pick", "guarded_abort", "failed", "skill_precondition",
                started_at=started, ended_at=time.time(), safe_exit_complete=True,
                feedback={"public_evidence": "The persistent worker is not waiting for a pick skill."},
            )
        self._send_action("pick")
        timeout_s = float(self.config.get("budgets", {}).get("episode_wall_timeout_s", 900)) + 60.0
        try:
            event, _events = self._wait_for_event("pick_ready_for_place", timeout_s=timeout_s)
        except (RuntimeError, TimeoutError) as exc:
            return SkillResult(
                observation.episode_id, observation.observation_id, observation.control_epoch,
                "pick", "system_error", "unknown", "system_error",
                started_at=started, ended_at=time.time(), safe_exit_complete=False,
                feedback={"public_evidence": f"Isaac worker did not reach the pick observation boundary: {type(exc).__name__}"},
            )
        frames = self._rgb_frame_paths(event.get("camera_frame_names", []))
        if self.rgb_observer and not frames:
            return SkillResult(
                observation.episode_id, observation.observation_id, observation.control_epoch,
                "pick", "system_error", "unknown", "missing_rgb_frames",
                started_at=started, ended_at=time.time(), safe_exit_complete=False,
                feedback={"public_evidence": "Pick boundary was reached without readable RGB frames."},
            )
        self._awaiting_action = "place"
        return SkillResult(
            observation.episode_id, observation.observation_id, observation.control_epoch,
            "pick", "completed", "unknown", None,
            started_at=started, ended_at=time.time(), safe_exit_complete=True,
            frames_after=frames,
            feedback={"public_evidence": "Controller reached the post-lift hold; RGB state recognition determines whether the cover is grasped."},
        )

    def place(self, observation: Observation, out_dir: Path) -> SkillResult:
        started = time.time()
        self.place_calls += 1
        self._start_worker()
        if self._awaiting_action == "pick":
            return SkillResult(
                observation.episode_id, observation.observation_id, observation.control_epoch,
                "place", "guarded_abort", "failed", "skill_precondition",
                started_at=started, ended_at=time.time(), safe_exit_complete=True,
                feedback={"public_evidence": "Place requires a completed and observed pick in the scattered task."},
            )
        self._send_action("place")
        timeout_s = float(self.config.get("budgets", {}).get("episode_wall_timeout_s", 900)) + 60.0
        if self._blocked_reported:
            status, metrics, _events = self._wait_for_result(wait_for_metrics=True, timeout_s=timeout_s)
        else:
            status, metrics, _events = self._wait_for_result(wait_for_metrics=True, timeout_s=timeout_s)
        if status == "blocked":
            self._blocked_reported = True
            feedback = {"public_evidence": "registered target blocker stopped the physical descent; worker remains in safe hold", "persistent_worker": True, "m0_event": "blocked_waiting_help"}
            if _public_mode(self.config):
                feedback["public_sensor"] = {"blocked_guard": True, "support_contact": False, "release_contact_free": False, "released": False}
            frame_names = next((event.get("camera_frame_names", []) for event in _events if event.get("event") == "blocked_waiting_help"), [])
            return SkillResult(
                observation.episode_id,
                observation.observation_id,
                observation.control_epoch,
                "place",
                "guarded_abort",
                "failed",
                "contact_guard",
                started_at=started,
                ended_at=time.time(),
                safe_exit_complete=True,
                frames_after=self._rgb_frame_paths(frame_names),
                feedback=feedback,
            )
        if status == "complete":
            return self._result_from_metrics(observation, metrics, started=started)
        return SkillResult(
            observation.episode_id,
            observation.observation_id,
            observation.control_epoch,
            "place",
            "system_error" if status == "system_error" else "guarded_abort",
            "unknown",
            "system_error" if status == "system_error" else "motion_timeout",
            started_at=started,
            ended_at=time.time(),
            safe_exit_complete=False,
            feedback={"public_evidence": f"persistent Isaac worker ended with status {status}", "persistent_worker": True},
        )

    def retract(self, observation: Observation, out_dir: Path) -> SkillResult:
        return SkillResult(observation.episode_id, observation.observation_id, observation.control_epoch, "retract", "completed", "succeeded", feedback={"public_evidence": "persistent worker includes the validated retract"})

    def help(self, observation: Observation, out_dir: Path) -> HelpReport:
        if not self._blocked_reported or self._command_path is None:
            return HelpReport("help_001", "clear_target_area", "hub", "unsupported", "No blocked persistent worker is waiting for the registered helper.", time.time(), time.time(), observation.control_epoch, observation.control_epoch)
        now = time.time()
        self._command_path.write_text(json.dumps({"action": "help", "request_type": "clear_target_area", "target": "hub"}), encoding="utf-8")
        deadline = now + float(self.config.get("helper", {}).get("post_intervention_settle_s", 0.5)) + 30.0
        applied = False
        while time.time() < deadline:
            for event in self._read_events():
                if event.get("event") == "helper_applied":
                    applied = True
                    frame_names = event.get("camera_frame_names", [])
                    frames = self._frame_map(frame_names)
            if applied:
                self._helper_sent = True
                self._awaiting_action = "place" if self.replan_after_help else None
                return HelpReport("help_001", "clear_target_area", "hub", "succeeded", "The helper moved only the registered blocker; the robot remains in the same Isaac episode and must replan before continuing.", now, time.time(), observation.control_epoch, observation.control_epoch + 1, frames)
            time.sleep(0.2)
        return HelpReport("help_001", "clear_target_area", "hub", "failed", "Persistent Isaac worker did not acknowledge the registered blocker removal.", now, time.time(), observation.control_epoch, observation.control_epoch)

    def close(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            try:
                self._worker.terminate()
                self._worker.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self._worker.kill()
                except OSError:
                    pass
        if self._docker_name:
            try:
                subprocess.run(
                    ["docker", "stop", "--timeout", "10", self._docker_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._worker = None


__all__ = ["ContractAdapter", "IsaacSubprocessAdapter", "IsaacPersistentWorkerAdapter"]
