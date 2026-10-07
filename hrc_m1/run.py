"""Command-line entry point for the M1 contract and simulator runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contracts import Decision, EpisodeMode, EpisodeState, stable_hash
from .evaluator import IndependentSeatEvaluator, SeatTolerances
from .logger import EventLogger
from .observer import ObserverVerifier, VLMVerdict
from .planner import HttpPlanner, PlannerUnavailable, RulePlanner
from .state_machine import M1StateMachine
from .task_adapter import MockTaskAdapter
from .vader import assessment_context, expected_outcome


def _git(path: str, args: list[str]) -> str:
    try:
        # The pinned Isaac container runs as root while the mounted checkout
        # is owned by the workstation user.  Tell Git exactly which mounted
        # repository is trusted; do not weaken the global safe-directory list.
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path}", "-C", path, *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the M1 task config") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("task config must be a YAML object")
    return data


def _manifest(config_path: Path, config: dict[str, Any], episode_id: str, mode: str, seed: int, args: argparse.Namespace | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    assets = [root / "assets" / "parts" / "Hub Cover Output.usd", root / "assets" / "parts" / "Casing Top.usd"]
    planner_endpoint = getattr(args, "planner_endpoint", None) if args is not None else None
    planner_model = getattr(args, "planner_model", None) if args is not None else None
    vlm_endpoint = getattr(args, "vlm_endpoint", None) if args is not None else None
    vlm_model = getattr(args, "vlm_model", None) if args is not None else None
    planner_kind = getattr(args, "planner", "unknown") if args is not None else "unknown"
    fault_profile = config.get("reset", {}).get("fault_profiles", {}).get(mode, {})
    return {
        "episode_id": episode_id,
        "mode": mode,
        "seed": int(seed),
        "fault_profile_public_hash": stable_hash(fault_profile),
        "task_id": config.get("task_id"),
        "pipeline": getattr(args, "pipeline", "m1") if args is not None else "m1",
        "interface_id": config.get("interface_id"),
        "sequence_step": config.get("sequence_step"),
        "canonical_instance": config.get("canonical_instance"),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "asset_sha256": {str(p.relative_to(root)): _hash_file(p) for p in assets},
        "main_repo": {"head": _git(str(root), ["rev-parse", "HEAD"]), "dirty": bool(_git(str(root), ["status", "--porcelain"]))},
        "roco_repo": {"head": _git(os.environ.get("ROCO_ROOT", "/home/sunsiliang/roco_runtime/gearboxAssembly"), ["rev-parse", "HEAD"]), "dirty": bool(_git(os.environ.get("ROCO_ROOT", "/home/sunsiliang/roco_runtime/gearboxAssembly"), ["status", "--porcelain"]))},
        "public_sensor_contract": config.get("public_sensors", []),
        "model_boundary": {
            "planner_kind": planner_kind,
            "planner_configured": bool(planner_endpoint or os.environ.get("HRC_M1_PLANNER_ENDPOINT")),
            "planner_model": planner_model or os.environ.get("HRC_M1_PLANNER_MODEL"),
            "vlm_configured": bool(vlm_endpoint or os.environ.get("HRC_M1_VLM_ENDPOINT")),
            "vlm_model": vlm_model or os.environ.get("HRC_M1_VLM_MODEL"),
            "decision_schema": "m1-decision-v1",
            "event_prompt_hashes": "events.jsonl:decision.metadata.prompt_hash and verification.prompt_hash",
        },
        "provenance": {
            "agent_evaluator_boundary": "evaluator_private.jsonl",
            "fault_profile_private_record": "evaluator_private.jsonl:fault_profile",
            "magic_assembly": "forbidden_in_episode",
        },
    }


def _tolerances(config: dict[str, Any]) -> SeatTolerances:
    physics = config.get("physics", {})
    return SeatTolerances(
        axial_depth_m=float(physics.get("axial_depth_m", 0.062)),
        axial_tolerance_m=float(physics.get("axial_tolerance_m", 0.008)),
        radial_tolerance_m=float(physics.get("radial_tolerance_m", 0.004)),
        tilt_tolerance_deg=float(physics.get("tilt_tolerance_deg", 2.0)),
        max_penetration_m=float(physics.get("max_penetration_m", 0.001)),
        settle_speed_mps=float(physics.get("settle_speed_mps", 0.01)),
        settle_window_s=float(physics.get("settle_window_s", 0.5)),
    )


def _decision(planner: Any, observation: Any, state: EpisodeState) -> tuple[Decision, dict[str, Any]]:
    return planner.next_decision(observation, state)


def _run_contract(config_path: Path, config: dict[str, Any], args: argparse.Namespace, out: Path) -> int:
    episode_id = f"m1_{args.mode}_{args.seed}_{int(time.time())}"
    manifest = _manifest(config_path, config, episode_id, args.mode, args.seed, args)
    manifest["backend"] = "mock_contract_only"
    with EventLogger(out, manifest) as logger:
        adapter = MockTaskAdapter()
        evaluator = IndependentSeatEvaluator(_tolerances(config), calibration_status="pending")
        sm = M1StateMachine(episode_id, replan_on_visual_failure=args.pipeline == "vader")
        sm.reset()
        obs = adapter.reset(episode_id, args.seed, args.mode)
        sm.accept_observation(obs)
        logger.event("reset", {"state": sm.state, "observation_id": obs.observation_id, "backend": adapter.backend_name})
        planner = RulePlanner()
        try:
            decision, meta = _decision(planner, obs, sm.state)
            sm.accept_decision(decision)
            logger.event("decision", {"decision": decision, "metadata": meta, "state": sm.state})
            if decision.action in {"PICK", "MOVE_TO_PREINSERT", "GUARDED_INSERT", "RELEASE_RETRACT"}:
                skill = {"PICK": "pick", "MOVE_TO_PREINSERT": "preinsert", "GUARDED_INSERT": "insert", "RELEASE_RETRACT": "release_retract"}[decision.action]
                sm.skill_started(skill)
                result = adapter.execute_skill(obs, skill, decision.skill_args)
                sm.skill_finished(result)
                logger.event("skill_finished", {"result": result, "state": sm.state})
                expected = expected_outcome(config, skill) if args.pipeline == "vader" else None
                verification = ObserverVerifier().verify(obs, expected_outcome=expected)
                logger.event("verification", {"verdict": verification.verdict, "evidence": verification.evidence, "prompt_hash": verification.prompt_hash, "response_hash": verification.response_hash})
                sm.verify(verification.verdict.value)
            else:
                sm.safe_stop("contract_backend_does_not_execute_model_decisions")
        except Exception as exc:
            sm.safe_stop(f"contract_run:{type(exc).__name__}")
            logger.event("error", {"error_type": type(exc).__name__})
        result = evaluator.evaluate(adapter.evaluator_measurement())
        logger.evaluator_event("evaluation", result.to_public_dict())
        logger.write_metrics({"run_status": "MOCK_ONLY", "task_success": False, "pipeline_fidelity": "contract_only_vader" if args.pipeline == "vader" else "contract_only", "state": sm.state, "evaluator_status": result.status})
        logger.event("completed", {"run_status": "MOCK_ONLY", "state": sm.state})
    adapter.close()
    return 0


def _run_roco(config_path: Path, config: dict[str, Any], args: argparse.Namespace, out: Path, simulation_app: Any) -> int:
    import torch
    from .roco_adapter import RocoTaskAdapter
    from .roco_env import make_env_classes

    episode_id = f"m1_{args.mode}_{args.seed}_{int(time.time())}"
    manifest = _manifest(config_path, config, episode_id, args.mode, args.seed, args)
    manifest["backend"] = "roco_isaaclab_m1"
    manifest["video"] = "episode.mp4"
    with EventLogger(out, manifest) as logger:
        cfg_cls, env_cls = make_env_classes()
        env_cfg = cfg_cls()
        # DirectRLEnv reports a missing seed if it is assigned only after
        # scene construction.  Set it before creating Isaac so scene and
        # reset RNGs share the recorded episode seed.
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = int(args.seed)
        env = env_cls(env_cfg)
        fault_profile = config.get("reset", {}).get("fault_profiles", {}).get(args.mode, {})
        adapter = RocoTaskAdapter(env, frame_dir=out / "frames", video_path=out / "episode.mp4", fault_profile=fault_profile)
        evaluator = IndependentSeatEvaluator(_tolerances(config), calibration_status=str(config.get("physics", {}).get("calibration_status", "pending")))
        sm = M1StateMachine(episode_id, replan_on_visual_failure=args.pipeline == "vader")
        sm.reset()
        obs = adapter.reset(episode_id, args.seed, args.mode)
        sm.accept_observation(obs)
        logger.evaluator_event("fault_profile", fault_profile)
        logger.event("reset", {"state": sm.state, "observation_id": obs.observation_id, "backend": adapter.backend_name, "sensor_availability": obs.sensor_availability})
        if args.smoke:
            adapter._step(args.smoke_steps)
            obs = adapter.observe(episode_id, args.smoke_steps)
            logger.event("sim_smoke", {"steps": args.smoke_steps, "observation_id": obs.observation_id, "sensor_availability": obs.sensor_availability})
            logger.write_metrics({"run_status": "SIM_SMOKE_ONLY", "task_success": False, "pipeline_fidelity": "reset_camera_step_only", "state": sm.state})
            adapter.close()
            return 0
        if args.physics_trial:
            trial_velocity = float(
                args.physics_velocity
                if args.physics_velocity is not None
                else config.get("physics", {}).get("velocity_mps", -0.008)
            )
            adapter.prepare_free_seat_trial(velocity_mps=trial_velocity)
            physics_steps = int(
                args.physics_steps
                if args.physics_steps is not None
                else config.get("physics", {}).get("steps", 1930)
            )
            adapter._step(max(1, physics_steps // int(env_cfg.decimation)))
            measurement = adapter.evaluator_measurement()
            evaluation = evaluator.evaluate(measurement)
            logger.evaluator_event("evaluation", evaluation.to_public_dict())
            logger.evaluator_event("physics_trial", {"physics_steps": physics_steps, "velocity_mps": trial_velocity, "measurement": evaluation.metrics})
            logger.write_metrics({"run_status": "PHYSICS_TRIAL_ONLY", "task_success": False, "pipeline_fidelity": "collision_on_interface_trial", "evaluator_status": evaluation.status, "physical_criteria_passed": evaluation.physical_criteria_passed})
            logger.event("completed", {"run_status": "PHYSICS_TRIAL_ONLY", "evaluator_status": evaluation.status})
            adapter.close()
            return 0
        if args.skill_smoke:
            skill_status = "PASS_TRACE"
            for skill in ("pick", "preinsert", "insert", "verify", "release_retract"):
                result = adapter.execute_skill(obs, skill, {})
                logger.event("skill_smoke_finished", {"result": result})
                if result.motion_status != "SUCCEEDED":
                    skill_status = "FAIL_TRACE"
                    break
                obs = adapter.observe(episode_id, adapter.index, result.to_dict())
            measurement = adapter.evaluator_measurement()
            evaluation = evaluator.evaluate(measurement)
            logger.evaluator_event("evaluation", evaluation.to_public_dict())
            logger.write_metrics({"run_status": "SKILL_SMOKE_ONLY", "skill_trace_status": skill_status, "task_success": False, "pipeline_fidelity": "low_level_skill_trace", "evaluator_status": evaluation.status})
            logger.event("completed", {"run_status": "SKILL_SMOKE_ONLY", "skill_trace_status": skill_status})
            adapter.close()
            return 0 if skill_status == "PASS_TRACE" else 2
        planner: Any
        if args.planner == "rule":
            planner = RulePlanner()
        else:
            endpoint = args.planner_endpoint or os.environ.get("HRC_M1_PLANNER_ENDPOINT")
            if not endpoint:
                logger.event("blocked", {"reason": "planner_endpoint_not_configured"})
                logger.write_metrics({"run_status": "BLOCKED", "task_success": False, "pipeline_fidelity": "not_started", "state": sm.state})
                adapter.close()
                return 2
            planner = HttpPlanner(endpoint, args.planner_model or os.environ.get("HRC_M1_PLANNER_MODEL", ""), pipeline=args.pipeline)
        verifier = ObserverVerifier(args.vlm_endpoint or os.environ.get("HRC_M1_VLM_ENDPOINT"), args.vlm_model or os.environ.get("HRC_M1_VLM_MODEL"))
        status = "INCOMPLETE"
        recent_skill: dict[str, Any] | None = None
        for _turn in range(args.max_turns):
            try:
                refresh_states = {EpisodeState.REOBSERVE, EpisodeState.OBSERVE}
                if args.pipeline == "vader":
                    refresh_states.add(EpisodeState.REPLAN)
                if sm.state in refresh_states:
                    obs = adapter.observe(episode_id, adapter.index, recent_skill)
                    sm.accept_observation(obs)
                    logger.event("observation", {"observation_id": obs.observation_id, "sensor_availability": obs.sensor_availability})
                if sm.state == EpisodeState.UPDATE_MEMORY:
                    sm.update_memory()
                    logger.event("memory_updated", {"observation_id": sm.current_observation_id, "state": sm.state})
                    continue
                if sm.state in {EpisodeState.SAFE_HOLD, EpisodeState.ASK_HUMAN}:
                    if args.mode == "fault_hil":
                        if args.interactive_hil:
                            from .human_help import HumanBroker
                            if HumanBroker(adapter, logger).run(sm):
                                continue
                            status = "ABORT"
                            break
                        sm.request_help("The guarded insertion did not establish a verified postcondition.") if sm.state == EpisodeState.SAFE_HOLD else None
                        logger.event("waiting_for_operator", {"state": sm.state, "message": "No real operator is attached to this run."})
                        status = "WAITING_FOR_OPERATOR"
                        break
                    sm.abort("no-human safety baseline")
                    status = "SAFE_ABORT"
                    break
                if sm.state in {EpisodeState.DONE, EpisodeState.ABORT}:
                    status = "DONE" if sm.state == EpisodeState.DONE else "ABORT"
                    break
                decision, meta = _decision(planner, obs, sm.state)
                sm.accept_decision(decision)
                logger.event("decision", {"decision": decision, "metadata": meta, "state": sm.state})
                if decision.action == "ASK_HUMAN":
                    sm.request_help(decision.ask_message or "Planner requested human help")
                    continue
                if decision.action == "ABORT":
                    status = "ABORT"
                    break
                if decision.action == "OBSERVE":
                    sm._set(EpisodeState.REOBSERVE, "planner_requested_observe")
                    continue
                if decision.action == "REPLAN":
                    continue
                if decision.action == "VERIFY":
                    expected = expected_outcome(config, "verify") if args.pipeline == "vader" else None
                    verification = verifier.verify(obs, expected_outcome=expected)
                    logger.event("verification", {"verdict": verification.verdict, "evidence": verification.evidence, "expected_outcome": expected, "prompt_hash": verification.prompt_hash, "response_hash": verification.response_hash})
                    result_context = {"skill_id": "verify", "motion_status": "SUCCEEDED"}
                    recent_skill = assessment_context(result_context, expected, verification.verdict.value, verification.evidence) if expected else result_context if verification.verdict == VLMVerdict.SUCCESS else None
                    sm.verify(verification.verdict.value)
                    continue
                skill = {"PICK": "pick", "MOVE_TO_PREINSERT": "preinsert", "GUARDED_INSERT": "insert", "RELEASE_RETRACT": "release_retract"}.get(decision.action)
                if skill is None:
                    sm.safe_stop("unsupported_decision")
                    continue
                sm.skill_started(skill)
                result = adapter.execute_skill(obs, skill, decision.skill_args)
                sm.skill_finished(result)
                logger.event("skill_finished", {"result": result, "state": sm.state})
                expected = expected_outcome(config, skill) if args.pipeline == "vader" else None
                verification = verifier.verify(adapter.observe(episode_id, adapter.index, result.to_dict()), expected_outcome=expected)
                logger.event("verification", {"verdict": verification.verdict, "evidence": verification.evidence, "expected_outcome": expected, "prompt_hash": verification.prompt_hash, "response_hash": verification.response_hash})
                recent_skill = assessment_context(result.to_dict(), expected, verification.verdict.value, verification.evidence) if expected else result.to_dict()
                verdict = verification.verdict.value
                if args.pipeline == "vader" and result.motion_status != "SUCCEEDED":
                    verdict = "UNKNOWN"
                sm.verify(verdict)
            except PlannerUnavailable as exc:
                sm.safe_stop("planner_unavailable")
                logger.event("blocked", {"reason": str(exc)})
                status = "BLOCKED"
                break
            except Exception as exc:
                sm.safe_stop(f"runtime:{type(exc).__name__}")
                logger.event("error", {"error_type": type(exc).__name__})
                status = "INCOMPLETE"
                break
        result = evaluator.evaluate(adapter.evaluator_measurement())
        logger.evaluator_event("evaluation", result.to_public_dict())
        logger.write_metrics({"run_status": status, "task_success": result.task_success and status == "DONE", "pipeline_fidelity": "vader_plan_execute_detect" if args.pipeline == "vader" else "roco_adapter", "state": sm.state, "evaluator_status": result.status, "turns": sm.turns})
        logger.event("completed", {"run_status": status, "state": sm.state})
        adapter.close()
    return 0 if status in {"DONE", "WAITING_FOR_OPERATOR", "SAFE_ABORT", "SIM_SMOKE_ONLY"} else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 gearbox HRC runner")
    parser.add_argument("--task-config", default="m1_hub_cover_output_top_seat.yaml")
    parser.add_argument("--mode", choices=[m.value for m in EpisodeMode], default="nominal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--backend", choices=["roco", "mock"], default=None)
    parser.add_argument("--planner", choices=["http", "rule"], default="http")
    parser.add_argument("--pipeline", choices=["m1", "vader"], default="m1")
    parser.add_argument("--planner-endpoint", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--vlm-endpoint", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--skill-smoke", action="store_true", help="Run fixed low-level skills for trace debugging; never claims M1 success")
    parser.add_argument("--physics-trial", action="store_true", help="Run reset-only collision-on seating trial; never claims robot task success")
    parser.add_argument("--physics-velocity", type=float, default=None, help="Override reset-only physics-trial insertion speed (m/s)")
    parser.add_argument("--physics-steps", type=int, default=None, help="Override reset-only physics-trial duration in physics steps")
    parser.add_argument("--interactive-hil", action="store_true", help="Read real operator JSON commands from stdin after SAFE_HOLD")
    return parser


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=["roco", "mock"], default=None)
    known, _ = pre.parse_known_args(argv)
    parser = _parser()
    simulation_app = None
    if known.backend != "mock":
        try:
            from isaaclab.app import AppLauncher
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args(argv)
            simulation_app = AppLauncher(args).app
        except ImportError as exc:
            parser.error(f"RoCo backend requires Isaac Lab; use --backend mock for contract tests: {exc}")
    else:
        args = parser.parse_args(argv)
    config_path = Path(args.task_config).resolve()
    config = _load_config(config_path)
    if args.seed is None:
        args.seed = int(config.get("seeds", {}).get("fault" if args.mode != "nominal" else "nominal", 1201))
    args.backend = args.backend or config.get("backend", "roco")
    out = Path(args.out or (Path("runs") / f"{args.mode}_{args.seed}_{int(time.time())}")).resolve()
    try:
        if args.backend == "mock":
            return _run_contract(config_path, config, args, out)
        return _run_roco(config_path, config, args, out, simulation_app)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
