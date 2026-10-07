"""M0 reset → observe → skill/help → reobserve → finish runner."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
import time
from pathlib import Path
from typing import Any

from .adapter import ContractAdapter, IsaacPersistentWorkerAdapter, IsaacSubprocessAdapter
from .config import load_config, preflight, repo_root, resolved_config
from .contracts import HelpReport, Observation, PlannerAction, SkillResult
from .evaluator import evaluate_artifact
from .logger import RepairLogger
from .observer import MetricsObserver
from .planner import AutoPlanner, FixedRetryPlanner, HttpRepairPlanner, PlannerUnavailable, RepairRulePlanner
from .state_recognizer import (
    assess_help,
    assess_initial,
    assess_pick,
    assess_place,
    expected_initial_state,
    expected_pick_outcome,
    expected_place_outcome,
    planner_observation,
    visual_success,
)

_PUBLIC_FAILURE_CODES = frozenset({
    "contact_guard", "helper_unsupported", "missing_rgb_frames", "motion_timeout",
    "no_metrics_artifact", "skill_precondition", "system_error",
})


def _public_skill_result(result: SkillResult) -> SkillResult:
    """Keep evaluator-derived failure labels out of online observations."""
    import dataclasses

    feedback = {}
    if "public_evidence" in result.feedback:
        feedback["public_evidence"] = result.feedback["public_evidence"]
    public_sensor = result.feedback.get("public_sensor")
    if isinstance(public_sensor, dict):
        allowed_sensors = {"blocked_guard", "release_contact_free", "released", "support_contact"}
        feedback["public_sensor"] = {
            key: bool(public_sensor[key]) for key in allowed_sensors if key in public_sensor
        }
    return dataclasses.replace(
        result,
        outcome="unknown" if result.skill_id == "place" else result.outcome,
        failure_code=result.failure_code if result.failure_code in _PUBLIC_FAILURE_CODES else None,
        frames_before=[],
        frames_after=[],
        feedback=feedback,
    )


def _out_root(config: dict[str, Any], config_path: Path) -> Path:
    value = config.get("experiment", {}).get("output_root", "runs/repair_m0")
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def _planner(config: dict[str, Any], method: str):
    model = config.get("model", {})
    endpoint = os.environ.get("HRC_REPAIR_PLANNER_ENDPOINT")
    if method in {"repair", "vader"}:
        if not endpoint:
            if not bool(model.get("allow_rule_fallback", False)):
                raise PlannerUnavailable(f"{method} requires HRC_REPAIR_PLANNER_ENDPOINT; rule_dev_only fallback is disabled")
            return RepairRulePlanner(allow_help=True, retries_before_help=int(config.get("fixed_retry", {}).get("retries_before_help", 1)))
        return HttpRepairPlanner(
            endpoint,
            os.environ.get("HRC_REPAIR_PLANNER_MODEL", str(model.get("model_id", ""))),
            api_key_env=str(model.get("api_key_env_name", "HRC_REPAIR_API_KEY")),
            timeout_s=float(model.get("request_timeout_s", 30)),
            allow_help=True,
            prompt_path=(
                "hrc_repair/prompts/vader_m0_planner_prompt.txt"
                if method == "vader"
                else str(model.get("planner_prompt", ""))
            ),
        )
    if endpoint and method != "auto":
        return HttpRepairPlanner(endpoint, os.environ.get("HRC_REPAIR_PLANNER_MODEL", str(model.get("model_id", ""))), api_key_env=str(model.get("api_key_env_name", "HRC_REPAIR_API_KEY")), timeout_s=float(model.get("request_timeout_s", 30)), allow_help=False)
    if method == "auto":
        return AutoPlanner()
    if method == "fixed_retry":
        return FixedRetryPlanner()
    # A deterministic rule planner is useful for a runnable protocol smoke,
    # but is explicitly marked in model_calls and reports, never presented as
    # an LLM result.
    return RepairRulePlanner(allow_help=True, retries_before_help=int(config.get("fixed_retry", {}).get("retries_before_help", 1)))


def _fresh_action(planner, observation: Observation, history: list[dict[str, Any]], budgets: dict[str, int], epoch: int, *, retry_invalid_output: bool):
    retried = False
    try:
        action, meta = planner.next_action(observation, history, budgets)
    except PlannerUnavailable as exc:
        format_error = isinstance(exc.__cause__, (ValueError, TypeError, KeyError, IndexError))
        if not retry_invalid_output or not format_error:
            raise
        correction = [{"protocol_error": "The previous response was not one valid action object. Return exactly one JSON object and copy observation_id and control_epoch from the current observation."}]
        action, meta = planner.next_action(observation, correction, budgets)
        meta["action_format_retry"] = True
        retried = True
    if action.observation_id != observation.observation_id or int(action.control_epoch) != epoch:
        if retry_invalid_output and not retried:
            correction = [{"protocol_error": "The previous action was discarded because its IDs were stale. Re-evaluate this observation and copy its observation_id and control_epoch exactly."}]
            action, meta = planner.next_action(observation, correction, budgets)
            meta["stale_action_retry"] = True
        if action.observation_id != observation.observation_id or int(action.control_epoch) != epoch:
            raise ValueError(
                f"stale observation_id/control_epoch: received {action.observation_id}/{action.control_epoch}; "
                f"expected {observation.observation_id}/{epoch}"
            )
    return action, meta


def _scatter_action_error(action, observation: Observation, *, place_once: bool = False) -> str | None:
    if action.action not in {"pick", "place", "help", "finish", "stop"}:
        return f"unsupported action {action.action}"
    if action.action == "pick" and observation.held != "no":
        return "pick requires held=no; the observation does not confirm an unheld cover"
    if action.action == "place" and observation.held != "yes":
        return "place requires held=yes; the observation does not confirm a grasp"
    recent = observation.recent_skill or {}
    if place_once and action.action == "place" and recent.get("skill_id") == "place":
        return "the live episode has already used its placement attempt"
    public_sensor = (recent.get("feedback") or {}).get("public_sensor", {})
    if action.action == "help" and (observation.held != "yes" or not public_sensor.get("blocked_guard")):
        return "help requires held=yes and public confirmation of a guarded placement block"
    if observation.placed == "yes" and observation.release_observed == "yes" and action.action not in {"finish", "stop"}:
        return "only finish or stop is valid after confirmed placement and release"
    return None


def _scatter_recovery_action(observation: Observation, *, help_budget: int) -> str:
    recent = observation.recent_skill or {}
    public_sensor = (recent.get("feedback") or {}).get("public_sensor", {})
    if observation.placed == "yes" and observation.release_observed == "yes":
        return "finish"
    if observation.held == "yes" and observation.placed == "no":
        if recent.get("skill_id") == "place" and public_sensor.get("blocked_guard") and help_budget > 0:
            return "help"
        return "place"
    if observation.held == "no" and observation.placed == "no" and recent.get("skill_id") not in {"pick", "place"}:
        return "pick"
    return "stop"


def _safe_stop_action(observation: Observation) -> PlannerAction:
    return PlannerAction(observation.observation_id, observation.control_epoch, "stop", {})


def _adapter(
    config: dict[str, Any], scenario: str, out_dir: Path, *,
    vader_vision: bool = False, rgb_observer: bool = False,
    initial_rgb_observation: bool = False,
):
    backend = str(config.get("experiment", {}).get("backend", "contract_debug"))
    if backend == "isaac_subprocess":
        return IsaacSubprocessAdapter(scenario=scenario, repo_root=repo_root(), config=config, out_dir=out_dir)
    if backend == "isaac_inprocess":
        return IsaacPersistentWorkerAdapter(
            scenario=scenario, repo_root=repo_root(), config=config, out_dir=out_dir,
            vader_vision=vader_vision, rgb_observer=rgb_observer,
            initial_rgb_observation=initial_rgb_observation,
        )
    return ContractAdapter(scenario=scenario)


def run_episode(config_path: Path, config: dict[str, Any], *, method: str, scenario: str, seed: int, out_dir: Path | None = None) -> dict[str, Any]:
    # The planner receives Observation.episode_id.  Keep it opaque so method,
    # scenario, and seed cannot leak into the model input; those remain in the
    # private manifest/evaluator fields below.
    episode_id = f"episode_{uuid.uuid4().hex[:16]}"
    out_dir = out_dir or (_out_root(config, config_path) / method / scenario / f"seed_{seed:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml_dump(resolved_config(config_path, config)), encoding="utf-8")
    check = preflight(config_path, config, strict=False)
    rgb_repair = method == "repair" and bool(config.get("model", {}).get("rgb_observer_enabled", False))
    vader_scattered = method == "vader" and str(config.get("scene", {}).get("start_stage", "")) == "scattered"
    scene = config.get("scene", {})
    scatter_start = str(scene.get("initial_layout", "")) == "scatter" or str(scene.get("start_stage", "")) == "scattered"
    stepwise_actions = bool(config.get("controller", {}).get("stepwise_actions", False) or ((rgb_repair or method == "vader") and scatter_start))
    manifest = {
        "episode_id": episode_id,
        "method": method,
        "scenario_private": scenario,
        "seed": int(seed),
        "backend": config.get("experiment", {}).get("backend"),
        "isaac_container_ipc_mode": os.environ.get("M1_ISAAC_IPC_MODE", "host"),
        "feedback_mode": config.get("experiment", {}).get("feedback_mode"),
        "model_id": config.get("model", {}).get("model_id"),
        "pipeline": "repair_m0_rgb_state_recognition_replan" if rgb_repair else "vader_m0_scattered_pick_place_detect" if vader_scattered else "vader_m0_plan_execute_detect" if method == "vader" else "repair_m0",
        "rgb_state_recognizer": rgb_repair or method == "vader",
        "replan_after_help": bool(config.get("controller", {}).get("replan_after_help", False) or method == "vader" or rgb_repair),
        "stepwise_skill_observations": stepwise_actions,
        "preflight": check,
        "public_task_id": config.get("experiment", {}).get("name", "repair_m0_cover_to_hub"),
        "scenario_exposure": "private evaluator only; absent from planner observation and model payload",
    }
    observer = MetricsObserver(feedback_mode=str(config.get("experiment", {}).get("feedback_mode", "privileged_debug")))
    adapter = _adapter(
        config, scenario, out_dir,
        vader_vision=method == "vader",
        rgb_observer=rgb_repair,
        initial_rgb_observation=rgb_repair or vader_scattered,
    )
    planner = _planner(config, method)
    vision_verifier = None
    if method == "vader" or rgb_repair:
        from hrc_m1.observer import ObserverVerifier

        model = config.get("model", {})
        endpoint = os.environ.get("HRC_REPAIR_VLM_ENDPOINT", os.environ.get("HRC_REPAIR_PLANNER_ENDPOINT"))
        vlm_model = os.environ.get("HRC_REPAIR_VLM_MODEL", str(model.get("model_id", "")))
        vision_verifier = ObserverVerifier(
            endpoint,
            vlm_model,
            api_key_env=str(model.get("api_key_env_name", "HRC_REPAIR_API_KEY")),
            timeout_s=float(model.get("request_timeout_s", 30)),
        )
    max_turns = int(config.get("budgets", {}).get("max_high_level_actions_total", 12))
    pick_budget = int(config.get("budgets", {}).get("max_pick_attempts_total", 1))
    place_budget = int(config.get("budgets", {}).get("max_place_attempts_total", 3))
    help_budget = int(config.get("budgets", {}).get("max_help_requests_total", 1))
    epoch = 0
    history: list[dict[str, Any]] = []
    online_status = "INCOMPLETE"
    helper_used = False
    obs: Observation | None = None
    try:
        with RepairLogger(out_dir, manifest) as logger:
            obs = adapter.reset(episode_id, int(seed), out_dir)
            logger.event("reset", {"observation_id": obs.observation_id, "control_epoch": epoch, "backend": adapter.backend_name})
            if rgb_repair or vader_scattered:
                reset_frames = dict(obs.frames)
                obs, verification = assess_initial(
                    vision_verifier, obs, list(reset_frames.values()), expected_initial_state(config)
                )
                logger.event("rgb_state_observation", {
                    "phase": "initial",
                    "frames_used": list(reset_frames),
                    "verdict": verification.verdict.value,
                    "evidence": verification.evidence,
                    "prompt_hash": verification.prompt_hash,
                    "response_hash": verification.response_hash,
                })
            logger.observation(obs.to_dict())
            for turn in range(max_turns):
                scattered = str(config.get("scene", {}).get("start_stage", "")) == "scattered"
                budgets = {"pick_remaining": pick_budget, "place_remaining": place_budget, "help_remaining": help_budget, "actions_remaining": max_turns - turn}
                safety_fallback = False
                try:
                    planner_obs = planner_observation(obs) if method == "vader" or rgb_repair else obs
                    planner_history = [] if method == "vader" else history
                    action, planner_meta = _fresh_action(planner, planner_obs, planner_history, budgets, epoch, retry_invalid_output=method == "vader" or rgb_repair)
                    if vader_scattered or (rgb_repair and scattered):
                        rejection = _scatter_action_error(action, obs, place_once=vader_scattered)
                        if rejection:
                            rejected_action = action.to_dict()
                            logger.event("planner_action_rejected", {"turn": turn, "action": rejected_action, "reason": rejection})
                            logger.model_call({"turn": turn, "provider": planner_meta.get("provider"), "model": planner_meta.get("model"), "prompt_hash": planner_meta.get("prompt_hash"), "public_input": planner_meta.get("public_input"), "action": rejected_action, "validation": "state_rejected"})
                            next_action = _scatter_recovery_action(obs, help_budget=help_budget)
                            if next_action == "stop":
                                action = _safe_stop_action(obs)
                                planner_meta = {"provider": "safety_fallback", "model": None}
                                safety_fallback = True
                                logger.event("safe_recovery_action", {"turn": turn, "action": action.to_dict(), "reason": rejection})
                            else:
                                correction = [{"protocol_error": f"The previous action was rejected: {rejection}. Current public state is held={obs.held}, placed={obs.placed}; choose `{next_action}` next."}]
                                action, planner_meta = _fresh_action(planner, planner_obs, correction, budgets, epoch, retry_invalid_output=True)
                                rejection = _scatter_action_error(action, obs, place_once=vader_scattered)
                                if rejection:
                                    logger.model_call({"turn": turn, "provider": planner_meta.get("provider"), "model": planner_meta.get("model"), "prompt_hash": planner_meta.get("prompt_hash"), "public_input": planner_meta.get("public_input"), "action": action.to_dict(), "validation": "state_rejected_after_retry"})
                                    action = _safe_stop_action(obs)
                                    planner_meta = {"provider": "safety_fallback", "model": None}
                                    safety_fallback = True
                                    logger.event("safe_recovery_action", {"turn": turn, "action": action.to_dict(), "reason": rejection})
                                else:
                                    planner_meta["state_action_retry"] = True
                                    planner_meta["rejected_action"] = rejected_action
                    if (rgb_repair or vader_scattered) and action.action not in {"pick", "place", "help", "finish", "stop"}:
                        raise ValueError(f"unsupported action for this REPAIR task adapter: {action.action}")
                    if rgb_repair and scattered and action.action == "place" and obs.held != "yes":
                        raise ValueError("place rejected: RGB observation does not confirm that the cover is held")
                    if rgb_repair and scattered and action.action == "pick" and obs.held != "no":
                        raise ValueError("pick rejected: RGB observation does not confirm that the cover is unheld")
                    recent = obs.recent_skill or {}
                    public_sensor = (recent.get("feedback") or {}).get("public_sensor", {})
                    if vader_scattered and action.action == "help" and (obs.held != "yes" or not public_sensor.get("blocked_guard")):
                        raise ValueError("help rejected: public evidence does not confirm a guarded placement block while holding the cover")
                    if vader_scattered and action.action == "place" and recent.get("skill_id") == "place":
                        raise ValueError("place rejected: this live episode has already used its placement attempt")
                    if (rgb_repair or vader_scattered) and obs.placed == "yes" and obs.release_observed == "yes" and action.action not in {"finish", "stop"}:
                        raise ValueError("only finish or stop is valid after observed placement and release")
                    if not safety_fallback:
                        logger.model_call({"turn": turn, "provider": planner_meta.get("provider"), "model": planner_meta.get("model"), "prompt_hash": planner_meta.get("prompt_hash"), "public_input": planner_meta.get("public_input"), "action": action.to_dict(), "action_format_retry": planner_meta.get("action_format_retry", False), "stale_action_retry": planner_meta.get("stale_action_retry", False), "state_action_retry": planner_meta.get("state_action_retry", False), "rejected_action": planner_meta.get("rejected_action"), "validation": "passed"})
                    logger.event("planner_action", {"turn": turn, "action": action.to_dict(), "control_epoch": epoch, "source": "safety_fallback" if safety_fallback else "planner"})
                except (PlannerUnavailable, ValueError, TypeError) as exc:
                    online_status = "SYSTEM_ERROR"
                    logger.event("system_error", {"turn": turn, "error_type": type(exc).__name__, "message": str(exc)})
                    break
                if action.action == "finish":
                    visual_ok = not (method == "vader" or rgb_repair) or visual_success(obs)
                    if obs.placed == "yes" and obs.release_observed == "yes" and visual_ok:
                        online_status = "DONE"
                        logger.event("finish_accepted", {"observation_id": obs.observation_id, "vqa_success": visual_ok})
                    else:
                        online_status = "FALSE_FINISH_REJECTED"
                        logger.event("finish_rejected", {"observation_id": obs.observation_id, "placed": obs.placed, "release_observed": obs.release_observed, "vqa_success": visual_ok})
                    break
                if action.action == "stop":
                    online_status = "STOPPED"
                    logger.event("stop", {"reason": "planner_stop"})
                    break
                if action.action == "observe":
                    obs = Observation(obs.episode_id, f"obs_{turn+1:04d}", epoch, time.time(), frames=obs.frames, held=obs.held, placed=obs.placed, target_visible=obs.target_visible, release_observed=obs.release_observed, unsafe_visible=obs.unsafe_visible, evidence="fresh observation requested", sensor_availability=obs.sensor_availability, recent_skill=obs.recent_skill)
                    logger.observation(obs.to_dict())
                    continue
                if action.action == "help":
                    if help_budget <= 0:
                        online_status = "BUDGET_EXHAUSTED"
                        logger.event("protocol_error", {"reason": "help budget exhausted"})
                        break
                    help_budget -= 1
                    old_epoch = epoch
                    epoch += 1
                    logger.event("handoff_prepared", {"control_epoch_before": old_epoch, "control_epoch_after": epoch, "cancelled_action": True, "safe_hold": True})
                    report = adapter.help(obs, out_dir)
                    helper_used = report.status == "succeeded"
                    logger.event("help", report.to_dict())
                    epoch = max(epoch + 1, report.control_epoch_after)
                    pseudo = SkillResult(
                        obs.episode_id, obs.observation_id, epoch, "help",
                        "completed" if report.status == "succeeded" else "system_error",
                        "succeeded" if report.status == "succeeded" else "unknown",
                        None if report.status == "succeeded" else "helper_unsupported",
                        feedback={"public_evidence": report.report},
                    )
                    obs = observer.after_skill(obs.episode_id, epoch, pseudo, frames=report.frames)
                    if (rgb_repair or method == "vader") and report.frames:
                        obs, verification = assess_help(
                            vision_verifier, obs, list(report.frames.values()),
                            "The hub cover remains visibly held by the robot, is not yet seated on the casing, and the robot is safely holding position after the helper cleared the area.",
                        )
                        logger.event("rgb_state_observation", {
                            "phase": "after_help",
                            "frames_used": list(report.frames),
                            "verdict": verification.verdict.value,
                            "evidence": verification.evidence,
                            "prompt_hash": verification.prompt_hash,
                            "response_hash": verification.response_hash,
                        })
                    logger.observation(obs.to_dict())
                    history.append({"action": action.to_dict(), "help_report": report.to_dict(), "observation": obs.to_dict()})
                    continue
                if action.action == "retract":
                    result = adapter.retract(obs, out_dir)
                elif action.action == "pick":
                    if pick_budget <= 0:
                        online_status = "BUDGET_EXHAUSTED"
                        logger.event("protocol_error", {"reason": "pick budget exhausted"})
                        break
                    pick_budget -= 1
                    result = adapter.pick(obs, out_dir)
                elif action.action == "place":
                    if place_budget <= 0:
                        online_status = "BUDGET_EXHAUSTED"
                        logger.event("protocol_error", {"reason": "place budget exhausted"})
                        break
                    place_budget -= 1
                    result = adapter.place(obs, out_dir)
                else:
                    online_status = "PROTOCOL_ERROR"
                    logger.event("protocol_error", {"reason": f"unhandled action {action.action}"})
                    break
                metrics_path = None
                if result.feedback.get("metrics_path"):
                    metrics_path = Path(str(result.feedback["metrics_path"]))
                    logger.gt({"skill": result.skill_id, "metrics_path": str(metrics_path), "metrics": load_json(metrics_path)})
                public_result = result
                if str(config.get("experiment", {}).get("feedback_mode", "")).startswith("public_"):
                    # Private evaluator labels, scores, paths, and image paths
                    # stay outside events, history, and planner input.
                    public_result = _public_skill_result(result)
                logger.event("skill_result", public_result.to_dict())
                logger.trajectory({"turn": turn, "skill": public_result.to_dict()})
                public_metrics_path = metrics_path if str(config.get("experiment", {}).get("feedback_mode", "")) == "privileged_debug" else None
                obs = observer.after_skill(obs.episode_id, epoch, public_result, metrics_path=public_metrics_path, frames={})
                if (rgb_repair or vader_scattered) and action.action == "pick":
                    expected = expected_pick_outcome(config)
                    obs, verification = assess_pick(vision_verifier, obs, result.frames_after, expected)
                    frame_aliases = [
                        alias for alias in ("head_rgb", "left_hand_rgb", "right_hand_rgb")
                        if any(alias in Path(frame).stem for frame in result.frames_after)
                    ]
                    logger.event("rgb_state_observation", {"observation_id": obs.observation_id, "phase": "pick", "expected_outcome": expected, "frames_used": frame_aliases, "verdict": verification.verdict.value, "evidence": verification.evidence, "prompt_hash": verification.prompt_hash, "response_hash": verification.response_hash})
                elif (method == "vader" or rgb_repair) and action.action == "place":
                    expected = expected_place_outcome(config)
                    obs, verification = assess_place(vision_verifier, obs, result.frames_after, expected, all_views=rgb_repair)
                    logger.event("rgb_state_observation" if rgb_repair else "vader_detection", {"observation_id": obs.observation_id, "expected_outcome": expected, "verdict": verification.verdict.value, "evidence": verification.evidence, "prompt_hash": verification.prompt_hash, "response_hash": verification.response_hash})
                logger.observation(obs.to_dict())
                history.append({"action": action.to_dict(), "skill_result": public_result.to_dict(), "observation": obs.to_dict()})
                if result.execution_status in {"safety_fault", "system_error"} and result.outcome == "unknown" and action.action != "place":
                    online_status = "SYSTEM_ERROR"
                    break
            else:
                online_status = "BUDGET_EXHAUSTED"
            metrics = evaluate_artifact(out_dir, online_status=online_status, helper_used=helper_used, scenario=scenario, feedback_mode=str(config.get("experiment", {}).get("feedback_mode", "privileged_debug")))
            protocol_version = "vader-m0-v1" if method == "vader" else "repair-m0-rgb-v1" if rgb_repair else "repair-m0-v1"
            metrics.update({"episode_id": episode_id, "method": method, "seed": int(seed), "pick_attempts": int(config.get("budgets", {}).get("max_pick_attempts_total", 1)) - pick_budget, "place_attempts": int(config.get("budgets", {}).get("max_place_attempts_total", 3)) - place_budget, "help_requests": int(config.get("budgets", {}).get("max_help_requests_total", 1)) - help_budget, "termination": online_status, "protocol_version": protocol_version})
            logger.metrics(metrics)
    finally:
        adapter.close()
    return metrics if 'metrics' in locals() else {"episode_id": episode_id, "termination": online_status}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def yaml_dump(value: dict[str, Any]) -> str:
    import yaml
    return yaml.safe_dump(value, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/repair_m0.yaml")
    parser.add_argument("--method", choices=("auto", "fixed_retry", "repair", "vader"), default="repair")
    parser.add_argument("--scenario", choices=("nominal", "blocked"), default="nominal")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--official-rgb", action="store_true",
        help="Enable the REPAIR-style RGB state recognizer and explicit post-help replanning for the task adaptation.",
    )
    parser.add_argument("--dual-gripper", action="store_true", help="Run the registered pick/place skill with synchronized two-arm grasp, lift, and transport.")
    parser.add_argument("--transport-direct", action="store_true", help="Use the registered skill's direct measured-state transport segment for this REPAIR episode.")
    parser.add_argument("--transport-clearance-z-m", type=float, default=None, help="Set vertical clearance above the Casing root for this REPAIR transport skill.")
    parser.add_argument("--reanchor-at-insertion-above", action="store_true", help="Recompute the grasp frame from measured robot/Hub state before pre-insertion alignment.")
    parser.add_argument("--correct-orientation-at-insertion-above", action="store_true", help="Use the registered place skill to correct the measured held-part orientation above the socket before insertion.")
    parser.add_argument("--release-right-after-lift", action="store_true", help="Use both arms to grasp and lift, then release the right gripper for single-arm placement.")
    parser.add_argument("--release-right-at-insertion-above", action="store_true", help="Keep both grippers through transport and orientation correction, then release/retract the right gripper above the socket before insertion.")
    parser.add_argument("--cover-xy", nargs=2, type=float, metavar=("X", "Y"), help="Override the supported Hub Cover start location in metres.")
    parser.add_argument("--casing-xy", nargs=2, type=float, metavar=("X", "Y"), help="Override the supported Casing Top location in metres.")
    parser.add_argument("--max-place-attempts", type=int, default=None, help="Override the episode place budget for a bounded smoke run.")
    parser.add_argument("--episode-wall-timeout-s", type=int, default=None, help="Override the wall-time budget for slow rendered physics episodes.")
    parser.add_argument("--backend", choices=("isaac_subprocess", "isaac_inprocess", "contract_debug"), default=None)
    args = parser.parse_args(argv)
    config_path, config = load_config(args.config)
    if args.method == "vader" and str(config.get("scene", {}).get("start_stage", "")) != "scattered":
        parser.error("VADER requires scene.start_stage=scattered for a live pick-and-place episode")
    if args.release_right_after_lift and not args.dual_gripper:
        parser.error("--release-right-after-lift requires --dual-gripper")
    if args.release_right_at_insertion_above and not args.dual_gripper:
        parser.error("--release-right-at-insertion-above requires --dual-gripper")
    if args.episode_wall_timeout_s is not None and args.episode_wall_timeout_s <= 0:
        parser.error("--episode-wall-timeout-s must be positive")
    if args.max_place_attempts is not None or args.episode_wall_timeout_s is not None or args.backend is not None or args.official_rgb or args.dual_gripper or args.transport_direct or args.transport_clearance_z_m is not None or args.reanchor_at_insertion_above or args.correct_orientation_at_insertion_above or args.release_right_after_lift or args.release_right_at_insertion_above or args.cover_xy or args.casing_xy:
        config = copy.deepcopy(config)
        if args.max_place_attempts is not None:
            config.setdefault("budgets", {})["max_place_attempts_total"] = int(args.max_place_attempts)
        if args.episode_wall_timeout_s is not None:
            config.setdefault("budgets", {})["episode_wall_timeout_s"] = int(args.episode_wall_timeout_s)
        if args.backend is not None:
            config.setdefault("experiment", {})["backend"] = args.backend
        if args.official_rgb:
            if args.method != "repair":
                parser.error("--official-rgb is only supported with --method repair")
            config.setdefault("model", {})["rgb_observer_enabled"] = True
            config.setdefault("controller", {})["replan_after_help"] = True
            config.setdefault("controller", {})["stepwise_actions"] = True
        if args.dual_gripper:
            controller = config.setdefault("controller", {})
            controller["m0_dual_gripper"] = True
            controller["m0_synchronous_dual_lift"] = True
            controller["m0_synchronous_dual_transport"] = True
            controller["m0_release_right_after_lift"] = False
        if args.transport_direct:
            config.setdefault("controller", {})["m0_transport_direct"] = True
        if args.transport_clearance_z_m is not None:
            if args.transport_clearance_z_m < 0:
                parser.error("--transport-clearance-z-m must be non-negative")
            config.setdefault("controller", {})["transport_clearance_z_m"] = args.transport_clearance_z_m
        if args.reanchor_at_insertion_above:
            config.setdefault("controller", {})["m0_reanchor_at_insertion_above"] = True
        if args.correct_orientation_at_insertion_above:
            config.setdefault("controller", {})["m0_correct_orientation_at_insertion_above"] = True
        if args.release_right_after_lift:
            config.setdefault("controller", {})["m0_release_right_after_lift"] = True
        if args.release_right_at_insertion_above:
            config.setdefault("controller", {})["m0_release_right_at_insertion_above"] = True
        if args.cover_xy:
            config.setdefault("scene", {})["cover_initial_xy"] = args.cover_xy
        if args.casing_xy:
            config.setdefault("scene", {})["casing_initial_xy"] = args.casing_xy
    check = preflight(config_path, config, strict=False)
    if check["errors"]:
        print(json.dumps(check, indent=2, ensure_ascii=False))
        return 2
    result = run_episode(config_path, config, method=args.method, scenario=args.scenario, seed=args.seed, out_dir=Path(args.out).resolve() if args.out else None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("pipeline_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
