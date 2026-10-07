"""Run VADER's plan-execute-detect loop on a full-physics Isaac task."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from hrc_repair.adapter import IsaacPersistentWorkerAdapter
from hrc_repair.config import repo_root
from hrc_repair.contracts import Observation


TASK = "Place Hub Cover Output Top onto Casing Top at socket_hub_output."
VQA_PROMPT = """You are VADER's visual state and affordance detector. Judge only the supplied RGB views.
Return SUCCESS only when the expected state is directly visible, FAILED when visible evidence contradicts it,
and UNKNOWN when the views do not resolve it. Do not infer contact from proximity. The Hub Cover Output Top
has a broad circular flange and a raised center opening; that opening belongs to the cover, not a loose ring.
Return one JSON object with keys verdict (SUCCESS, FAILED, UNKNOWN) and assessment (one concise sentence).
Expected outcome: {expected}"""
LMP_PROMPT = """You are the language-model planner in VADER's plan-execute-detect loop.
Task: {task}
Choose exactly one item from available_skills; any other action is invalid. For pick or place, also write the natural-language expected_outcome
that the VQA detector must check after execution. Use the latest visual VQA assessment to continue or replan.
Never claim finish unless the latest VQA assessment is SUCCESS for the seated-and-released final state.
If the state is UNKNOWN, or a skill failed without an available recovery skill, choose stop. Use help only when
the latest skill result reports a guarded placement obstruction and help is listed as available. If initial visual
affordance is uncertain and observe is available, choose observe rather than assuming the cover is held.
Return JSON only: {{"action":"pick|place|help|finish|stop|observe","expected_outcome":"..."}}.
Current public observation:
{observation}
Available skills: {available_skills}"""

EXPECTED = {
    "initial": "Hub Cover Output Top rests separately on its support, visibly outside the black gripper fingers; Casing Top is separate.",
    "pick": "The robot is holding Hub Cover Output Top lifted clear of its physical support; it is not yet seated on Casing Top.",
    "place": "Hub Cover Output Top is seated on Casing Top around socket_hub_output, and the robot has released it and withdrawn clear.",
    "help": "The placement obstruction is cleared; the robot still holds Hub Cover Output Top and it is not seated yet.",
    "initial_recheck": "Hub Cover Output Top rests separately on its support, visibly outside the black gripper fingers; Casing Top is separate.",
}


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _request(endpoint: str, model: str, api_key: str, messages: list[dict[str, Any]], timeout_s: float) -> tuple[str, dict[str, Any]]:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    request = Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    started = time.monotonic()
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8"))
    content = decoded["choices"][0]["message"]["content"]
    return str(content), {
        "prompt_hash": _hash(body),
        "response_hash": _hash(raw),
        "latency_s": round(time.monotonic() - started, 3),
    }


def _image_data(path: str) -> str:
    image = Path(path)
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"


def _vqa(config: dict[str, Any], frames: dict[str, str], expected: str) -> dict[str, Any]:
    model = config["models"]
    content: list[dict[str, Any]] = [{"type": "text", "text": VQA_PROMPT.format(expected=expected)}]
    for name, path in sorted(frames.items()):
        content.extend([
            {"type": "text", "text": f"Camera view: {name}"},
            {"type": "image_url", "image_url": {"url": _image_data(path)}},
        ])
    raw, meta = _request(model["endpoint"], model["vlm"], os.environ[model["api_key_env"]], [{"role": "user", "content": content}], float(model.get("timeout_s", 90)))
    try:
        result = _json_object(raw)
        verdict = str(result.get("verdict", "UNKNOWN")).upper()
        if verdict not in {"SUCCESS", "FAILED", "UNKNOWN"}:
            verdict = "UNKNOWN"
        evidence = str(result.get("assessment", "")).strip()[:1200]
    except (ValueError, TypeError):
        verdict, evidence = "UNKNOWN", raw[:1200]
    return {"verdict": verdict, "evidence": evidence, **meta}


def _plan(config: dict[str, Any], observation: Observation, available: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    public = {
        "observation_id": observation.observation_id,
        "held": observation.held,
        "placed": observation.placed,
        "release_observed": observation.release_observed,
        "evidence": observation.evidence,
        "recent_skill": observation.recent_skill,
    }
    prompt = LMP_PROMPT.format(
        task=TASK,
        observation=json.dumps(public, ensure_ascii=False),
        available_skills=json.dumps(available),
    )
    model = config["models"]
    raw, meta = _request(model["endpoint"], model["lmp"], os.environ[model["api_key_env"]], [{"role": "user", "content": prompt}], float(model.get("timeout_s", 90)))
    result = _json_object(raw)
    action = str(result.get("action", "")).lower()
    if action not in {"pick", "place", "help", "finish", "stop", "observe"}:
        raise ValueError(f"unsupported VADER action: {action!r}")
    expected = str(result.get("expected_outcome", "")).strip()
    if action in {"pick", "place"} and not expected:
        raise ValueError(f"VADER {action} plan omitted expected_outcome")
    return {"action": action, "expected_outcome": expected}, meta


def _frames(paths: list[str]) -> dict[str, str]:
    result = {}
    for path in paths:
        name = Path(path).stem
        alias = next((camera for camera in ("head_rgb", "left_hand_rgb", "right_hand_rgb") if camera in name), None)
        if alias:
            result[alias] = path
    return result


def _detected_observation(observation: Observation, *, phase: str, result: dict[str, Any], frame_paths: list[str], expected_outcome: str, skill: dict[str, Any] | None = None, epoch: int | None = None) -> Observation:
    verdict = result["verdict"]
    held, placed, released = "unknown", "unknown", "unknown"
    if phase in {"initial", "initial_recheck"} and verdict == "SUCCESS":
        held, placed, released = "no", "no", "no"
    elif phase in {"pick", "help"} and verdict == "SUCCESS":
        held, placed, released = "yes", "no", "no"
    elif phase == "place":
        if verdict == "SUCCESS":
            held, placed, released = "no", "yes", "yes"
        elif verdict == "FAILED":
            placed, released = "no", "no"
    recent = {
        **(skill or {}),
        "expected_outcome": expected_outcome,
        "vqa_assessment": {"verdict": verdict, "evidence": result["evidence"]},
    }
    return replace(
        observation,
        observation_id=f"obs_{int(observation.observation_id.split('_')[-1]) + 1:04d}" if phase != "initial" else observation.observation_id,
        control_epoch=observation.control_epoch if epoch is None else epoch,
        timestamp=time.time(),
        frames=_frames(frame_paths),
        held=held,
        placed=placed,
        release_observed=released,
        evidence=result["evidence"],
        recent_skill=recent,
        sensor_availability={**observation.sensor_availability, "rgb": bool(frame_paths)},
    )


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _available(observation: Observation, last_skill: str, blocked: bool) -> list[str]:
    verdict = str(((observation.recent_skill or {}).get("vqa_assessment") or {}).get("verdict", "UNKNOWN"))
    if blocked and observation.held == "yes":
        return ["help", "stop"]
    if last_skill == "help" and verdict == "SUCCESS" and observation.held == "yes":
        return ["place", "stop"]
    if last_skill == "pick" and verdict == "SUCCESS" and observation.held == "yes":
        return ["place", "stop"]
    if last_skill == "place" and verdict == "SUCCESS" and observation.placed == "yes" and observation.release_observed == "yes":
        return ["finish", "stop"]
    if last_skill in {"", "observe"} and verdict == "SUCCESS" and observation.held == "no":
        return ["pick", "stop"]
    if last_skill == "" and observation.frames:
        return ["observe", "stop"]
    return ["stop"]


def run_episode(config_path: Path, config: dict[str, Any], *, scenario: str, seed: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    episode_id = f"vader_{uuid.uuid4().hex[:12]}"
    model = config["models"]
    if model["api_key_env"] not in os.environ:
        raise RuntimeError(f"missing model credential environment variable {model['api_key_env']}")
    adapter = IsaacPersistentWorkerAdapter(
        scenario=scenario,
        repo_root=repo_root(),
        config=config,
        out_dir=out_dir,
        vader_vision=True,
        initial_rgb_observation=True,
    )
    event_path, call_path = out_dir / "events.jsonl", out_dir / "model_calls.jsonl"
    status, last_skill, physics_path, epoch = "INCOMPLETE", "", None, 0
    try:
        observation = adapter.reset(episode_id, seed, out_dir)
        initial = _vqa(config, observation.frames, EXPECTED["initial"])
        _write_jsonl(call_path, {"kind": "vqa", "phase": "initial", "views": sorted(observation.frames), "expected_outcome": EXPECTED["initial"], **initial})
        observation = _detected_observation(observation, phase="initial", result=initial, frame_paths=list(observation.frames.values()), expected_outcome=EXPECTED["initial"])
        _write_jsonl(event_path, {"event": "detection", "phase": "initial", "observation_id": observation.observation_id, "verdict": initial["verdict"], "evidence": initial["evidence"]})
        max_steps = int(config.get("budgets", {}).get("max_plan_steps", 8))
        for step in range(max_steps):
            blocked = bool(((observation.recent_skill or {}).get("public_sensor") or {}).get("blocked_guard"))
            available = _available(observation, last_skill, blocked)
            plan, plan_meta = _plan(config, observation, available)
            if plan["action"] not in available:
                _write_jsonl(event_path, {"event": "plan_rejected", "step": step, "action": plan["action"], "available_skills": available})
                retry_obs = replace(observation, recent_skill={**(observation.recent_skill or {}), "planner_feedback": f"Action {plan['action']!r} is not executable now. Choose one of {available}."})
                plan, plan_meta = _plan(config, retry_obs, available)
                if plan["action"] not in available:
                    raise ValueError(f"planner selected unavailable skill {plan['action']!r}; available={available}")
            _write_jsonl(call_path, {"kind": "lmp", "step": step, "public_observation": {"observation_id": observation.observation_id, "held": observation.held, "placed": observation.placed, "evidence": observation.evidence, "recent_skill": observation.recent_skill}, "available_skills": available, "action": plan, **plan_meta})
            _write_jsonl(event_path, {"event": "plan", "step": step, "observation_id": observation.observation_id, **plan})
            action = plan["action"]
            if action == "finish":
                status = "DONE"
                break
            if action == "stop":
                status = "STOPPED"
                break
            if action == "help":
                report = adapter.help(observation, out_dir)
                frame_map = report.frames or {}
                phase = "help"
                skill_summary = {"skill_id": "help", "execution_status": report.status, "public_evidence": report.report}
                frames = list(frame_map.values())
                epoch = report.control_epoch_after
                last_skill = "help"
                expected = plan["expected_outcome"] or EXPECTED["help"]
            elif action == "observe":
                phase = "initial_recheck"
                frame_map = observation.frames
                frames = list(frame_map.values())
                skill_summary = {"skill_id": "observe", "execution_status": "completed", "public_evidence": "VADER requested another visual affordance check; robot and parts were not moved."}
                last_skill = "observe"
                expected = EXPECTED["initial_recheck"]
            else:
                result = adapter.pick(observation, out_dir) if action == "pick" else adapter.place(observation, out_dir)
                phase = action
                frames = result.frames_after
                skill_summary = {
                    "skill_id": result.skill_id,
                    "execution_status": result.execution_status,
                    "failure_code": result.failure_code if result.failure_code in {"contact_guard", "system_error", "motion_timeout", "missing_rgb_frames"} else None,
                    "public_evidence": result.feedback.get("public_evidence", ""),
                    "public_sensor": {"blocked_guard": bool((result.feedback.get("public_sensor") or {}).get("blocked_guard"))},
                }
                physics_path = result.feedback.get("metrics_path", physics_path)
                last_skill = action
                expected = plan["expected_outcome"]
            frame_map = _frames(frames) if isinstance(frames, list) else frame_map
            assessment = _vqa(config, frame_map, expected) if frame_map else {"verdict": "UNKNOWN", "evidence": "no camera frames after skill"}
            _write_jsonl(call_path, {"kind": "vqa", "phase": phase, "views": sorted(frame_map), "expected_outcome": expected, **assessment})
            observation = _detected_observation(observation, phase=phase, result=assessment, frame_paths=list(frame_map.values()), expected_outcome=expected, skill=skill_summary, epoch=epoch)
            observation = replace(observation, observation_id=f"obs_{step + 1:04d}")
            _write_jsonl(event_path, {"event": "execution_detection", "step": step, "skill": phase, "execution_status": skill_summary["execution_status"], "observation_id": observation.observation_id, "verdict": assessment["verdict"], "evidence": assessment["evidence"]})
        else:
            status = "STEP_BUDGET_EXHAUSTED"
    except Exception as exc:
        status = "PLANNER_INVALID" if isinstance(exc, ValueError) and "planner selected unavailable skill" in str(exc) else "SYSTEM_ERROR"
        _write_jsonl(event_path, {"event": "system_error", "type": type(exc).__name__, "message": str(exc)})
    finally:
        adapter.close()
    physics = None
    if physics_path and Path(str(physics_path)).is_file():
        try:
            raw = json.loads(Path(str(physics_path)).read_text(encoding="utf-8"))
            physics = {
                "artifact": str(Path(str(physics_path)).relative_to(out_dir)),
                "task_success": raw.get("task_success"),
                "physical_place_score": raw.get("physical_place_score"),
                "placement_score": raw.get("placement_score"),
            }
        except (OSError, ValueError):
            physics = {"artifact": str(physics_path)}
    metrics = {
        "episode_id": episode_id,
        "method": "VADER",
        "scenario": scenario,
        "seed": seed,
        "vader_status": status,
        "latest_vqa": (observation.recent_skill or {}).get("vqa_assessment"),
        "physics_evaluation_posthoc": physics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vader_hub_cover_to_casing_top.yaml")
    parser.add_argument("--scenario", choices=("nominal", "blocked"), default="nominal")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else repo_root() / "validation_logs" / f"vader_hub_cover_casing_top_{time.strftime('%Y%m%d_%H%M%S')}"
    result = run_episode(config_path, config, scenario=args.scenario, seed=args.seed, out_dir=out_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["vader_status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
