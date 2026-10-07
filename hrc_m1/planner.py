"""Strict, provider-agnostic LLM planner boundary for M1."""

from __future__ import annotations

import json
import os
import base64
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Mapping

from .contracts import Decision, EpisodeState, Observation


class PlannerUnavailable(RuntimeError):
    pass


def parse_planner_json(payload: str | Mapping[str, Any]) -> Decision:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"planner output is not JSON: {exc}") from exc
    else:
        data = dict(payload)
    return Decision.from_dict(data)


class HttpPlanner:
    """OpenAI-compatible JSON planner; no fallback is silently substituted."""

    def __init__(self, endpoint: str, model: str, *, api_key_env: str = "HRC_M1_API_KEY", timeout_s: float = 30.0, pipeline: str = "m1") -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.pipeline = pipeline

    def next_decision(self, observation: Observation, state: EpisodeState) -> tuple[Decision, dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise PlannerUnavailable(f"planner credential {self.api_key_env} is not configured")
        system = (
            "Plan the next action in the gearbox assembly. Return ONLY one JSON decision, never repeat the input. "
            "Copy the exact episode_id and observation_id from the request. Choose the action from allowed_actions. "
            "When state is PLAN and recent_skill is null, choose PICK. After a SUCCESS assessment, advance in order: "
            "pick to MOVE_TO_PREINSERT, preinsert to GUARDED_INSERT, insert to VERIFY, verify to RELEASE_RETRACT. "
            "Use expected_outcome and vqa_assessment when replanning. For FAILED, choose a safe retry only when the "
            "visual evidence supports it; otherwise choose ASK_HUMAN. For UNKNOWN, choose ASK_HUMAN or ABORT. "
            "Never treat UNKNOWN as success. The decision must contain schema_version, episode_id, observation_id, "
            "and action. For PICK, MOVE_TO_PREINSERT, GUARDED_INSERT, and RELEASE_RETRACT, set target_part to "
            "Hub_Cover_Output_Top. For MOVE_TO_PREINSERT and GUARDED_INSERT, also set target_socket to "
            "socket_hub_output. No markdown or extra keys."
            if self.pipeline == "vader"
            else "You are an M1 assembly action planner. Return exactly one JSON object matching the supplied schema. Use only allowed actions and never infer hidden evaluator fields."
        )
        public_observation = observation.to_dict()
        if self.pipeline == "vader":
            # VADER sends images to the verifier; the LMP consumes its textual assessment.
            public_observation["frames"] = {}
        else:
            public_observation["frames"] = {name: _frame_data_uri(value) for name, value in observation.frames.items()}
        user = {
            "task_id": observation.task_id,
            "state": state.value,
            "observation": public_observation,
            "allowed_actions": ["OBSERVE", "PICK", "MOVE_TO_PREINSERT", "GUARDED_INSERT", "RELEASE_RETRACT", "VERIFY", "ASK_HUMAN", "REPLAN", "ABORT"],
            "decision_ids": {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id},
            "assembly_targets": {"part": "Hub_Cover_Output_Top", "socket": "socket_hub_output"},
        }
        request_payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=True)}],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PlannerUnavailable(f"planner request failed: {type(exc).__name__}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerUnavailable("planner response has no choices[0].message.content") from exc
        if self.pipeline == "vader":
            data = json.loads(content) if isinstance(content, str) else dict(content)
            action = str(data.get("action", "")).upper()
            if action in {"PICK", "MOVE_TO_PREINSERT", "GUARDED_INSERT", "RELEASE_RETRACT"}:
                data.setdefault("target_part", "Hub_Cover_Output_Top")
            if action in {"MOVE_TO_PREINSERT", "GUARDED_INSERT"}:
                data.setdefault("target_socket", "socket_hub_output")
            content = data
        return parse_planner_json(content), {
            "provider": "http",
            "model": self.model,
            "prompt_hash": _hash_response(request_payload),
            "raw_response_hash": _hash_response(raw),
        }


class RulePlanner:
    """Deterministic contract exerciser, explicitly not a model or M1 result."""

    def next_decision(self, observation: Observation, state: EpisodeState) -> tuple[Decision, dict[str, Any]]:
        recent = observation.recent_skill or {}
        motion = str(recent.get("motion_status", "")).upper()
        previous = str(recent.get("skill_id", ""))
        if motion in {"FAILED", "UNKNOWN"}:
            action = "ASK_HUMAN" if motion == "FAILED" else "ABORT"
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": action, "ask_message": "The guarded skill did not establish a safe postcondition." if action == "ASK_HUMAN" else None}
        elif previous == "pick":
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": "MOVE_TO_PREINSERT", "target_part": "Hub_Cover_Output_Top", "target_socket": "socket_hub_output"}
        elif previous == "preinsert":
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": "GUARDED_INSERT", "target_part": "Hub_Cover_Output_Top", "target_socket": "socket_hub_output"}
        elif previous == "insert":
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": "VERIFY", "expected_postcondition": "seat_candidate"}
        elif previous == "verify":
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": "RELEASE_RETRACT", "target_part": "Hub_Cover_Output_Top"}
        else:
            data = {"schema_version": "m1-decision-v1", "episode_id": observation.episode_id, "observation_id": observation.observation_id, "action": "PICK", "target_part": "Hub_Cover_Output_Top"}
        return Decision.from_dict(data), {"provider": "rule_dev_only", "warning": "not a model and not an M1 rollout"}


def _hash_response(value: Any) -> str:
    return __import__("hashlib").sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _frame_data_uri(value: str) -> str:
    """Turn a local run-artifact path into an image URL at the HTTP boundary."""
    if value.startswith("data:image/"):
        return value
    path = Path(value)
    if path.is_file():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
        return f"data:{mime};base64,{encoded}"
    return value


__all__ = ["HttpPlanner", "PlannerUnavailable", "RulePlanner", "parse_planner_json"]
