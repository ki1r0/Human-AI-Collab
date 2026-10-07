"""Small strict message contracts for the M1 state machine.

This module intentionally uses the standard library.  Model output is an
untrusted boundary: unknown keys, stale observations, arbitrary commands and
hidden evaluator fields are rejected before they can reach an adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


ALLOWED_ACTIONS = (
    "OBSERVE",
    "PICK",
    "MOVE_TO_PREINSERT",
    "GUARDED_INSERT",
    "RELEASE_RETRACT",
    "VERIFY",
    "ASK_HUMAN",
    "REPLAN",
    "ABORT",
)

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "ground_truth",
        "evaluator_state",
        "evaluator_private",
        "fault_cause",
        "fault_label",
        "penetration",
        "contact_truth",
        "usd_pose",
        "world_pose_truth",
    }
)


class EpisodeMode(str, Enum):
    NOMINAL = "nominal"
    FAULT_HIL = "fault_hil"
    FAULT_NO_HELP = "fault_no_help"


class EpisodeState(str, Enum):
    RESET = "RESET"
    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    RELEASE_RETRACT = "RELEASE_RETRACT"
    SAFE_HOLD = "SAFE_HOLD"
    ASK_HUMAN = "ASK_HUMAN"
    VALIDATE_HANDOFF = "VALIDATE_HANDOFF"
    REOBSERVE = "REOBSERVE"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    REPLAN = "REPLAN"
    DONE = "DONE"
    ABORT = "ABORT"


class ControlOwner(str, Enum):
    AUTO = "AUTO"
    HUMAN = "HUMAN"
    SAFE_STOP = "SAFE_STOP"


def _reject_hidden(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden evaluator field at {path}/{key}")
            _reject_hidden(child, path=f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _reject_hidden(child, path=f"{path}/{idx}")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _plain_json(value: Any) -> Any:
    """Convert dataclass/enum values to deterministic JSON-compatible data."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numeric value is not valid JSON")
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(_plain_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    episode_id: str
    observation_id: str
    timestamp: float
    frames: dict[str, str] = field(default_factory=dict)
    qpos: list[float] = field(default_factory=list)
    qvel: list[float] = field(default_factory=list)
    gripper: dict[str, float] = field(default_factory=dict)
    sensor_availability: dict[str, bool] = field(default_factory=dict)
    task_id: str = "m1_hub_cover_output_top_seat"
    recent_skill: dict[str, Any] | None = None
    human_feedback: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.episode_id, "episode_id")
        _required_text(self.observation_id, "observation_id")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        _reject_hidden(self.frames)
        _reject_hidden(self.recent_skill or {})
        _reject_hidden(self.human_feedback or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "observation_id": self.observation_id,
            "timestamp": float(self.timestamp),
            "frames": dict(self.frames),
            "qpos": [float(v) for v in self.qpos],
            "qvel": [float(v) for v in self.qvel],
            "gripper": {str(k): float(v) for k, v in self.gripper.items()},
            "sensor_availability": {str(k): bool(v) for k, v in self.sensor_availability.items()},
            "task_id": self.task_id,
            "recent_skill": self.recent_skill,
            "human_feedback": self.human_feedback,
        }


@dataclass(frozen=True)
class SkillResult:
    episode_id: str
    observation_id: str
    skill_id: str
    motion_status: str
    held_status: str = "UNKNOWN"
    seat_status: str = "UNKNOWN"
    steps: int = 0
    failure_code: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    feedback: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.episode_id, "episode_id")
        _required_text(self.observation_id, "observation_id")
        _required_text(self.skill_id, "skill_id")
        _required_text(self.motion_status, "motion_status")
        if int(self.steps) < 0:
            raise ValueError("steps cannot be negative")
        _reject_hidden(self.feedback)

    def to_dict(self) -> dict[str, Any]:
        return _plain_json(asdict(self))


@dataclass(frozen=True)
class Decision:
    schema_version: str
    episode_id: str
    observation_id: str
    action: str
    target_part: str | None = None
    target_socket: str | None = None
    skill_args: dict[str, Any] = field(default_factory=dict)
    expected_postcondition: str | None = None
    evidence_summary: str | None = None
    ask_message: str | None = None
    abort_reason: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        if not isinstance(payload, Mapping):
            raise ValueError("decision must be a JSON object")
        allowed = {
            "schema_version",
            "episode_id",
            "observation_id",
            "action",
            "target_part",
            "target_socket",
            "skill_args",
            "expected_postcondition",
            "evidence_summary",
            "ask_message",
            "abort_reason",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown decision fields: {sorted(unknown)}")
        _reject_hidden(payload)
        action = _required_text(payload.get("action"), "action").upper()
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        args = payload.get("skill_args", {})
        if not isinstance(args, dict):
            raise ValueError("skill_args must be an object")
        if action in {"PICK", "MOVE_TO_PREINSERT", "GUARDED_INSERT", "RELEASE_RETRACT"}:
            if payload.get("target_part") != "Hub_Cover_Output_Top":
                raise ValueError("M1 skill target_part must be Hub_Cover_Output_Top")
        if action in {"MOVE_TO_PREINSERT", "GUARDED_INSERT"}:
            if payload.get("target_socket") != "socket_hub_output":
                raise ValueError("M1 insertion target_socket must be socket_hub_output")
        return cls(
            schema_version=_required_text(payload.get("schema_version"), "schema_version"),
            episode_id=_required_text(payload.get("episode_id"), "episode_id"),
            observation_id=_required_text(payload.get("observation_id"), "observation_id"),
            action=action,
            target_part=_optional_text(payload.get("target_part"), "target_part"),
            target_socket=_optional_text(payload.get("target_socket"), "target_socket"),
            skill_args=dict(args),
            expected_postcondition=_optional_text(payload.get("expected_postcondition"), "expected_postcondition"),
            evidence_summary=_optional_text(payload.get("evidence_summary"), "evidence_summary"),
            ask_message=_optional_text(payload.get("ask_message"), "ask_message"),
            abort_reason=_optional_text(payload.get("abort_reason"), "abort_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain_json(asdict(self))


def new_observation_id(episode_id: str, index: int) -> str:
    return f"{episode_id}:obs:{int(index):04d}"
