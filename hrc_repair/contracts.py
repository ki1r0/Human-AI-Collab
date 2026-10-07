"""Public M0 data contracts.

The planner sees only public observations and skill outcomes.  Ground-truth
poses, contact data, scenario labels, and private evaluator fields are kept
out of these structures by validation rather than by prompt wording alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

TRI_VALUES = ("yes", "no", "unknown")
ALLOWED_ACTIONS = ("observe", "pick", "place", "retract", "help", "finish", "stop")
PRIVATE_KEYS = frozenset({
    "ground_truth", "gt", "evaluator_private", "scenario", "fault_profile",
    "fault_cause", "usd_pose", "world_pose_truth", "penetration",
    "contact_truth", "hub_pose", "casing_pose", "blocker_prim",
})


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _reject_private(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in PRIVATE_KEYS:
                raise ValueError(f"private evaluator key at {path}/{key}")
            _reject_private(child, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private(child, f"{path}/{index}")


def _text(value: Any, name: str, max_len: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = value.strip()
    if max_len is not None and len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} characters")
    return value


def _tri(value: Any, name: str) -> str:
    value = str(value).lower()
    if value not in TRI_VALUES:
        raise ValueError(f"{name} must be yes, no, or unknown")
    return value


@dataclass(frozen=True)
class Observation:
    episode_id: str
    observation_id: str
    control_epoch: int
    timestamp: float
    frames: dict[str, str] = field(default_factory=dict)
    held: str = "unknown"
    placed: str = "unknown"
    target_visible: str = "unknown"
    release_observed: str = "unknown"
    unsafe_visible: str = "unknown"
    evidence: str = ""
    sensor_availability: dict[str, bool] = field(default_factory=dict)
    recent_skill: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        _text(self.observation_id, "observation_id")
        if int(self.control_epoch) < 0:
            raise ValueError("control_epoch must be non-negative")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        for field_name in ("held", "placed", "target_visible", "release_observed", "unsafe_visible"):
            _tri(getattr(self, field_name), field_name)
        _reject_private(self.frames)
        _reject_private(self.recent_skill or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillResult:
    episode_id: str
    observation_id: str
    control_epoch: int
    skill_id: str
    execution_status: str
    outcome: str
    failure_code: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    safe_exit_complete: bool = True
    frames_before: list[str] = field(default_factory=list)
    frames_after: list[str] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        _text(self.observation_id, "observation_id")
        _text(self.skill_id, "skill_id")
        if int(self.control_epoch) < 0:
            raise ValueError("control_epoch must be non-negative")
        if self.execution_status not in {"completed", "guarded_abort", "safety_fault", "system_error"}:
            raise ValueError("invalid execution_status")
        if self.outcome not in {"succeeded", "failed", "unknown"}:
            raise ValueError("invalid outcome")
        _reject_private(self.feedback)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerAction:
    observation_id: str
    control_epoch: int
    action: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlannerAction":
        if not isinstance(raw, Mapping):
            raise ValueError("planner action must be a JSON object")
        allowed = {"observation_id", "control_epoch", "action", "args"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown planner keys: {sorted(unknown)}")
        observation_id = _text(raw.get("observation_id"), "observation_id")
        epoch = int(raw.get("control_epoch"))
        action = _text(raw.get("action"), "action").lower()
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        args = raw.get("args", {})
        if not isinstance(args, Mapping):
            raise ValueError("args must be an object")
        args = dict(args)
        _reject_private(args)
        if action in {"observe", "retract", "finish", "stop"} and args:
            raise ValueError(f"{action} does not accept args")
        if action == "place" and args != {"target": "hub"}:
            raise ValueError("place accepts only {target: hub}")
        if action == "pick" and args != {"target": "cover"}:
            raise ValueError("pick accepts only {target: cover}")
        if action == "help":
            required = {"request_type", "target", "request"}
            if set(args) != required or args.get("request_type") != "clear_target_area" or args.get("target") != "hub":
                raise ValueError("help accepts only clear_target_area(hub)")
            _text(args.get("request"), "help request", 800)
        return cls(observation_id, epoch, action, args)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HelpReport:
    help_id: str
    request_type: str
    target: str
    status: str
    report: str
    started_at: float
    ended_at: float
    control_epoch_before: int
    control_epoch_after: int
    frames: dict[str, str] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("frames", None)
        return value


__all__ = ["Observation", "SkillResult", "PlannerAction", "HelpReport", "ALLOWED_ACTIONS", "stable_hash"]
