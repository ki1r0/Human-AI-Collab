"""Typed public data; simulator truth never belongs in these objects."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from hrc_repair.contracts import _reject_private as _legacy_reject_private


def public_only(value: Any) -> None:
    _legacy_reject_private(value)
    forbidden = {"scenario_family", "condition", "true_offset", "blocker_present",
                 "helper_gt_success", "reward", "score", "metrics_path", "private_gt",
                 "seat_gt", "registration_gt", "primary_gt", "terminal_success",
                 "initial_category", "private_reset", "penetration_m"}
    if isinstance(value, Mapping):
        if forbidden.intersection(value):
            raise ValueError("private fields cannot enter public context")
        for item in value.values():
            public_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            public_only(item)


def finite(value: float, name: str, *, minimum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True)
class Channel:
    value: float | bool | str | list[float] | None
    units: str
    sensor_timestamp: float
    source: str
    valid: bool = True

    def __post_init__(self) -> None:
        finite(self.sensor_timestamp, "sensor timestamp")
        if isinstance(self.value, float):
            finite(self.value, "channel value")
        if isinstance(self.value, list):
            for value in self.value:
                finite(value, "channel vector")
        if not self.units or not self.source:
            raise ValueError("channel units/source are required")


CHANNELS = {
    "qpos", "qvel", "tcp_xyz_m", "tcp_quat_wxyz", "tracking_error_m",
    "gripper_opening_m", "gripper_contact_N", "contact_scalar_N", "contact_exposure_Ns", "holding",
    "tcp_axial_delta_mm", "part_depth_est_mm", "tilt_est_deg", "target_visible",
    "visual_seated", "released", "stable_observed",
    "insertion_state", "insertion_reason", "insertion_elapsed_s",
}


@dataclass(frozen=True)
class ObservationPack:
    episode_id: str
    observation_id: str
    control_epoch: int
    physics_tick: int
    wall_timestamp: float
    stage: str
    channels: Mapping[str, Channel] = field(default_factory=dict)
    # Paths stay at the image-loading boundary; contexts contain opaque frame IDs.
    frames: Mapping[str, str] = field(default_factory=dict)
    frame_timestamps: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.episode_id or not self.observation_id or not self.stage:
            raise ValueError("observation identifiers/stage are required")
        if self.control_epoch < 0 or self.physics_tick < 0:
            raise ValueError("negative epoch/tick")
        finite(self.wall_timestamp, "observation timestamp")
        if set(self.channels) - CHANNELS:
            raise ValueError(f"unregistered public channels: {set(self.channels) - CHANNELS}")
        if set(self.frames) != set(self.frame_timestamps):
            raise ValueError("every image requires its acquisition timestamp")
        for stamp in self.frame_timestamps.values():
            finite(stamp, "image timestamp")

    def value(self, name: str, now: float, freshness_s: float) -> Any:
        channel = self.channels.get(name)
        if channel is None or not channel.valid or channel.value is None:
            return None
        age = now - channel.sensor_timestamp
        return channel.value if -1e-6 <= age <= freshness_s else None

    def public_dict(self, now: float, freshness_s: float) -> dict[str, Any]:
        result = {
            "episode_id": self.episode_id, "observation_id": self.observation_id,
            "control_epoch": self.control_epoch, "physics_tick": self.physics_tick,
            "wall_timestamp": self.wall_timestamp, "stage": self.stage, "valid": self.valid,
            "channels": {
                name: {**asdict(channel), "stale": self.value(name, now, freshness_s) is None}
                for name, channel in self.channels.items()
            },
            "frames": {
                alias: {"frame_id": f"{self.observation_id}:{alias}",
                        "sensor_timestamp": self.frame_timestamps[alias],
                        "stale": not (-1e-6 <= now - self.frame_timestamps[alias] <= freshness_s)}
                for alias in self.frames
            },
        }
        public_only(result)
        return result


@dataclass(frozen=True)
class Resources:
    contacts: int = 0
    probes: int = 0
    inspections: int = 0
    helps: int = 0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("contacts", "probes", "inspections", "helps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid resource {name}")
        finite(self.duration_s, "duration", minimum=0)

    def __add__(self, other: Resources) -> Resources:
        return Resources(**{name: getattr(self, name) + getattr(other, name)
                            for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class ToolSpec:
    candidate_id: str
    tool: str
    profile: str = ""
    resources: Resources = field(default_factory=Resources)
    certified: bool = False
    requires_holding: bool = False
    verification: bool = False


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    candidate_id: str
    control_epoch: int
    based_on_observation: str
    args: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x for x in
                   (self.call_id, self.candidate_id, self.based_on_observation)):
            raise ValueError("tool identifiers are required")
        if type(self.control_epoch) is not int or self.control_epoch < 0:
            raise ValueError("invalid control epoch")
        public_only(dict(self.args))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ToolCall:
        allowed = set(cls.__dataclass_fields__)
        if set(raw) - allowed:
            raise ValueError("unknown tool call fields")
        public_only(raw)
        return cls(**{**raw, "evidence_refs": tuple(raw.get("evidence_refs", ()))})


@dataclass(frozen=True)
class MeasuredCost:
    # Wall time is measured by the runtime; these are components, never added twice.
    motion_s: float = 0.0
    helper_s: float = 0.0
    model_s: float = 0.0
    contact_exposure_Ns: float = 0.0
    helper_effort: float = 0.0
    model_money: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name, minimum=0)

    def __add__(self, other: MeasuredCost) -> MeasuredCost:
        return MeasuredCost(**{name: getattr(self, name) + getattr(other, name)
                               for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    status: str
    exit_reason: str
    actual_motion: Mapping[str, float]
    observation_id_after: str
    control_epoch: int
    measured_cost: MeasuredCost = field(default_factory=MeasuredCost)
    public_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETED", "STALLED", "ABORTED", "REJECTED", "UNKNOWN"}:
            raise ValueError("invalid tool result status")
        for value in self.actual_motion.values():
            finite(value, "actual motion")


@dataclass(frozen=True)
class HelpRequest:
    request_id: str
    target: str
    operation: str
    allowed_scope: str
    desired_postconditions: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    observed_problem: str


@dataclass(frozen=True)
class HelpReport:
    status: str
    performed_operation: str
    report: str
    elapsed_s: float
    effort: float

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETED", "REJECTED", "TIMEOUT", "UNKNOWN"}:
            raise ValueError("invalid helper status")
        finite(self.elapsed_s, "helper elapsed", minimum=0)
        finite(self.effort, "helper effort", minimum=0)
