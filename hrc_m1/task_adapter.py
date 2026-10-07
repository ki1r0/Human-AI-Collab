"""Task-adapter interfaces; simulator imports remain optional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import Observation, SkillResult, new_observation_id
from .evaluator import SeatMeasurement


class TaskAdapter(Protocol):
    backend_name: str

    def reset(self, episode_id: str, seed: int, mode: str) -> Observation: ...
    def observe(self, episode_id: str, index: int, recent_skill: dict[str, Any] | None = None) -> Observation: ...
    def execute_skill(self, observation: Observation, skill_id: str, args: dict[str, Any]) -> SkillResult: ...
    def evaluator_measurement(self) -> SeatMeasurement: ...
    def close(self) -> None: ...


@dataclass
class MockTaskAdapter:
    """Contract-only backend.  Never reports calibrated task success."""

    backend_name: str = "mock_contract_only"
    _episode_id: str = ""
    _index: int = 0
    _recent: dict[str, Any] | None = None

    def reset(self, episode_id: str, seed: int, mode: str) -> Observation:
        self._episode_id = episode_id
        self._index = 0
        self._recent = None
        return self.observe(episode_id, 0)

    def observe(self, episode_id: str, index: int, recent_skill: dict[str, Any] | None = None) -> Observation:
        self._index = int(index)
        self._recent = recent_skill
        return Observation(episode_id=episode_id, observation_id=new_observation_id(episode_id, index), timestamp=float(index), frames={}, qpos=[], qvel=[], gripper={"left": 0.0, "right": 0.0}, sensor_availability={"rgb": False, "force": False}, recent_skill=recent_skill)

    def execute_skill(self, observation: Observation, skill_id: str, args: dict[str, Any]) -> SkillResult:
        status = "FAILED" if skill_id == "insert" else "SUCCEEDED"
        result = SkillResult(observation.episode_id, observation.observation_id, skill_id, status, held_status="UNKNOWN", seat_status="UNKNOWN", steps=1, failure_code="MOCK_ONLY" if status == "FAILED" else None)
        self._index += 1
        self._recent = result.to_dict()
        return result

    def evaluator_measurement(self) -> SeatMeasurement:
        return SeatMeasurement()

    def close(self) -> None:
        return None


__all__ = ["MockTaskAdapter", "TaskAdapter"]
