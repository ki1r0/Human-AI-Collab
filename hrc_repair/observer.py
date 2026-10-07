"""Public observer boundary for simulator skill outcomes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .contracts import Observation, SkillResult


class MetricsObserver:
    """Derive a tri-state public observation from an adapter result.

    The current workstation has not yet exposed a fully sensor-only observer
    for the M0 preplace skill.  Therefore this implementation is explicitly
    marked ``privileged_debug`` by the config and never presents its private
    pose/force fields to the planner.
    """

    def __init__(self, *, feedback_mode: str = "privileged_debug") -> None:
        self.feedback_mode = feedback_mode
        self._counter = 0

    def reset(self, episode_id: str, control_epoch: int, *, frames: dict[str, str] | None = None) -> Observation:
        return Observation(
            episode_id=episode_id,
            observation_id=f"obs_{self._counter:04d}",
            control_epoch=control_epoch,
            timestamp=time.time(),
            frames=frames or {},
            held="yes",
            placed="unknown",
            target_visible="yes",
            release_observed="no",
            unsafe_visible="no",
            evidence="Preplace setup is registered; placement has not been attempted.",
            sensor_availability={"rgb": bool(frames), "robot_state": True, "contact_or_guard": True},
        )

    def after_skill(self, episode_id: str, control_epoch: int, result: SkillResult, *, metrics_path: Path | None = None, frames: dict[str, str] | None = None) -> Observation:
        self._counter += 1
        metrics = {}
        if self.feedback_mode == "privileged_debug" and metrics_path and metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = {}
        public_sensor = result.feedback.get("public_sensor", {}) if isinstance(result.feedback.get("public_sensor", {}), dict) else {}
        score = metrics.get("physical_place_score", {}) if isinstance(metrics, dict) else {}
        score_total = int(score.get("total", 0)) if isinstance(score, dict) else 0
        placement = metrics.get("placement_score", {}) if isinstance(metrics, dict) else {}
        placement_success = bool(isinstance(placement, dict) and placement.get("success", False))
        placement_success = placement_success or bool(result.feedback.get("placement_score", {}).get("success", False)) if isinstance(result.feedback.get("placement_score", {}), dict) else placement_success
        verdict = str(metrics.get("insertion_verdict", "")) if isinstance(metrics, dict) else ""
        # The contract adapter has no private physics artifact; its outcome is
        # deliberately labelled scripted_debug in the config.  The Isaac
        # adapter must still satisfy an independent physical score.
        contract_success = result.outcome == "succeeded" and "contract backend" in str(result.feedback.get("public_evidence", ""))
        if self.feedback_mode.startswith("public_"):
            success = bool(public_sensor.get("placement_observed", False))
        else:
            success = result.outcome == "succeeded" and ("SUCCESS" in verdict or score_total >= 4 or placement_success or contract_success)
        failure = result.outcome == "failed" or result.execution_status in {"guarded_abort", "safety_fault", "system_error"}
        placed = "yes" if success else "no" if failure else "unknown"
        held = "no" if public_sensor.get("released", False) or (result.skill_id in {"place", "retract"} and success) else "yes" if result.skill_id != "place" or public_sensor.get("blocked_guard", False) else "unknown"
        release = "yes" if public_sensor.get("released", False) else "unknown" if result.skill_id == "place" else "no"
        unsafe = "yes" if result.execution_status in {"safety_fault", "guarded_abort"} else "no"
        evidence = str(result.feedback.get("public_evidence", result.failure_code or result.outcome))
        if self.feedback_mode.startswith("public_") and public_sensor:
            evidence = (
                f"{evidence}; public sensors: "
                f"support_contact={bool(public_sensor.get('support_contact'))}, "
                f"release_contact_free={bool(public_sensor.get('release_contact_free'))}, "
                f"released={bool(public_sensor.get('released'))}; "
                f"placed={'yes' if success else 'no'}, "
                f"release_observed={'yes' if public_sensor.get('released') else 'unknown'}"
            )
        return Observation(
            episode_id=episode_id,
            observation_id=f"obs_{self._counter:04d}",
            control_epoch=control_epoch,
            timestamp=time.time(),
            frames=frames or {},
            held=held,
            placed=placed,
            target_visible="yes",
            release_observed=release,
            unsafe_visible=unsafe,
            evidence=evidence,
            sensor_availability={"rgb": bool(frames), "robot_state": True, "contact_or_guard": True},
            recent_skill=result.to_dict(),
        )


__all__ = ["MetricsObserver"]
