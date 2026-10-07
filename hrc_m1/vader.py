"""VADER outcome descriptions for the current gearbox assembly skills."""

from __future__ import annotations

from typing import Any


def expected_outcome(config: dict[str, Any], skill_id: str) -> str:
    source = str(config.get("source_part", "the source part")).replace("_", " ")
    target = str(config.get("target_part", "the target part")).replace("_", " ")
    socket = str(config.get("target_frame", "the target socket")).replace("_", " ")
    outcomes = {
        "pick": f"The robot gripper visibly holds {source} and lifts it clear of the work surface.",
        "preinsert": f"The gripper holds {source} above {socket} on {target}, aligned for insertion but not yet seated.",
        "insert": f"{source} is visibly seated in {socket} on {target}; the robot still controls the part.",
        "verify": f"{source} remains visibly seated in {socket} on {target} without obvious tilt or separation.",
        "release_retract": f"{source} remains seated in {socket} on {target} after the gripper releases and withdraws.",
    }
    try:
        return outcomes[skill_id]
    except KeyError as exc:
        raise ValueError(f"no VADER outcome description for skill {skill_id!r}") from exc


def assessment_context(result: dict[str, Any], expected: str, verdict: str, evidence: str) -> dict[str, Any]:
    """Attach the visual postcondition to the last skill for LMP replanning."""
    return {
        **result,
        "expected_outcome": expected,
        "vqa_assessment": {"verdict": verdict, "evidence": evidence},
    }


__all__ = ["expected_outcome", "assessment_context"]
