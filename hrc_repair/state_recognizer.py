"""Task-adapted RGB state recognition for the REPAIR pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import Observation


def expected_place_outcome(config: dict[str, Any]) -> str:
    scene = config.get("scene", {})
    cover = str(scene.get("cover_prim", "Hub_Cover_Output_Top")).rstrip("/").split("/")[-1].replace("_", " ")
    casing = str(scene.get("hub_prim", "Casing_Top")).rstrip("/").split("/")[-1].replace("_", " ")
    socket = str(scene.get("registered_goal_frame", "socket_hub_output")).replace("_", " ")
    return (
        f"{cover} is visibly seated and centered on {casing} around {socket}: its broad circular flange "
        "rests on the casing, the cover and socket centers are concentric, and the flange bolt holes "
        "visibly align with their corresponding casing holes wherever the views expose them. "
        "Its raised center opening is part of the cover. Judge only visible part-to-casing seating and alignment; "
        "a separate ring resting on the tabletop or its staging support is not seated, and a visible gap "
        "between the flange and casing is a failure. Require direct visual evidence of the flange resting "
        "on the casing and of concentricity/alignment. A clearly off-center or rotated cover is a failure. "
        "If perspective or occlusion prevents judging centering or hole alignment, return UNKNOWN; do not "
        "infer alignment from contact, proximity, or the expected description. "
        "Gripper release is checked separately using public gripper-state and contact signals."
    )


def expected_initial_state(config: dict[str, Any]) -> str:
    scene = config.get("scene", {})
    cover = str(scene.get("cover_prim", "Hub_Cover_Output_Top")).rstrip("/").split("/")[-1].replace("_", " ")
    casing = str(scene.get("hub_prim", "Casing_Top")).rstrip("/").split("/")[-1].replace("_", " ")
    return f"The {cover} rests on its physical tabletop support, separate from and not yet placed on {casing}; the robot is not holding it."


def expected_pick_outcome(config: dict[str, Any]) -> str:
    cover = str(config.get("scene", {}).get("cover_prim", "Hub_Cover_Output_Top")).rstrip("/").split("/")[-1].replace("_", " ")
    return f"The robot is visibly holding the {cover} after lifting it clear of its physical support; the cover is not yet seated on the casing."


def _assess(verifier: Any, observation: Observation, frame_paths: list[str], expected: str, *, phase: str, all_views: bool = False):
    from hrc_m1.contracts import Observation as VisualObservation

    if phase == "place" and not all_views:
        release_confirmed = observation.placed == "yes" and observation.release_observed == "yes"
        preferred_camera = "right_hand_rgb" if release_confirmed else "head_rgb"
        preferred_views = [str(path) for path in frame_paths if preferred_camera in Path(path).name]
        head_views = [str(path) for path in frame_paths if "head_rgb" in Path(path).name]
        selected_paths = preferred_views or head_views or frame_paths
    else:
        selected_paths = frame_paths
    frames = {
        f"view_{index}": str(path)
        for index, path in enumerate(selected_paths)
        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"} and Path(path).is_file()
    }
    visual_observation = VisualObservation(
        observation.episode_id,
        observation.observation_id,
        observation.timestamp,
        frames=frames,
    )
    result = verifier.verify(visual_observation, expected_outcome=expected)
    recent = dict(observation.recent_skill or {})
    recent["rgb_state_observation"] = phase
    recent["expected_outcome"] = expected
    recent["vqa_assessment"] = {"verdict": result.verdict.value, "evidence": result.evidence}
    evidence = f"RGB {phase} assessment: {result.verdict.value}; {result.evidence}"
    if phase == "initial" and result.verdict.value == "SUCCESS":
        updated = replace(
            observation, frames={}, held="no", placed="no", target_visible="yes",
            release_observed="no", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "initial":
        updated = replace(
            observation, frames={}, held="unknown", placed="unknown", target_visible="unknown",
            release_observed="unknown", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "after_help" and result.verdict.value == "SUCCESS":
        updated = replace(
            observation, frames={}, held="yes", placed="no", target_visible="yes",
            release_observed="no", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "after_help":
        updated = replace(
            observation, frames={}, held="unknown", placed="unknown", target_visible="unknown",
            release_observed="unknown", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "pick" and result.verdict.value == "SUCCESS":
        updated = replace(
            observation, frames={}, held="yes", placed="no", target_visible="yes",
            release_observed="no", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "pick":
        updated = replace(
            observation, frames={}, held="unknown", placed="no", target_visible="unknown",
            release_observed="unknown", evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    elif phase == "place":
        if result.verdict.value == "SUCCESS":
            public_sensor = ((observation.recent_skill or {}).get("feedback") or {}).get("public_sensor", {})
            has_placement_sensor = isinstance(public_sensor, dict) and "placement_observed" in public_sensor
            blocked_guard = isinstance(public_sensor, dict) and bool(public_sensor.get("blocked_guard"))
            sensor_confirmed = (
                isinstance(public_sensor, dict)
                and not blocked_guard
                and (not has_placement_sensor or bool(public_sensor.get("placement_observed")))
            )
            has_release_sensors = isinstance(public_sensor, dict) and {
                "released", "release_contact_free"
            }.issubset(public_sensor)
            release_confirmed = (
                bool(public_sensor.get("released")) and bool(public_sensor.get("release_contact_free"))
                if has_release_sensors else sensor_confirmed
            )
            updated = replace(
                observation, frames={}, held="no" if release_confirmed else "unknown",
                placed="yes" if sensor_confirmed else "no" if blocked_guard or has_placement_sensor else "unknown",
                target_visible="yes", release_observed="yes" if release_confirmed else "no" if has_release_sensors else "unknown",
                evidence=evidence, recent_skill=recent,
                sensor_availability={**observation.sensor_availability, "rgb": True},
            )
        elif result.verdict.value == "FAILED":
            public_sensor = ((observation.recent_skill or {}).get("feedback") or {}).get("public_sensor", {})
            has_release_sensors = isinstance(public_sensor, dict) and {
                "released", "release_contact_free"
            }.issubset(public_sensor)
            release_confirmed = bool(public_sensor.get("released")) and bool(public_sensor.get("release_contact_free")) if has_release_sensors else False
            updated = replace(
                observation, frames={}, held="no" if release_confirmed else "unknown", placed="no",
                release_observed="yes" if release_confirmed else "no" if has_release_sensors else "unknown",
                evidence=evidence, recent_skill=recent,
                sensor_availability={**observation.sensor_availability, "rgb": True},
            )
        else:
            updated = replace(
                observation, frames={}, placed="unknown", release_observed="unknown",
                evidence=evidence, recent_skill=recent,
                sensor_availability={**observation.sensor_availability, "rgb": True},
            )
    else:
        updated = replace(
            observation, frames={}, evidence=evidence, recent_skill=recent,
            sensor_availability={**observation.sensor_availability, "rgb": True},
        )
    return updated, result


def assess_initial(verifier: Any, observation: Observation, frame_paths: list[str], expected: str):
    return _assess(verifier, observation, frame_paths, expected, phase="initial")


def assess_help(verifier: Any, observation: Observation, frame_paths: list[str], expected: str):
    return _assess(verifier, observation, frame_paths, expected, phase="after_help")


def assess_pick(verifier: Any, observation: Observation, frame_paths: list[str], expected: str):
    return _assess(verifier, observation, frame_paths, expected, phase="pick")


def assess_place(verifier: Any, observation: Observation, frame_paths: list[str], expected: str, *, all_views: bool = False):
    return _assess(verifier, observation, frame_paths, expected, phase="place", all_views=all_views)


def planner_observation(observation: Observation) -> Observation:
    """Keep the current turn IDs unambiguous; current observation has the needed assessment."""
    if not observation.recent_skill:
        return replace(observation, frames={})
    recent = dict(observation.recent_skill)
    recent.pop("observation_id", None)
    recent.pop("control_epoch", None)
    return replace(observation, frames={}, recent_skill=recent)


def visual_success(observation: Observation) -> bool:
    recent = observation.recent_skill or {}
    assessment = recent.get("vqa_assessment", {})
    return isinstance(assessment, dict) and assessment.get("verdict") == "SUCCESS"


__all__ = [
    "assess_help", "assess_initial", "assess_pick", "assess_place", "expected_initial_state",
    "expected_pick_outcome", "expected_place_outcome", "planner_observation", "visual_success",
]
