"""Small, dependency-free contract/oracle for the constrained task MVP.

This is intentionally not the physical oracle.  It is the pre-Isaac contract
that checks the intended causal relation and prevents a malformed task from
being rendered or used for a model run.  The Isaac implementation must later
replace the transition checks with swept-volume, contact, reachability, and
terminal-predicate checks while preserving this public interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "pilot_12pair" / "config" / "constrained_tasks.json"


@dataclass(frozen=True)
class OracleResult:
    task_id: str
    variant: str
    sequence: tuple[str, ...]
    valid: bool
    complete: bool
    failure_step: int | None
    failure_reason: str | None
    events: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "sequence": list(self.sequence),
            "valid": self.valid,
            "complete": self.complete,
            "failure_step": self.failure_step,
            "failure_reason": self.failure_reason,
            "events": list(self.events),
        }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def task_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in config["tasks"]}


def _variant(task: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    try:
        return task["variants"][variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant {variant!r} for {task['task_id']}") from exc


def expected_feasible_sequences(task: Mapping[str, Any], variant: str) -> set[tuple[str, ...]]:
    """Return the four terminal candidate labels implied by the contract."""
    can_reverse = bool(_variant(task, variant)["can_b_before_a"])
    return {
        ("A", "B"),
        *( [("B", "A")] if can_reverse else [] ),
    }


def evaluate_sequence(
    task: Mapping[str, Any], variant: str, sequence: Iterable[str]
) -> OracleResult:
    """Evaluate one terminal A/B candidate against the pre-Isaac relation.

    The evaluator deliberately rejects duplicate/unknown actions and requires
    both actions at termination.  It never adds a recovery action implicitly.
    """
    seq = tuple(sequence)
    variant_cfg = _variant(task, variant)
    events: list[str] = []
    seen: set[str] = set()
    allowed = {"A", "B"}
    for index, action in enumerate(seq):
        if action not in allowed:
            return OracleResult(task["task_id"], variant, seq, False, False, index,
                                f"unknown action {action!r}", tuple(events))
        if action in seen:
            return OracleResult(task["task_id"], variant, seq, False, False, index,
                                f"duplicate action {action}", tuple(events))
        if action == "B" and "A" not in seen and not variant_cfg["can_b_before_a"]:
            reason = variant_cfg["feasibility_reason"]
            return OracleResult(task["task_id"], variant, seq, False, False, index,
                                reason, tuple(events))
        seen.add(action)
        event = task["events"][action]
        events.append(event)

    complete = seen == allowed
    if not complete:
        missing = ", ".join(sorted(allowed - seen))
        return OracleResult(task["task_id"], variant, seq, False, False, None,
                            f"terminal goal incomplete; missing {missing}", tuple(events))
    return OracleResult(task["task_id"], variant, seq, True, True, None, None, tuple(events))


def validate_config(config: Mapping[str, Any], repo_root: Path = ROOT) -> list[str]:
    """Return all contract errors; an empty list means the pack is coherent."""
    errors: list[str] = []
    shared = config.get("shared_contract", {})
    tolerance = float(shared.get("calibration_tolerance_mm", 0.0))
    min_margin = float(shared.get("minimum_intervention_margin_mm", 0.0))
    if tolerance <= 0:
        errors.append("shared calibration tolerance must be positive")
    if min_margin < 2 * tolerance:
        errors.append("minimum intervention margin must be at least twice tolerance")

    tasks = config.get("tasks", [])
    if len(tasks) != 5:
        errors.append(f"expected exactly five tasks, found {len(tasks)}")
    ids: set[str] = set()
    assets = config.get("source_assets", {})
    for task in tasks:
        tid = task.get("task_id")
        if not tid or tid in ids:
            errors.append(f"duplicate or missing task id: {tid!r}")
        ids.add(tid)
        if task.get("action_a", {}).get("id") != "A" or task.get("action_b", {}).get("id") != "B":
            errors.append(f"{tid}: action ids must be A and B")
        if set(task.get("variants", {})) != {"HARD", "COMMUTABLE"}:
            errors.append(f"{tid}: variants must be exactly HARD and COMMUTABLE")
        for part in task.get("parts", []):
            if part not in assets:
                errors.append(f"{tid}: missing source asset record for {part}")
            else:
                for role in ("visual", "measurement"):
                    path = repo_root / assets[part][role]
                    if not path.exists():
                        errors.append(f"{tid}: {role} asset does not exist: {path}")
        for name, variant in task.get("variants", {}).items():
            margin = float(variant.get("intervention_margin_mm", -1.0))
            if margin < 2 * tolerance:
                errors.append(f"{tid}/{name}: intervention margin {margin} is below 2*tolerance")
            if name == "HARD" and variant.get("can_b_before_a"):
                errors.append(f"{tid}/HARD: reverse order marked feasible")
            if name == "COMMUTABLE" and not variant.get("can_b_before_a"):
                errors.append(f"{tid}/COMMUTABLE: reverse order marked infeasible")

    # Mechanism-specific inequalities catch common visual/collision mistakes.
    by_id = task_index(config)
    hcf = by_id.get("HCF-01")
    if hcf:
        geo = hcf["geometry"]
        if not (geo["fastener_shaft_diameter_mm"] < geo["round_hole_diameter_mm"] < geo["fastener_head_diameter_mm"] < geo["keyhole_lobe_diameter_mm"]):
            errors.append("HCF-01: shaft < round hole < head < keyhole lobe must hold")
        if geo["head_underside_gap_mm"] <= geo["cover_flange_thickness_mm"]:
            errors.append("HCF-01: retained head leaves no axial cover clearance")
    wsg = by_id.get("WSG-01")
    if wsg:
        geo = wsg["geometry"]
        if geo["side_slot_width_mm"] <= geo["required_side_insertion_clearance_mm"]:
            errors.append("WSG-01: side slot has no insertion margin")
    key = by_id.get("KEY-01")
    if key:
        geo = key["geometry"]
        if geo["side_access_width_mm"] <= geo["key_width_mm"]:
            errors.append("KEY-01: side access window cannot admit the key")
    cas = by_id.get("CAS-01")
    if cas:
        geo = cas["geometry"]
        if geo["captive_pocket_depth_mm"] <= geo["head_clearance_mm"]:
            errors.append("CAS-01: captive pocket is shallower than head clearance")
    dow = by_id.get("DOW-01")
    if dow:
        geo = dow["geometry"]
        if geo["captive_slot_width_mm"] <= geo["dowel_diameter_mm"]:
            errors.append("DOW-01: captive slot cannot admit the dowel")
    return errors


def expected_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in config["tasks"]:
        for variant in ("HARD", "COMMUTABLE"):
            rows.append({
                "task_id": task["task_id"],
                "variant": variant,
                "AB": evaluate_sequence(task, variant, ("A", "B")).as_dict(),
                "BA": evaluate_sequence(task, variant, ("B", "A")).as_dict(),
                "A_only": evaluate_sequence(task, variant, ("A",)).as_dict(),
                "B_only": evaluate_sequence(task, variant, ("B",)).as_dict(),
            })
    return rows

