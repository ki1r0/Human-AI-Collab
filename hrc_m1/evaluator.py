"""Evaluator-side seating criteria, isolated from planner-visible state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SeatTolerances:
    axial_depth_m: float = 0.062
    axial_tolerance_m: float = 0.008
    radial_tolerance_m: float = 0.004
    tilt_tolerance_deg: float = 2.0
    max_penetration_m: float = 0.001
    settle_speed_mps: float = 0.01
    settle_window_s: float = 0.5


@dataclass(frozen=True)
class SeatMeasurement:
    axial_depth_m: float | None = None
    radial_error_m: float | None = None
    tilt_deg: float | None = None
    penetration_m: float | None = None
    released: bool = False
    stable: bool = False
    contact_valid: bool = False
    settle_speed_mps: float | None = None
    settle_window_s: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeatMeasurement":
        allowed = {
            "axial_depth_m", "radial_error_m", "tilt_deg", "penetration_m",
            "released", "stable", "contact_valid", "settle_speed_mps", "settle_window_s",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown evaluator measurement fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name in allowed:
            if name in value:
                kwargs[name] = value[name]
        for name in ("axial_depth_m", "radial_error_m", "tilt_deg", "penetration_m", "settle_speed_mps", "settle_window_s"):
            if name in kwargs and kwargs[name] is not None:
                kwargs[name] = float(kwargs[name])
        for name in ("released", "stable", "contact_valid"):
            if name in kwargs:
                kwargs[name] = bool(kwargs[name])
        return cls(**kwargs)


@dataclass(frozen=True)
class EvaluationResult:
    task_success: bool
    physical_criteria_passed: bool
    status: str
    calibration_status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "physical_criteria_passed": self.physical_criteria_passed,
            "status": self.status,
            "calibration_status": self.calibration_status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


class IndependentSeatEvaluator:
    """Evaluate physical measurements supplied by the simulator-side evaluator.

    A pending calibration can produce a provisional physical result but never a
    public M1 task success.  The planner only receives a tri-state verifier
    result, not these measurements or reasons.
    """

    def __init__(self, tolerances: SeatTolerances | None = None, *, calibration_status: str = "pending") -> None:
        self.tolerances = tolerances or SeatTolerances()
        if calibration_status not in {"pending", "calibrated", "provisional"}:
            raise ValueError("calibration_status must be pending, provisional or calibrated")
        self.calibration_status = calibration_status

    def evaluate(self, measurement: SeatMeasurement | Mapping[str, Any]) -> EvaluationResult:
        if not isinstance(measurement, SeatMeasurement):
            measurement = SeatMeasurement.from_mapping(measurement)
        tol = self.tolerances
        reasons: list[str] = []
        checks: dict[str, bool] = {}

        if measurement.axial_depth_m is None:
            checks["axial"] = False
            reasons.append("axial_depth_unavailable")
        else:
            checks["axial"] = abs(measurement.axial_depth_m - tol.axial_depth_m) <= tol.axial_tolerance_m
            if not checks["axial"]:
                reasons.append("axial_depth_out_of_tolerance")
        if measurement.radial_error_m is None:
            checks["radial"] = False
            reasons.append("radial_error_unavailable")
        else:
            checks["radial"] = abs(measurement.radial_error_m) <= tol.radial_tolerance_m
            if not checks["radial"]:
                reasons.append("radial_error_out_of_tolerance")
        if measurement.tilt_deg is None:
            checks["tilt"] = False
            reasons.append("tilt_unavailable")
        else:
            checks["tilt"] = abs(measurement.tilt_deg) <= tol.tilt_tolerance_deg
            if not checks["tilt"]:
                reasons.append("tilt_out_of_tolerance")
        if measurement.penetration_m is None:
            checks["penetration"] = False
            reasons.append("penetration_unavailable")
        else:
            checks["penetration"] = measurement.penetration_m <= tol.max_penetration_m
            if not checks["penetration"]:
                reasons.append("penetration_exceeded")
        checks["contact"] = bool(measurement.contact_valid)
        checks["released"] = bool(measurement.released)
        checks["stable"] = bool(measurement.stable)
        if not checks["contact"]:
            reasons.append("valid_contact_not_observed")
        if not checks["released"]:
            reasons.append("not_released")
        if not checks["stable"]:
            reasons.append("not_stable")
        if measurement.settle_speed_mps is not None and measurement.settle_speed_mps > tol.settle_speed_mps:
            reasons.append("settle_speed_exceeded")
            checks["settle_speed"] = False
        else:
            checks["settle_speed"] = measurement.settle_speed_mps is not None
            if measurement.settle_speed_mps is None:
                reasons.append("settle_speed_unavailable")
        # The stability accumulator advances in fixed simulation quanta.  Do
        # not turn an exact boundary hit (e.g. 0.49999999999999994 for a
        # requested 0.5 s) into a false failure solely because of float
        # representation.
        if measurement.settle_window_s is not None and measurement.settle_window_s + 1.0e-9 < tol.settle_window_s:
            reasons.append("settle_window_too_short")
            checks["settle_window"] = False
        else:
            checks["settle_window"] = measurement.settle_window_s is not None
            if measurement.settle_window_s is None:
                reasons.append("settle_window_unavailable")

        physical = all(checks.values())
        unavailable = any(reason.endswith("_unavailable") for reason in reasons)
        if unavailable:
            status = "UNKNOWN"
            task_success = False
            physical = False
        elif physical and self.calibration_status == "calibrated":
            status = "PASS"
            task_success = True
        elif physical:
            status = "PROVISIONAL_PASS"
            task_success = False
            reasons.append("calibration_not_final")
        else:
            status = "FAIL"
            task_success = False
        metrics = {
            "axial_depth_m": measurement.axial_depth_m,
            "radial_error_m": measurement.radial_error_m,
            "tilt_deg": measurement.tilt_deg,
            "penetration_m": measurement.penetration_m,
            "released": measurement.released,
            "stable": measurement.stable,
            "contact_valid": measurement.contact_valid,
            "settle_speed_mps": measurement.settle_speed_mps,
            "settle_window_s": measurement.settle_window_s,
        }
        return EvaluationResult(task_success, physical, status, self.calibration_status, tuple(reasons), metrics)


__all__ = ["EvaluationResult", "IndependentSeatEvaluator", "SeatMeasurement", "SeatTolerances"]
