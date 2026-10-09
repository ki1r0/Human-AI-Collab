"""Append-only public evidence and fixed, observable features."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

from .contracts import ObservationPack, finite, public_only


class InsertionCheck:
    """Contact plus sustained lack of TCP progress; never a seating-success detector."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        names = ("window_s", "min_progress_mm", "contact_threshold_N", "target_tolerance_mm", "timeout_s")
        self.configured = all(settings.get(name) is not None for name in names)
        for name in names:
            if settings.get(name) is not None:
                self.settings[name] = finite(settings[name], name, minimum=0)
                if self.settings[name] <= 0:
                    raise ValueError(f"{name} must be positive")
        self.samples = deque()

    def update(self, elapsed_s: float, progress_mm: float | None, remaining_mm: float | None,
               contact_N: float | None, holding: str) -> dict[str, Any]:
        elapsed_s = finite(elapsed_s, "insertion elapsed", minimum=0)
        result = {"state": "UNKNOWN", "reason": "unconfigured_insertion_check", "elapsed_s": elapsed_s}
        if not self.configured:
            return result
        if any(value is None for value in (progress_mm, remaining_mm, contact_N)) or holding != "yes":
            self.samples.clear()
            result["reason"] = "missing_insertion_measurement_or_verified_grasp"
            return result
        progress_mm = finite(progress_mm, "TCP progress")
        remaining_mm = finite(remaining_mm, "remaining TCP travel")
        contact_N = finite(contact_N, "contact force", minimum=0)
        if self.samples and elapsed_s <= self.samples[-1][0]:
            raise ValueError("insertion sample times must increase")
        self.samples.append((elapsed_s, progress_mm, contact_N))
        cutoff = elapsed_s - self.settings["window_s"]
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        duration = elapsed_s - self.samples[0][0]
        progress = max(sample[1] for sample in self.samples) - self.samples[0][1]
        result.update(state="IN_PROGRESS", reason="collecting_progress_window",
                      tcp_axial_progress_mm=progress_mm, remaining_tcp_mm=remaining_mm,
                      contact_scalar_N=contact_N,
                      window_duration_s=duration, window_progress_mm=progress,
                      min_window_contact_N=min(sample[2] for sample in self.samples))
        # ponytail: TCP travel is only a proxy; add measured part depth if grasp slip matters.
        if abs(remaining_mm) <= self.settings["target_tolerance_mm"]:
            result["reason"] = "tcp_target_reached_not_seating_success"
        elif elapsed_s >= self.settings["timeout_s"]:
            result.update(state="BLOCKED", reason="insertion_timeout")
        elif (duration + 1e-9 >= self.settings["window_s"] and
              progress <= self.settings["min_progress_mm"] and
              result["min_window_contact_N"] >= self.settings["contact_threshold_N"]):
            result.update(state="BLOCKED", reason="sustained_contact_without_progress")
        elif duration + 1e-9 >= self.settings["window_s"]:
            result["reason"] = "no_sustained_contact_stall"
        return result


class Ledger:
    def __init__(self, sink=None) -> None:
        self._events: list[dict[str, Any]] = []
        self._hypotheses: list[dict[str, Any]] = []
        self._sink = sink

    def append_public(self, kind: str, payload: dict[str, Any]) -> str:
        public_only(payload)
        event_id = f"ev_{len(self._events):05d}"
        self._events.append(copy.deepcopy({"event_id": event_id, "kind": kind, "payload": payload}))
        if self._sink:
            self._sink(copy.deepcopy(self._events[-1]))
        return event_id

    def check_refs(self, refs: tuple[str, ...] | list[str]) -> None:
        ids = {event["event_id"] for event in self._events}
        if not set(refs) <= ids:
            raise ValueError("unknown evidence reference")

    def update_hypotheses(self, hypotheses: list[dict[str, Any]]) -> None:
        if len(hypotheses) > 3:
            raise ValueError("at most three concurrent hypotheses")
        for item in hypotheses:
            if set(item) - {"statement", "support_refs", "contradiction_refs", "prediction", "unknown"}:
                raise ValueError("unknown hypothesis fields")
            self.check_refs(item.get("support_refs", []))
            self.check_refs(item.get("contradiction_refs", []))
        public_only(hypotheses)
        self._hypotheses = copy.deepcopy(hypotheses)

    def public_context(self, *, compact_observations=False) -> dict[str, Any]:
        context = copy.deepcopy({"events": self._events, "hypotheses": self._hypotheses})
        if compact_observations:
            # Full recent packets are already in observation/outcome_window; raw events stay intact.
            for event in context["events"]:
                if event["kind"] == "observation":
                    event["payload"] = {name: event["payload"][name] for name in
                                        ("observation_id", "control_epoch", "physics_tick", "stage")
                                        if name in event["payload"]}
        return context


def features(obs: ObservationPack, now: float, freshness_s: float, ledger: Ledger) -> dict[str, Any]:
    def band(name: str, low: float, high: float) -> str:
        value = obs.value(name, now, freshness_s)
        if value is None:
            return "missing"
        return "low" if abs(float(value)) < low else "high" if abs(float(value)) > high else "middle"

    events = ledger.public_context()["events"]
    contact_calls = {e["payload"]["call_id"] for e in events if e["kind"] == "tool_call" and
                     e["payload"]["tool"] in {"seat_once", "probe_xy", "probe_angle", "probe_speed"}}
    results = [e["payload"] for e in events if e["kind"] == "tool_result" and
               e["payload"]["call_id"] in contact_calls and e["payload"]["control_epoch"] == obs.control_epoch]
    stalls = 0
    for result in reversed(results):
        if result.get("status") != "STALLED":
            break
        stalls += 1
    return {
        "stage": obs.stage,
        "holding": obs.value("holding", now, freshness_s) or "unknown",
        "progress_band": band("tcp_axial_delta_mm", 0.1, 1.0),
        "force_band": band("contact_scalar_N", 1.0, 10.0),
        "tracking_band": band("tracking_error_m", 0.001, 0.01),
        "consecutive_stalls": min(stalls, 3),
        "probe_history": [e["payload"]["candidate_id"] for e in events
                          if e["kind"] == "tool_call" and e["payload"].get("is_probe")],
    }


def monitor(obs: ObservationPack, now: float, freshness_s: float) -> dict[str, Any]:
    value = lambda name: obs.value(name, now, freshness_s)
    complete = all(value(name) is True for name in ("visual_seated", "released", "stable_observed"))
    lost = value("holding") == "no" and obs.stage == "prealigned" and value("released") is not True
    if not obs.valid:
        state, reason = "UNKNOWN", "invalid_observation"
    elif lost:
        state, reason = "BLOCKED", "grasp_lost"
    elif value("insertion_state") == "BLOCKED":
        state, reason = "BLOCKED", value("insertion_reason")
    elif complete:
        state, reason = "CANDIDATE_COMPLETE", "visual_seated_released_and_stable"
    elif value("insertion_state") == "UNKNOWN":
        state, reason = "UNKNOWN", value("insertion_reason")
    elif value("insertion_state") == "IN_PROGRESS":
        state, reason = "IN_PROGRESS", value("insertion_reason")
    elif any(value(name) is False for name in ("visual_seated", "released", "stable_observed")):
        state, reason = "IN_PROGRESS", "completion_not_observed"
    else:
        state, reason = "UNKNOWN", "insufficient_completion_evidence"
    return {
        "state": state, "reason": reason,
        "reported_success": state == "CANDIDATE_COMPLETE",
        "grasp_lost": lost,
        "sensor_invalid": not obs.valid,
        "observation_id": obs.observation_id,
    }


def verify_after_help(obs: ObservationPack, *, epoch: int, after_tick: int,
                      now: float, freshness_s: float, tracking_limit_m: float) -> str:
    if obs.control_epoch != epoch or obs.physics_tick <= after_tick or not obs.valid:
        return "UNKNOWN"
    if now - obs.wall_timestamp > freshness_s:
        return "UNKNOWN"
    holding = obs.value("holding", now, freshness_s)
    error = obs.value("tracking_error_m", now, freshness_s)
    if holding == "no" or (error is not None and error > tracking_limit_m):
        return "FAIL"
    if holding != "yes" or error is None or obs.value("target_visible", now, freshness_s) is not True:
        return "UNKNOWN"
    return "PASS"
