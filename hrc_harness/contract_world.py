"""Deterministic interface test double. NOT a physics or policy validation backend."""

import time
import uuid

from .contracts import Channel, HelpReport, MeasuredCost, ObservationPack


class ContractWorld:
    hold_certified = True
    def __init__(self, condition="nominal", *, clock=time.monotonic):
        if condition not in {"nominal", "blocked", "offset", "grasp"}:
            raise ValueError("unknown test condition")
        self._condition, self.clock = condition, clock
        self.episode_id, self.tick, self.stage = uuid.uuid4().hex, 0, "initial"
        self._holding, self._seated, self._cleared = "no", False, False
        self._progress, self._force = 0.0, 0.0
        self.stopped = False
        self.partial_help = False
        self.invalid_after_help = False

    def observe(self, epoch):
        now = self.clock()
        values = {"holding": self._holding, "tracking_error_m": 0.0,
                  "contact_exposure_Ns": 0.0,
                  "contact_scalar_N": self._force, "tcp_axial_delta_mm": self._progress,
                  "target_visible": not self.invalid_after_help, "visual_seated": self._seated,
                  "released": self._seated, "stable_observed": self._seated}
        return ObservationPack(self.episode_id, uuid.uuid4().hex, epoch, self.tick, now, self.stage,
            {name: Channel(value, "Ns" if name == "contact_exposure_Ns" else "N" if name == "contact_scalar_N" else
                           "mm" if name == "tcp_axial_delta_mm" else
                           "m" if name == "tracking_error_m" else "state", now, "contract_test_double")
             for name, value in values.items()})

    def idle_step(self):
        self.tick += 1

    def safe_hold(self):
        self.stopped = True

    def quiesce(self, check):
        check()
        self.idle_step()

    def protection(self, safety, *, motion=False):
        return None

    def execute(self, spec, check):
        for _ in range(4):
            check()
            self.idle_step()
            check()
        if spec.tool == "pick_and_prealign":
            self.stage, self._holding = "prealigned", "yes"
        if spec.tool in {"seat_once", "probe_xy", "probe_angle", "probe_speed"}:
            failed = self._condition != "nominal" and not self._cleared
            if spec.tool == "probe_xy" and self._condition == "offset":
                failed = False
            if failed:
                self._force, self._progress = 12.0, 0.0
                return "STALLED", "measured_insufficient_progress", {"tcp_axial_delta_mm": 0.0}, MeasuredCost()
            self.stage, self._holding, self._seated = "released", "no", True
            self._force, self._progress = 2.0, 2.0
        return "COMPLETED", "bounded_tool_complete", {}, MeasuredCost()

    def help(self, request, check):
        if (request.target, request.operation, request.allowed_scope) != (
                "socket_hub_output", "clear_target_area", "target_area_only"):
            return HelpReport("REJECTED", "none", "Outside authorized scope", 0, 0)
        for _ in range(4):
            check()
            self.idle_step()
        if not self.partial_help:
            self._cleared = True
        return HelpReport("COMPLETED", request.operation, "Intervention attempted; verify observations.", 0, 1)

    def evaluate(self):
        return {"backend": "contract_only", "seat_gt": "PASS" if self._seated else "FAIL",
                "registration_gt": "UNKNOWN", "condition": self._condition,
                "physical_validation": False}
