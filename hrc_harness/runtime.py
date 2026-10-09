"""Single-owner, interruptible tool execution with shared resource accounting."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, replace
from typing import Any

from .contracts import Channel, HelpRequest, MeasuredCost, Resources, ToolCall, ToolResult, ToolSpec
from .evidence import Ledger, monitor, verify_after_help


TOOLS = {"observe", "pick_and_prealign", "seat_once", "inspect", "probe_xy",
         "probe_angle", "probe_speed", "ask_act", "stop", "finish"}
CONTACT_TOOLS = {"seat_once", "probe_xy", "probe_angle", "probe_speed"}


class Interrupted(RuntimeError):
    pass


class Runtime:
    def __init__(self, adapter: Any, config: dict[str, Any], *, clock=time.monotonic, sink=None) -> None:
        self.adapter, self.config, self.clock = adapter, config, clock
        self.started = clock()
        self.ledger = Ledger(sink)
        self.epoch, self.owner = 0, "robot"
        self.cancel = threading.Event()
        self.calls: set[str] = set()
        self.used = Resources()
        self.cost = MeasuredCost()
        self.help_mode = config.get("help_mode", "request_only")
        if self.help_mode not in {"request_only", "intervene_resume"}:
            raise ValueError("unknown help mode")
        self.help_request = None
        self.terminal: str | None = None
        self.verification = "PASS"
        self.help_tick: int | None = None
        self.observation = None
        self.visual_facts = {}
        self.image_history = []
        self.registry = {}
        for raw in config["tools"]:
            spec = ToolSpec(**{**raw, "resources": Resources(**raw.get("resources", {}))})
            if spec.tool not in TOOLS or spec.candidate_id in self.registry:
                raise ValueError("invalid/duplicate registry entry")
            if spec.tool in CONTACT_TOOLS and spec.resources.contacts < 1:
                raise ValueError("every seating/probe must charge contact attempts")
            if spec.tool.startswith("probe_") and spec.resources.probes < 1:
                raise ValueError("probe budget cannot be bypassed")
            if spec.tool == "ask_act" and spec.resources.helps < 1:
                raise ValueError("help must be charged")
            if spec.tool == "inspect" and spec.resources.inspections < 1:
                raise ValueError("inspection must be charged")
            self.registry[spec.candidate_id] = spec
        self.observe()

    @property
    def wall_s(self) -> float:
        return self.clock() - self.started

    def feasible(self, need: Resources) -> bool:
        limits = self.config["budget"]
        return (all(getattr(self.used, name) + getattr(need, name) <= limits[name]
                    for name in ("contacts", "probes", "inspections", "helps"))
                and self.wall_s + need.duration_s <= limits["wall_s"])

    @property
    def resume_resources(self):
        spec = self.registry.get(self.config.get("resume_candidate_id", "seat"))
        if spec is None or spec.tool != "seat_once":
            raise ValueError("help resume must name a registered seating tool")
        return replace(spec.resources, duration_s=max(spec.resources.duration_s, self.config["resume_duration_s"]))

    def _guard(self, *, motion: bool = False, deadline: float | None = None) -> None:
        if self.cancel.is_set():
            raise Interrupted("cancelled")
        if self.wall_s >= self.config["budget"]["wall_s"]:
            raise Interrupted("wall_budget")
        if deadline is not None and self.clock() >= deadline:
            raise Interrupted("tool_timeout")
        if motion and self.owner != "robot":
            raise Interrupted("ownership")
        reason = self.adapter.protection(self.config["safety"], motion=motion)
        if reason:
            raise Interrupted(reason)

    def observe(self):
        self.observation = self.adapter.observe(self.epoch)
        self.ledger.append_public("observation", self.observation.public_dict(
            self.clock(), self.config["freshness_s"]))
        self.ledger.append_public("monitor", monitor(self.decision_observation(), self.clock(), self.config["freshness_s"]))
        self.image_history = (self.image_history + [self.observation])[-3:]
        self._verify()
        return self.observation

    def _verify(self):
        if self.help_tick is not None:
            limit = self.config["safety"].get("tracking_limit_m")
            self.verification = "UNKNOWN" if limit is None else verify_after_help(
                self.decision_observation(), epoch=self.epoch, after_tick=self.help_tick,
                now=self.clock(), freshness_s=self.config["freshness_s"], tracking_limit_m=limit)
            self.ledger.append_public("verification", {"verdict": self.verification,
                                      "observation_id": self.observation.observation_id})
            if self.verification == "PASS":
                self.help_tick = None

    def decision_observation(self):
        """An explicit derived view; model estimates never rewrite raw sensor events."""
        channels = dict(self.observation.channels)
        for name, fact in self.visual_facts.items():
            if fact["epoch"] != self.epoch:
                continue
            raw = self.observation.value(name, self.clock(), self.config["freshness_s"])
            if raw not in (None, "unknown"):
                continue
            channels[name] = Channel(fact["value"], "state", fact["timestamp"], "multimodal_model_estimate")
        return replace(self.observation, channels=channels)

    def apply_visual_facts(self, facts):
        events = {row["event_id"]: row for row in self.ledger.public_context()["events"]}
        for fact in facts:
            referenced = [events[ref]["payload"] for ref in fact["evidence_refs"]
                          if events[ref]["kind"] == "observation"]
            matching = [obs for obs in referenced if obs["observation_id"] == fact["observation_id"]
                        and obs["control_epoch"] == self.epoch and obs["frames"]]
            if not matching:
                continue
            stamp = max(frame["sensor_timestamp"] for frame in matching[-1]["frames"].values())
            if not 0 <= self.clock() - stamp <= self.config["freshness_s"]:
                continue
            if fact["name"] == "stable_observed" and fact["value"] is True:
                valid = [obs for obs in referenced if obs["control_epoch"] == self.epoch and obs["frames"]]
                ticks = [obs["physics_tick"] for obs in valid]
                stamps = {frame["sensor_timestamp"] for obs in valid for frame in obs["frames"].values()}
                dt = self.config.get("task_snapshot", {}).get("scene", {}).get("physics_dt_s")
                dwell = self.config.get("task_snapshot", {}).get("evaluation", {}).get("settle_window_s")
                if not dt or not dwell or len(stamps) < 2 or (max(ticks)-min(ticks))*dt < dwell:
                    continue
            value = {"value": fact["value"], "timestamp": stamp, "epoch": self.epoch}
            self.visual_facts[fact["name"]] = value
            self.ledger.append_public("visual_assessment", {**fact, "source": "multimodal_model_estimate",
                                                            "sensor_timestamp": stamp})
        self._verify()

    def candidates(self) -> list[ToolSpec]:
        result = []
        view = self.decision_observation()
        holding = view.value("holding", self.clock(), self.config["freshness_s"])
        for spec in self.registry.values():
            if not spec.certified or not self.feasible(spec.resources):
                continue
            if spec.requires_holding and holding != "yes":
                continue
            if spec.tool == "pick_and_prealign" and self.observation.stage != "initial":
                continue
            if spec.tool == "finish" and not monitor(view, self.clock(), self.config["freshness_s"])["reported_success"]:
                continue
            if self.verification != "PASS" and spec.tool not in {"stop", "observe", "inspect", "ask_act"} and not spec.verification:
                continue
            result.append(spec)
        return result

    def execute(self, call: ToolCall) -> ToolResult:
        if call.call_id in self.calls:
            return self._result(call, "REJECTED", "duplicate_call_id")
        self.calls.add(call.call_id)
        spec = self.registry.get(call.candidate_id)
        reason = None
        if self.terminal:
            reason = "episode_ended"
        elif self.owner != "robot":
            reason = "ownership"
        elif call.control_epoch != self.epoch:
            reason = "stale_epoch"
        elif call.based_on_observation != self.observation.observation_id:
            reason = "stale_observation"
        elif self.clock() - self.observation.wall_timestamp > self.config["freshness_s"]:
            reason = "stale_observation"
        elif call.args:
            reason = "arguments_not_in_registered_profile"
        elif spec not in self.candidates():
            reason = "uncertified_precondition_or_budget"
        try:
            self.ledger.check_refs(call.evidence_refs)
        except ValueError:
            reason = "invalid_evidence_refs"
        if reason:
            return self._result(call, "REJECTED", reason)
        self.ledger.append_public("tool_call", {**asdict(call), "tool": spec.tool,
                                               "profile": spec.profile, "is_probe": bool(spec.resources.probes)})
        if spec.tool == "ask_act" and self.help_mode == "intervene_resume" and not self.feasible(spec.resources + self.resume_resources):
            return self._result(call, "REJECTED", "help_resume_budget")
        self.used += spec.resources
        start = self.clock()
        deadline = start + spec.resources.duration_s if spec.resources.duration_s else None
        actual, cost = {}, MeasuredCost()
        self.pending_help_cost = MeasuredCost()
        try:
            self._guard()
            if spec.tool in {"stop", "finish"}:
                if spec.tool == "finish" and not monitor(self.decision_observation(), self.clock(), self.config["freshness_s"])["reported_success"]:
                    return self._result(call, "REJECTED", "no_public_completion_evidence")
                self.adapter.safe_hold()
                self.terminal = spec.tool
                status, exit_reason = "COMPLETED", spec.tool
            elif spec.tool == "ask_act":
                cost = self._help(call, deadline)
                status, exit_reason = "COMPLETED", ("help_request_emitted" if self.help_mode == "request_only"
                                                     else "help_report_not_task_success")
            else:
                status, exit_reason, actual, cost = self.adapter.execute(
                    spec, lambda: self._guard(motion=spec.tool not in {"observe", "inspect"}, deadline=deadline))
                self.observe()
        except Interrupted as error:
            if spec.tool == "ask_act":
                cost = self.pending_help_cost
            self.adapter.safe_hold()
            self.epoch += 1
            status, exit_reason = "ABORTED", str(error)
            # A certified hold permits recovery after a bounded contact abort.
            recoverable = str(error) in {"force_limit", "tracking_limit", "tool_timeout"}
            self.terminal = None if recoverable and self.adapter.hold_certified else "stop"
            self.observe()
        except Exception:
            self.adapter.safe_hold()
            self.epoch += 1
            self.terminal = "stop"
            self.observe()
            self._result(call, "ABORTED", "adapter_error")
            raise
        cost = replace(cost, motion_s=self.clock() - start if spec.tool != "ask_act" else 0)
        self.cost += cost
        return self._result(call, status, exit_reason, actual, cost)

    def _help(self, call: ToolCall, deadline: float | None) -> MeasuredCost:
        scope = self.config["helper"]
        request = HelpRequest(call.call_id, scope["target"], scope["operation"], scope["allowed_scope"],
                              tuple(scope["desired_postconditions"]), call.evidence_refs,
                              "Assembly assistance requested from public evidence; cause remains unverified.")
        self.adapter.safe_hold()
        if self.help_mode == "request_only":
            self.epoch += 1
            self.help_request = request
            self.terminal = "help_requested"
            self.ledger.append_public("help_request", {"request": asdict(request),
                                      "delivery": "local_outbox", "control_epoch": self.epoch})
            self.observe()
            return MeasuredCost()
        self.adapter.quiesce(lambda: self._guard(deadline=deadline))
        self.epoch += 1
        self.owner = "helper"
        self.visual_facts.clear()
        self.verification = "UNKNOWN"
        self.ledger.append_public("handover", {"owner": self.owner, "control_epoch": self.epoch,
                                               "request": asdict(request)})
        helper_started = self.clock()
        report = None
        try:
            report = self.adapter.help(request, lambda: self._guard(deadline=deadline))
            self.ledger.append_public("help_report", asdict(report))
            if report.status != "COMPLETED":
                raise Interrupted("help_" + report.status.lower())
        finally:
            self.pending_help_cost = MeasuredCost(helper_s=self.clock()-helper_started,
                                                  helper_effort=report.effort if report else 0)
            self.adapter.safe_hold()
            self.epoch += 1
            self.owner = "robot"
            self.help_tick = self.adapter.tick
            self.adapter.idle_step()
            self.observe()
            self.ledger.append_public("handover", {"owner": self.owner, "control_epoch": self.epoch})
        return self.pending_help_cost

    def _result(self, call, status, reason, actual=None, cost=None) -> ToolResult:
        result = ToolResult(call.call_id, status, reason, actual or {},
                            self.observation.observation_id, self.epoch, cost or MeasuredCost())
        ref = self.ledger.append_public("tool_result", asdict(result))
        return replace(result, public_event_ids=(ref,))

    def reason(self, proposer, context):
        """Keep physics/guards alive during a blocking model call; no simulator work in its thread."""
        result, errors = [], []
        snapshot = self.observation
        def invoke():
            try:
                result.append(proposer(context, snapshot))
            except Exception as error:
                errors.append(error)
        started = self.clock()
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                self._guard(deadline=started + self.config["model_timeout_s"])
                self.adapter.idle_step()
                worker.join(timeout=0.001)
        except Interrupted:
            self.adapter.safe_hold()
            self.terminal = "stop"
            raise
        finally:
            self.cost += MeasuredCost(model_s=self.clock() - started)
        if errors:
            raise errors[0]
        return result[0]
