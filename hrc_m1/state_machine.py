"""Finite-state and control-ownership guards for M1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    ALLOWED_ACTIONS,
    ControlOwner,
    Decision,
    EpisodeState,
    Observation,
    SkillResult,
)


@dataclass
class Transition:
    state: EpisodeState
    owner: ControlOwner
    reason: str
    observation_id: str | None = None


@dataclass
class M1StateMachine:
    episode_id: str
    replan_on_visual_failure: bool = False
    state: EpisodeState = EpisodeState.RESET
    owner: ControlOwner = ControlOwner.SAFE_STOP
    current_observation_id: str | None = None
    pending_decision: Decision | None = None
    turns: int = 0
    max_turns: int = 64
    transitions: list[Transition] = field(default_factory=list)

    def _set(self, state: EpisodeState, reason: str, *, owner: ControlOwner | None = None) -> None:
        self.state = state
        if owner is not None:
            self.owner = owner
        self.transitions.append(Transition(state, self.owner, reason, self.current_observation_id))

    def reset(self) -> None:
        self.pending_decision = None
        self.turns = 0
        self.current_observation_id = None
        self.owner = ControlOwner.AUTO
        self._set(EpisodeState.OBSERVE, "reset_complete")

    def accept_observation(self, observation: Observation) -> None:
        if observation.episode_id != self.episode_id:
            raise ValueError("observation belongs to another episode")
        if self.owner == ControlOwner.HUMAN and self.state not in {
            EpisodeState.VALIDATE_HANDOFF,
            EpisodeState.REOBSERVE,
            EpisodeState.UPDATE_MEMORY,
        }:
            raise ValueError("cannot replace observation while human owns control")
        self.current_observation_id = observation.observation_id
        self.pending_decision = None
        if self.state in {EpisodeState.RESET, EpisodeState.OBSERVE}:
            self.owner = ControlOwner.AUTO
            self._set(EpisodeState.PLAN, "observation_accepted")
        elif self.state == EpisodeState.REOBSERVE:
            # A fresh post-handoff/post-recovery observation must pass through
            # an explicit memory update before a planner decision can be
            # accepted.  This prevents a stale trajectory from being resumed
            # merely because a new frame arrived.
            self.owner = ControlOwner.AUTO
            self._set(EpisodeState.UPDATE_MEMORY, "observation_accepted_for_memory_update")

    def accept_decision(self, decision: Decision) -> None:
        if decision.episode_id != self.episode_id:
            raise ValueError("decision belongs to another episode")
        if decision.observation_id != self.current_observation_id:
            raise ValueError("stale decision observation_id")
        if self.owner != ControlOwner.AUTO:
            raise ValueError("only AUTO may submit planner decisions")
        if self.state not in {EpisodeState.PLAN, EpisodeState.REPLAN}:
            raise ValueError(f"decision not accepted in state {self.state.value}")
        if self.turns >= self.max_turns:
            raise ValueError("agent turn budget exhausted")
        self.turns += 1
        self.pending_decision = decision
        if decision.action == "ABORT":
            self._set(EpisodeState.ABORT, "planner_abort", owner=ControlOwner.SAFE_STOP)
        elif decision.action == "ASK_HUMAN":
            self._set(EpisodeState.SAFE_HOLD, "planner_requested_help", owner=ControlOwner.SAFE_STOP)
        elif decision.action == "OBSERVE":
            self._set(EpisodeState.OBSERVE, "planner_requested_observe")
        elif decision.action == "REPLAN":
            self._set(EpisodeState.REPLAN, "planner_requested_replan")
        elif decision.action == "VERIFY":
            self._set(EpisodeState.VERIFY, "planner_requested_verify")
        elif decision.action == "RELEASE_RETRACT":
            self._set(EpisodeState.RELEASE_RETRACT, "planner_requested_release")
        else:
            self._set(EpisodeState.EXECUTE, f"planner_selected_{decision.action.lower()}")

    def skill_started(self, skill_id: str) -> None:
        if self.owner != ControlOwner.AUTO:
            raise ValueError("skill cannot start without AUTO ownership")
        if self.state not in {EpisodeState.EXECUTE, EpisodeState.RELEASE_RETRACT}:
            raise ValueError(f"skill started in state {self.state.value}")
        if self.pending_decision is None or self.pending_decision.action not in ALLOWED_ACTIONS:
            raise ValueError("no valid pending decision")
        if not skill_id:
            raise ValueError("skill_id is required")

    def skill_finished(self, result: SkillResult) -> None:
        if result.episode_id != self.episode_id:
            raise ValueError("skill result belongs to another episode")
        if result.observation_id != self.current_observation_id:
            raise ValueError("skill result belongs to a stale observation")
        if self.state not in {EpisodeState.EXECUTE, EpisodeState.RELEASE_RETRACT}:
            raise ValueError(f"skill finished in state {self.state.value}")
        self._set(EpisodeState.VERIFY, f"skill_finished:{result.motion_status}")

    def verify(self, verdict: str) -> None:
        verdict = str(verdict).upper()
        if verdict not in {"SUCCESS", "FAILED", "UNKNOWN"}:
            raise ValueError("verdict must be SUCCESS, FAILED or UNKNOWN")
        if self.state != EpisodeState.VERIFY:
            raise ValueError(f"verify in state {self.state.value}")
        if verdict == "SUCCESS":
            if self.pending_decision and self.pending_decision.action == "RELEASE_RETRACT":
                self._set(EpisodeState.DONE, "release_verified", owner=ControlOwner.SAFE_STOP)
            else:
                self._set(EpisodeState.REOBSERVE, "seat_candidate_verified")
        elif verdict == "FAILED" and self.replan_on_visual_failure:
            self._set(EpisodeState.REPLAN, "vader_visual_failure_replan", owner=ControlOwner.AUTO)
        elif verdict == "FAILED":
            self._set(EpisodeState.SAFE_HOLD, "verification_failed", owner=ControlOwner.SAFE_STOP)
        else:
            self._set(EpisodeState.SAFE_HOLD, "verification_unknown", owner=ControlOwner.SAFE_STOP)

    def request_help(self, message: str) -> None:
        if self.state != EpisodeState.SAFE_HOLD or self.owner != ControlOwner.SAFE_STOP:
            raise ValueError("help may only be requested from SAFE_HOLD")
        if not message.strip():
            raise ValueError("help message is required")
        self._set(EpisodeState.ASK_HUMAN, "help_request_emitted")

    def take_control(self) -> None:
        if self.state != EpisodeState.ASK_HUMAN:
            raise ValueError("TAKE_CONTROL is only valid after ASK_HUMAN")
        self._set(EpisodeState.VALIDATE_HANDOFF, "human_take_control", owner=ControlOwner.HUMAN)

    def human_jog(self, command: dict[str, Any]) -> None:
        if self.owner != ControlOwner.HUMAN or self.state != EpisodeState.VALIDATE_HANDOFF:
            raise ValueError("human jog requires HUMAN ownership")
        if not isinstance(command, dict) or command.get("type") != "cartesian_delta":
            raise ValueError("only bounded cartesian_delta teleop commands are allowed")
        delta = command.get("delta_m")
        if not isinstance(delta, (list, tuple)) or len(delta) != 3:
            raise ValueError("cartesian_delta requires three delta_m values")
        if any(abs(float(v)) > 0.01 for v in delta):
            raise ValueError("teleop delta exceeds 10 mm command limit")

    def return_control(self) -> None:
        if self.owner != ControlOwner.HUMAN or self.state != EpisodeState.VALIDATE_HANDOFF:
            raise ValueError("RETURN_CONTROL requires active human ownership")
        # The pre-handoff decision/trajectory is no longer valid.  A new
        # observation and explicit memory update must precede replanning.
        self.pending_decision = None
        self.owner = ControlOwner.AUTO
        self._set(EpisodeState.REOBSERVE, "human_returned_control")

    def update_memory(self) -> None:
        if self.state != EpisodeState.UPDATE_MEMORY or self.owner != ControlOwner.AUTO:
            raise ValueError("memory update requires a fresh AUTO observation")
        self._set(EpisodeState.REPLAN, "memory_updated_from_fresh_observation")

    def safe_stop(self, reason: str) -> None:
        self.pending_decision = None
        self._set(EpisodeState.SAFE_HOLD, reason, owner=ControlOwner.SAFE_STOP)

    def abort(self, reason: str) -> None:
        self.pending_decision = None
        self._set(EpisodeState.ABORT, reason, owner=ControlOwner.SAFE_STOP)
