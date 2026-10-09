"""Empirical response tables and budget-aware depth-two branch enumeration."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contracts import Resources, ToolSpec, finite


STATE_FIELDS = {"stage", "holding", "progress_band", "force_band", "tracking_band",
                "consecutive_stalls", "probe_history"}


def state_key(state: dict[str, Any]) -> str:
    if set(state) != STATE_FIELDS:
        raise ValueError("estimator inputs must be the fixed public feature set")
    return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Outcome:
    probability: float
    next_state: dict[str, Any] | None = None
    terminal_success_probability: float | None = None


@dataclass(frozen=True)
class Estimate:
    success_probability: float
    support_count: int
    cost: dict[str, float]
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    information_gain_bits: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Estimate:
        probability = finite(value["success_probability"], "success probability", minimum=0)
        if probability > 1 or not isinstance(value["support_count"], int) or value["support_count"] < 1:
            raise ValueError("invalid empirical estimate")
        cost = value["cost"]
        if set(cost) != {"wall_s", "contact_exposure_Ns", "helper_effort", "model_money"}:
            raise ValueError("cost table must contain non-overlapping cost components")
        for name, amount in cost.items():
            finite(amount, name, minimum=0)
        outcomes = {name: Outcome(**item) for name, item in value.get("outcomes", {}).items()}
        if set(outcomes) - {"SUCCESS", "PROGRESS", "STILL_STALLED", "GRASP_LOST", "PROTECTION", "UNKNOWN"}:
            raise ValueError("unregistered observable outcome")
        for outcome in outcomes.values():
            finite(outcome.probability, "outcome probability", minimum=0)
            if outcome.next_state is not None:
                state_key(outcome.next_state)
            if outcome.terminal_success_probability is not None:
                terminal_p = finite(outcome.terminal_success_probability, "conditional terminal probability", minimum=0)
                if terminal_p > 1:
                    raise ValueError("invalid conditional terminal probability")
        if "SUCCESS" in outcomes and outcomes["SUCCESS"].terminal_success_probability is None:
            raise ValueError("public SUCCESS needs empirical terminal correctness, not automatic GT success")
        if outcomes and abs(sum(item.probability for item in outcomes.values()) - 1) > 1e-6:
            raise ValueError("outcome probabilities must sum to one")
        gain = value.get("information_gain_bits")
        if gain is not None:
            finite(gain, "information gain", minimum=0)
        return cls(probability, value["support_count"], dict(cost), outcomes, gain)


class Estimator:
    def __init__(self, table: dict[str, Any] | None = None, *, minimum_support: int = 2,
                 allow_contract_data: bool = False) -> None:
        self.minimum_support = minimum_support
        self.table: dict[str, dict[str, Estimate]] = {}
        self.provenance = (table or {}).get("provenance", {})
        if table and self.provenance.get("backend") not in {"isaac", "contract_only"}:
            raise ValueError("estimates need calibration source provenance")
        if self.provenance.get("backend") == "contract_only" and not allow_contract_data:
            raise ValueError("synthetic estimates cannot support physical experiments")
        if table is not None and table.get("version") != 1:
            raise ValueError("unsupported estimator version")
        for key, actions in (table or {}).get("states", {}).items():
            if state_key(json.loads(key)) != key:
                raise ValueError("invalid public state key")
            self.table[key] = {action: Estimate.from_dict(value) for action, value in actions.items()}

    @classmethod
    def load(cls, path: Path | None, **kwargs: Any) -> Estimator:
        return cls(json.loads(path.read_text()) if path else None, **kwargs)

    def predict(self, state: dict[str, Any], candidate_id: str) -> Estimate | None:
        estimate = self.table.get(state_key(state), {}).get(candidate_id)
        return estimate if estimate and estimate.support_count >= self.minimum_support else None


class Selector:
    def __init__(self, estimator: Estimator, *, reward: float = 1, failure_loss: float = 0,
                 cost_weights: dict[str, float] | None = None, seed: int = 0,
                 help_mode: str = "request_only") -> None:
        self.estimator = estimator
        if help_mode not in {"request_only", "intervene_resume"}:
            raise ValueError("unknown help mode")
        self.help_mode = help_mode
        self.reward = finite(reward, "success reward", minimum=0)
        self.loss = finite(failure_loss, "failure loss", minimum=0)
        self.weights = cost_weights or {"wall_s": 0.01, "contact_exposure_Ns": 0,
                                        "helper_effort": 0.1, "model_money": 1}
        for name, value in self.weights.items():
            if name not in {"wall_s", "contact_exposure_Ns", "helper_effort", "model_money"}:
                raise ValueError("time subcomponents must not be charged again")
            finite(value, name, minimum=0)
        self.rng = random.Random(seed)

    def cost(self, estimate: Estimate) -> float:
        return sum(self.weights.get(name, 0) * amount for name, amount in estimate.cost.items())

    def direct(self, estimate: Estimate) -> float:
        p = estimate.success_probability
        return p * self.reward - (1 - p) * self.loss - self.cost(estimate)

    def rank(self, state: dict[str, Any], candidates: list[ToolSpec],
             feasible: Callable[[Resources], bool], resume_duration_s: float | Resources) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            branch = candidate.resources
            if candidate.tool == "ask_act" and self.help_mode == "intervene_resume":
                branch += self._resume_resources(resume_duration_s)
            if not feasible(branch):
                continue
            if candidate.tool == "stop":
                rows.append({"candidate_id": candidate.candidate_id, "supported": True,
                             "Q_hat": -self.loss, "support_count": 0, "direct_finish": 0,
                             "continuation_value": 0, "predicted_cost": 0})
                continue
            if candidate.tool == "ask_act" and self.help_mode == "request_only":
                rows.append({"candidate_id": candidate.candidate_id, "supported": False,
                             "Q_hat": None, "support_count": 0,
                             "reason": "request_only_help_utility_not_defined",
                             "completion_scope": "help_request_only"})
                continue
            estimate = self.estimator.predict(state, candidate.candidate_id)
            if estimate is None:
                rows.append({"candidate_id": candidate.candidate_id, "supported": False,
                             "Q_hat": None, "support_count": 0, "reason": "unsupported_public_state_action"})
                continue
            continuation = 0.0
            changes = 0.0
            if candidate.tool in {"inspect", "probe_xy", "probe_angle", "probe_speed"}:
                if not estimate.outcomes:
                    continue
                before = self._best_continuation(state, candidates, feasible, Resources(), resume_duration_s)
                direct_finish = 0.0
                for label, outcome in estimate.outcomes.items():
                    if label == "SUCCESS":
                        p = outcome.terminal_success_probability
                        direct_finish += outcome.probability * (p * self.reward - (1-p) * self.loss)
                        continue
                    best = (-self.loss, "stop")
                    if label not in {"GRASP_LOST", "PROTECTION"} and outcome.next_state is not None:
                        best = self._best_continuation(outcome.next_state, candidates, feasible,
                                                       candidate.resources, resume_duration_s)
                    continuation += outcome.probability * best[0]
                    changes += outcome.probability * (best[1] != before[1])
                q = direct_finish + continuation - self.cost(estimate)
            else:
                direct_finish = estimate.success_probability * self.reward
                continuation = -(1 - estimate.success_probability) * self.loss
                q = self.direct(estimate)
            rows.append({"candidate_id": candidate.candidate_id, "supported": True,
                         "Q_hat": q, "support_count": estimate.support_count,
                         "direct_finish": direct_finish, "continuation_value": continuation,
                         "predicted_cost": self.cost(estimate), "decision_change_probability": changes,
                         "information_gain_bits": estimate.information_gain_bits,
                         "estimate": asdict(estimate),
                         "completion_scope": "help_verify_robot_resume" if candidate.tool == "ask_act" else "action"})
        return rows

    def _best_continuation(self, state: dict[str, Any], candidates: list[ToolSpec],
                           feasible: Callable[[Resources], bool], spent: Resources,
                           resume_duration_s: float | Resources) -> tuple[float, str]:
        best = (-self.loss, "stop")
        for candidate in candidates:
            if candidate.tool not in {"seat_once", "ask_act"}:
                continue
            if candidate.tool == "ask_act" and self.help_mode == "request_only":
                continue
            if candidate.requires_holding and state["holding"] != "yes":
                continue
            need = spent + candidate.resources
            if candidate.tool == "ask_act":
                need += self._resume_resources(resume_duration_s)
            estimate = self.estimator.predict(state, candidate.candidate_id)
            if estimate and feasible(need) and self.direct(estimate) > best[0]:
                best = (self.direct(estimate), candidate.candidate_id)
        return best

    @staticmethod
    def _resume_resources(value):
        return value if isinstance(value, Resources) else Resources(contacts=1, duration_s=value)

    def choose(self, rows: list[dict[str, Any]], candidates: list[ToolSpec], *,
               method: str, suggested: str, fallback: str) -> tuple[str, str]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        available = {row["candidate_id"] for row in rows}
        fallback = fallback if fallback in available else "stop"
        if method in {"generic", "repair_adapted"}:
            return (suggested, "shared_reasoner") if suggested in available else (fallback, "invalid_suggestion_fallback")
        supported = [row for row in rows if row["supported"] and row["candidate_id"] != "stop"]
        probes = [row for row in rows if by_id[row["candidate_id"]].resources.probes]
        if method == "random_safe" and probes:
            return self.rng.choice(probes)["candidate_id"], "random_from_shared_certified_candidates"
        if method == "information" and probes:
            informative = [row for row in probes if row.get("information_gain_bits") is not None]
            if informative:
                return max(informative, key=lambda row: row["information_gain_bits"])["candidate_id"], "initial_category_information_gain"
            return fallback, "unsupported_information_gain_fallback"
        if self.help_mode == "request_only" and any(
                row.get("reason") == "request_only_help_utility_not_defined" for row in rows):
            return fallback, "request_only_help_utility_not_defined_fallback"
        if supported:
            best = max(supported, key=lambda row: row["Q_hat"])
            return (best["candidate_id"], "depth2_value") if best["Q_hat"] > -self.loss else ("stop", "nonpositive_value")
        return fallback, "unsupported_estimates_fallback"
