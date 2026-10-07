"""Single-action REPAIR-style planners and optional HTTP boundary."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .contracts import Observation, PlannerAction, SkillResult, stable_hash


class PlannerUnavailable(RuntimeError):
    pass


def _action(obs: Observation, action: str, args: dict[str, Any] | None = None) -> PlannerAction:
    return PlannerAction(obs.observation_id, obs.control_epoch, action, args or {})


class RepairRulePlanner:
    """Deterministic REPAIR protocol exerciser, not a learned-model result."""

    def __init__(self, *, allow_help: bool = True, retries_before_help: int = 1) -> None:
        self.allow_help = allow_help
        self.retries_before_help = int(retries_before_help)
        self.place_attempts = 0
        self.help_used = False

    def next_action(self, observation: Observation, history: list[dict[str, Any]], budgets: Mapping[str, int]) -> tuple[PlannerAction, dict[str, Any]]:
        recent = observation.recent_skill or {}
        skill = str(recent.get("skill_id", ""))
        outcome = str(recent.get("outcome", ""))
        if observation.placed == "yes":
            return _action(observation, "finish"), {"provider": "rule_dev_only", "warning": "not an LLM result"}
        if skill == "place":
            if str(recent.get("execution_status", "")) == "system_error":
                return _action(observation, "stop"), {"provider": "rule_dev_only", "reason": "system error; help is not a renderer/control recovery"}
            if outcome == "succeeded":
                return _action(observation, "observe"), {"provider": "rule_dev_only"}
            if self.allow_help and not self.help_used and int(budgets.get("help_remaining", 0)) > 0:
                self.help_used = True
                return _action(observation, "help", {"request_type": "clear_target_area", "target": "hub", "request": "放置未完成，请检查并清除目标区域中的可移除阻挡；cover 尚未放置，仍需机器人完成。"}), {"provider": "rule_dev_only", "warning": "not an LLM result"}
            if self.place_attempts < self.retries_before_help + 1 and int(budgets.get("place_remaining", 0)) > 0:
                self.place_attempts += 1
                return _action(observation, "place", {"target": "hub"}), {"provider": "rule_dev_only"}
            return _action(observation, "stop"), {"provider": "rule_dev_only", "reason": "placement budget exhausted"}
        if skill == "help":
            if int(budgets.get("place_remaining", 0)) <= 0:
                return _action(observation, "stop"), {"provider": "rule_dev_only", "reason": "placement budget exhausted"}
            self.place_attempts += 1
            return _action(observation, "place", {"target": "hub"}), {"provider": "rule_dev_only"}
        if skill == "retract":
            if int(budgets.get("place_remaining", 0)) <= 0:
                return _action(observation, "stop"), {"provider": "rule_dev_only", "reason": "placement budget exhausted"}
            return _action(observation, "place", {"target": "hub"}), {"provider": "rule_dev_only"}
        if int(budgets.get("place_remaining", 0)) > 0:
            self.place_attempts += 1
        return _action(observation, "place", {"target": "hub"}), {"provider": "rule_dev_only"}


class FixedRetryPlanner(RepairRulePlanner):
    def __init__(self) -> None:
        super().__init__(allow_help=True, retries_before_help=1)


class AutoPlanner(RepairRulePlanner):
    def __init__(self) -> None:
        super().__init__(allow_help=False, retries_before_help=1)


class HttpRepairPlanner:
    """OpenAI-compatible single-action planner; no silent fallback."""

    def __init__(self, endpoint: str, model: str, *, api_key_env: str = "HRC_REPAIR_API_KEY", timeout_s: float = 30.0, allow_help: bool = True, prompt_path: str | None = None) -> None:
        self.endpoint, self.model, self.api_key_env = endpoint, model, api_key_env
        self.timeout_s = float(timeout_s)
        self.allow_help = allow_help
        self.prompt_path = prompt_path
        self.system_prompt = (
            Path(prompt_path).read_text(encoding="utf-8")
            if prompt_path and Path(prompt_path).is_file()
            else "Return exactly one JSON REPAIR M0 action. Do not output hidden state or coordinates."
        )

    def next_action(self, observation: Observation, history: list[dict[str, Any]], budgets: Mapping[str, int]) -> tuple[PlannerAction, dict[str, Any]]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise PlannerUnavailable(f"{self.api_key_env} is not configured")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps({"observation": observation.to_dict(), "history": history[-2:], "budgets": dict(budgets), "help_allowed": self.allow_help and int(budgets.get("help_remaining", 0)) > 0}, ensure_ascii=True)}],
            "response_format": {"type": "json_object"},
        }
        request_hash = stable_hash(payload)
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_bytes = response.read()
            raw = json.loads(raw_bytes.decode("utf-8"))
            content = str(raw["choices"][0]["message"]["content"]).strip()
            format_repair = False
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                decoded = _format_repair_action(content, observation)
                format_repair = True
            if isinstance(decoded, Mapping) and isinstance(decoded.get("action"), str) and decoded["action"].lower().startswith(("place ", "help ")):
                decoded = _format_repair_action(str(decoded["action"]), observation)
                format_repair = True
            elif isinstance(decoded, Mapping) and str(decoded.get("action", "")).lower() in {"place", "help"}:
                expected_args = {"target": "hub"} if str(decoded["action"]).lower() == "place" else {"request_type": "clear_target_area", "target": "hub", "request": "请清除目标区域中的可移除阻挡。"}
                if dict(decoded.get("args", {})) != expected_args and str(decoded["action"]).lower() == "place":
                    decoded = {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "place", "args": {"target": "hub"}}
                    format_repair = True
                elif str(decoded["action"]).lower() == "help" and not {"request_type", "target", "request"}.issubset(dict(decoded.get("args", {}))):
                    decoded = {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "help", "args": expected_args}
                    format_repair = True
            elif isinstance(decoded, Mapping) and str(decoded.get("action", "")).lower() == "pick" and dict(decoded.get("args", {})) != {"target": "cover"}:
                decoded = {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "pick", "args": {"target": "cover"}}
                format_repair = True
            action = PlannerAction.from_dict(decoded)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise PlannerUnavailable(f"planner request or validation failed: {type(exc).__name__}") from exc
        return action, {
            "provider": "http",
            "model": self.model,
            "prompt_hash": request_hash,
            "response_hash": stable_hash(raw),
            "latency_s": None,
            # This payload contains only the public observation/history.  It
            # is retained for leakage auditing; Authorization is never put
            # into the returned metadata.
            "public_input": payload,
            "format_repair": format_repair,
        }


def _format_repair_action(content: str, observation: Observation) -> dict[str, Any]:
    """Repair one output-format error without inventing a planner decision."""
    import re

    text = content.strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    if text.lower().startswith("json\n"):
        text = text[5:].lstrip()
    try:
        fenced = json.loads(text)
        if isinstance(fenced, Mapping):
            return dict(fenced)
    except json.JSONDecodeError:
        pass
    match = re.match(r"^(pick|place|help|finish|stop|observe|retract)\b(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("planner response is not a JSON action")
    action = match.group(1).lower()
    tail = match.group(2).strip()
    if action == "place":
        return {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "place", "args": {"target": "hub"}}
    if action == "pick":
        return {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "pick", "args": {"target": "cover"}}
    if action == "help":
        return {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": "help", "args": {"request_type": "clear_target_area", "target": "hub", "request": tail or "请清除目标区域中的可移除阻挡，cover 仍由机器人放置。"}}
    return {"observation_id": observation.observation_id, "control_epoch": observation.control_epoch, "action": action, "args": {}}


__all__ = ["RepairRulePlanner", "FixedRetryPlanner", "AutoPlanner", "HttpRepairPlanner", "PlannerUnavailable"]
