"""Minimal stdin human-control broker for the optional fault-HIL run."""

from __future__ import annotations

import json
from typing import Any

from .contracts import EpisodeState
from .state_machine import M1StateMachine


class HumanBroker:
    """Interactive command broker; it never writes USD poses directly."""

    def __init__(self, adapter: Any, logger: Any, *, input_stream: Any = None) -> None:
        import sys

        self.adapter = adapter
        self.logger = logger
        self.input_stream = input_stream or sys.stdin

    def run(self, state_machine: M1StateMachine) -> bool:
        if state_machine.state == EpisodeState.SAFE_HOLD:
            state_machine.request_help("operator requested after guarded skill failure")
        print("HRC HIL: type JSON TAKE_CONTROL, JOG {delta_m:[x,y,z]}, RETURN_CONTROL_AND_DONE or ABORT", flush=True)
        for line in self.input_stream:
            try:
                command = json.loads(line)
                event_type = str(command.get("type", "")).upper()
                if event_type == "TAKE_CONTROL":
                    state_machine.take_control()
                elif event_type == "JOG":
                    delta = command.get("delta_m")
                    state_machine.human_jog({"type": "cartesian_delta", "delta_m": delta})
                    self.adapter.human_jog(delta)
                elif event_type == "RETURN_CONTROL_AND_DONE":
                    state_machine.return_control()
                    self.logger.event("human_return_control", {"type": event_type})
                    return True
                elif event_type == "ABORT":
                    state_machine.abort("operator_abort")
                    return False
                else:
                    raise ValueError("unsupported human command")
                self.logger.event("human_command", {"type": event_type, "delta_m": command.get("delta_m")})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.logger.event("human_command_rejected", {"error_type": type(exc).__name__})
        return False


__all__ = ["HumanBroker"]
