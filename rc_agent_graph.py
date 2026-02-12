"""Hybrid Theory-of-Mind Agent Orchestrator using LangGraph StateGraph.

Nodes:
  check_trigger  -> Conditional entry: evaluates GT changes or user input.
  context_builder -> Packages last 5 frames + current GT state JSON.
  query_memory   -> Queries Mem0 (or SQLite fallback) for long-term history.
  reason         -> Calls Cosmos Reason 2 with visual + text payload.
  plan_action    -> Parses VLM output into a robot target pose.
  execute        -> Passes target to RobotController (non-blocking).

Falls back to a sequential runner when LangGraph is not installed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from rc_long_term_memory import LongTermMemory, MemoryItem
from rc_short_term_memory import ShortTermMemory


AgentState = Dict[str, Any]


@dataclass
class AgentOutput:
    reply: str
    action: Dict[str, Any]
    belief_update: Dict[str, Any]
    ltm_snippets: List[str]
    raw: Dict[str, Any]
    gt_state: Dict[str, Any]  # Ground-truth state at inference time.
    trigger: Dict[str, Any]   # What triggered this inference.


class AgentGraph:
    """LangGraph-based Hybrid ToM agent with GT-aware triggering.

    The graph runs off-thread in the cognitive worker. USD writes are forbidden.
    """

    def __init__(
        self,
        *,
        short_memory: ShortTermMemory,
        long_memory: LongTermMemory,
        reason2_fn: Callable[[AgentState], Dict[str, Any]],
        logger=print,
    ) -> None:
        self._stm = short_memory
        self._ltm = long_memory
        self._reason2_fn = reason2_fn
        self._log = logger

        self._runner: Callable[[AgentState], AgentState]

        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(dict)
            graph.add_node("check_trigger", self._check_trigger)
            graph.add_node("context_builder", self._context_builder)
            graph.add_node("query_memory", self._query_memory)
            graph.add_node("reason", self._reason)
            graph.add_node("plan_action", self._plan_action)
            graph.add_node("execute", self._execute)

            graph.set_entry_point("check_trigger")
            graph.add_conditional_edges(
                "check_trigger",
                self._should_proceed,
                {"proceed": "context_builder", "skip": END},
            )
            graph.add_edge("context_builder", "query_memory")
            graph.add_edge("query_memory", "reason")
            graph.add_edge("reason", "plan_action")
            graph.add_edge("plan_action", "execute")
            graph.add_edge("execute", END)

            compiled = graph.compile()
            self._runner = compiled.invoke
            self._log("[INFO] AgentGraph: LangGraph enabled (Hybrid ToM).")
        except Exception as exc:
            self._runner = self._fallback_run
            self._log(f"[WARN] AgentGraph: LangGraph not available ({exc}); using fallback runner.")

    def run(self, state: AgentState) -> AgentOutput:
        out_state = self._runner(dict(state))
        raw = dict(out_state.get("model_output") or {})
        reply = str(raw.get("reply") or "")
        action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
        belief_update = raw.get("belief_update") if isinstance(raw.get("belief_update"), dict) else {}
        ltm_snips = out_state.get("ltm_snippets") or []
        if not isinstance(ltm_snips, list):
            ltm_snips = []
        gt_state = out_state.get("gt_state") or {}
        trigger = out_state.get("trigger") or {}
        return AgentOutput(
            reply=reply,
            action=action,
            belief_update=belief_update,
            ltm_snippets=[str(s) for s in ltm_snips],
            raw=raw,
            gt_state=gt_state if isinstance(gt_state, dict) else {},
            trigger=trigger if isinstance(trigger, dict) else {},
        )

    # --- LangGraph nodes ---

    def _check_trigger(self, state: AgentState) -> AgentState:
        """Evaluate whether this invocation should proceed (GT change or user input)."""
        state["ts"] = float(state.get("ts") or time.time())
        trigger = state.get("trigger") or {}
        if not isinstance(trigger, dict):
            trigger = {}
        state["trigger"] = trigger

        user_text = str(state.get("user_text") or "").strip()
        state["user_text"] = user_text

        trigger_type = str(trigger.get("type") or "periodic")
        state["should_proceed"] = True

        # Always proceed on user input.
        if user_text:
            state["should_proceed"] = True
            if not trigger.get("type"):
                trigger["type"] = "user_input"
            state["trigger"] = trigger
            return state

        # Always proceed on GT change.
        if trigger_type in ("gt_change", "initial"):
            state["should_proceed"] = True
            return state

        # Periodic triggers always proceed (backwards compat).
        state["should_proceed"] = True
        return state

    def _should_proceed(self, state: AgentState) -> str:
        """Conditional edge: route to 'proceed' or 'skip'."""
        return "proceed" if state.get("should_proceed", True) else "skip"

    def _context_builder(self, state: AgentState) -> AgentState:
        """Package the last 5 frames + GT state JSON for the VLM."""
        frames = state.get("frames") or []
        if not isinstance(frames, list):
            frames = list(frames)
        state["frames"] = frames

        # GT state is passed from the main thread via the trigger.
        gt_state = state.get("gt_state") or {}
        if not isinstance(gt_state, dict):
            gt_state = {}
        state["gt_state"] = gt_state

        # Decay STM and build summary.
        self._stm.decay(now=state["ts"])
        state["stm_summary"] = self._stm.compact_summary()
        state["stm_json"] = self._stm.to_json()

        # Build context string combining GT + STM for the VLM.
        gt_context = ""
        if gt_state.get("ground_truth_objects"):
            gt_context = f"\nGround-truth object poses:\n{json.dumps(gt_state['ground_truth_objects'], indent=1)}\n"
        state["gt_context"] = gt_context

        return state

    def _query_memory(self, state: AgentState) -> AgentState:
        """Query long-term memory for relevant history."""
        def _clip(s: str, n: int = 220) -> str:
            s = str(s or "").strip().replace("\n", " ")
            return s if len(s) <= n else (s[: n - 3] + "...")

        query = state.get("user_text") or ""
        if not query:
            query = str(state.get("stm_summary") or "")
        items: List[MemoryItem] = self._ltm.retrieve(str(query), top_k=3)
        snippets = []
        for it in items:
            meta = it.metadata or {}
            tag = meta.get("type") or meta.get("tag") or "mem"
            snippets.append(f"[{tag}] {_clip(it.text, 220)}")
        state["ltm_snippets"] = snippets[:3]
        return state

    def _reason(self, state: AgentState) -> AgentState:
        """Call Cosmos Reason 2 with the assembled visual + text payload."""
        state["model_output"] = self._reason2_fn(state)
        return state

    def _plan_action(self, state: AgentState) -> AgentState:
        """Parse the VLM output into a structured robot action."""
        out = state.get("model_output") or {}
        if not isinstance(out, dict):
            return state

        action = out.get("action")
        if isinstance(action, dict):
            state["pending_action"] = action
        else:
            state["pending_action"] = {"type": "noop", "args": {}}

        return state

    def _execute(self, state: AgentState) -> AgentState:
        """Pass-through: action execution happens on the main thread.

        HARD RULE: No USD writes or robot actuation off-thread.
        """
        return state

    def _reflect_store(self, state: AgentState) -> AgentState:
        """Store dialogue in LTM (called from worker, not a graph node)."""
        def _clip(s: str, n: int) -> str:
            s = str(s or "").strip().replace("\n", " ")
            return s if len(s) <= n else (s[: n - 3] + "...")

        out = state.get("model_output") or {}
        if not isinstance(out, dict):
            return state
        user_text = str(state.get("user_text") or "")
        reply = str(out.get("reply") or "")
        if user_text:
            self._ltm.add(
                f"USER: {_clip(user_text, 180)}\nASSISTANT: {_clip(reply, 260)}",
                {"type": "dialogue", "ts": state["ts"]},
            )

        # Also store GT-triggered events for episodic memory.
        trigger = state.get("trigger") or {}
        if trigger.get("type") == "gt_change" and trigger.get("changed_objects"):
            self._ltm.add(
                f"GT_CHANGE: objects={trigger['changed_objects']}, disp={trigger.get('max_displacement', 0):.3f}, reply={_clip(reply, 160)}",
                {"type": "gt_event", "ts": state["ts"]},
            )

        return state

    # --- Fallback runner ---

    def _fallback_run(self, state: AgentState) -> AgentState:
        state = self._check_trigger(state)
        if not state.get("should_proceed", True):
            return state
        state = self._context_builder(state)
        state = self._query_memory(state)
        state = self._reason(state)
        state = self._plan_action(state)
        state = self._execute(state)
        return state
