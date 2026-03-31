"""Dual-brain agent orchestrator (physics observer + cognitive commander).

Pipeline:
  check_trigger -> context_builder -> perceive_physics -> cognitive_reasoning
  -> merge_belief -> queue_action -> execute

Design constraints:
- Pure-Python importable (no omni.* or isaaclab.* imports at module import time)
- No USD writes inside this graph
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypedDict


class AgentStateTyped(TypedDict, total=False):
    visual_buffer: List[Any]
    belief_state: Dict[str, Any]
    static_context: Dict[str, Any]
    user_history: List[Dict[str, Any]]

    frames: List[Any]
    user_text: str
    trigger: Dict[str, Any]
    gt_state: Dict[str, Any]
    ts: float
    should_proceed: bool

    stm_summary: str
    stm_json: str
    gt_context: str
    ltm_snippets: List[str]
    tracked_object_ids: List[str]

    latest_observation: str
    commander_output: Dict[str, Any]
    model_output: Dict[str, Any]
    pending_action: Dict[str, Any]


AgentState = Dict[str, Any]


@dataclass
class AgentOutput:
    reply: str
    action: Dict[str, Any]
    belief_update: Dict[str, Any]
    stm_observation: str
    ltm_snippets: List[str]
    raw: Dict[str, Any]
    gt_state: Dict[str, Any]
    trigger: Dict[str, Any]


class AgentGraph:
    """StateGraph runner for the dual-model architecture."""

    def __init__(
        self,
        *,
        short_memory,
        long_memory,
        physics_observer_fn: Callable[..., str] | None = None,
        commander_fn: Callable[..., Dict[str, Any]] | None = None,
        logger=print,
    ) -> None:
        self._stm = short_memory
        self._ltm = long_memory
        self._physics_observer_fn = physics_observer_fn or self._default_physics_observer
        self._commander_fn = commander_fn or self._default_commander
        self._log = logger

        self._runner: Callable[[AgentState], AgentState]
        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(dict)
            graph.add_node("check_trigger", self._check_trigger)
            graph.add_node("context_builder", self._context_builder)
            graph.add_node("perceive_physics", self._perceive_physics)
            graph.add_node("cognitive_reasoning", self._cognitive_reasoning)
            graph.add_node("merge_belief", self._merge_belief)
            graph.add_node("queue_action", self._queue_action)
            graph.add_node("execute", self._execute)

            graph.set_entry_point("check_trigger")
            graph.add_conditional_edges(
                "check_trigger",
                self._should_proceed,
                {"proceed": "context_builder", "skip": END},
            )
            graph.add_edge("context_builder", "perceive_physics")
            graph.add_edge("perceive_physics", "cognitive_reasoning")
            graph.add_edge("cognitive_reasoning", "merge_belief")
            graph.add_edge("merge_belief", "queue_action")
            graph.add_edge("queue_action", "execute")
            graph.add_edge("execute", END)

            compiled = graph.compile()
            self._runner = compiled.invoke
            self._log("[INFO] AgentGraph: LangGraph enabled (dual-brain pipeline).")
        except Exception as exc:
            self._runner = self._fallback_run
            self._log(f"[WARN] AgentGraph: LangGraph unavailable ({exc}); using fallback runner.")

    def run(self, state: AgentState) -> AgentOutput:
        out_state = self._runner(dict(state))
        raw = dict(out_state.get("model_output") or {})
        reply = str(raw.get("reply") or "")
        action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
        belief_update = raw.get("belief_update") if isinstance(raw.get("belief_update"), dict) else {}
        stm_observation = str(raw.get("stm_observation") or "")
        ltm_snips = out_state.get("ltm_snippets") or []
        if not isinstance(ltm_snips, list):
            ltm_snips = []
        gt_state = out_state.get("gt_state") or {}
        trigger = out_state.get("trigger") or {}
        return AgentOutput(
            reply=reply,
            action=action,
            belief_update=belief_update,
            stm_observation=stm_observation,
            ltm_snippets=[str(s) for s in ltm_snips],
            raw=raw,
            gt_state=gt_state if isinstance(gt_state, dict) else {},
            trigger=trigger if isinstance(trigger, dict) else {},
        )

    def _default_physics_observer(self, **kwargs) -> str:
        from runtime.vlm_cosmos import observe_physics

        return observe_physics(
            frames=kwargs.get("frames") or [],
            object_ids=kwargs.get("object_ids") or [],
            trigger=kwargs.get("trigger") if isinstance(kwargs.get("trigger"), dict) else {},
            user_text=str(kwargs.get("user_text") or ""),
        )

    def _default_commander(self, **kwargs) -> Dict[str, Any]:
        from runtime.llm_commander import commander_reason

        return commander_reason(
            belief_state=kwargs.get("belief_state") if isinstance(kwargs.get("belief_state"), dict) else {},
            user_chat=str(kwargs.get("user_chat") or ""),
            latest_observation=str(kwargs.get("latest_observation") or ""),
            long_memory_snippets=[str(s) for s in (kwargs.get("long_memory_snippets") or [])],
            frames=kwargs.get("frames") or [],
            object_ids=kwargs.get("object_ids") or [],
            trigger=kwargs.get("trigger") if isinstance(kwargs.get("trigger"), dict) else {},
        )

    # --- nodes ---

    def _check_trigger(self, state: AgentState) -> AgentState:
        state["ts"] = float(state.get("ts") or time.time())
        trigger = state.get("trigger") or {}
        if not isinstance(trigger, dict):
            trigger = {}
        state["trigger"] = trigger

        user_text = str(state.get("user_text") or "").strip()
        state["user_text"] = user_text

        trigger_type = str(trigger.get("type") or "periodic")
        state["should_proceed"] = bool(user_text) or trigger_type in (
            "gt_change",
            "initial",
            "grounding",
            "periodic",
            "user",
            "user_input",
        )
        return state

    def _should_proceed(self, state: AgentState) -> str:
        return "proceed" if state.get("should_proceed", True) else "skip"

    def _context_builder(self, state: AgentState) -> AgentState:
        frames = state.get("frames") or []
        if not isinstance(frames, list):
            frames = list(frames)
        state["frames"] = frames

        gt_state = state.get("gt_state") or {}
        if not isinstance(gt_state, dict):
            gt_state = {}
        state["gt_state"] = gt_state

        belief_state = state.get("belief_state")
        if isinstance(belief_state, dict) and "static_context" not in state:
            state["static_context"] = belief_state.get("static_context", {})

        self._stm.decay(now=state["ts"])
        state["stm_summary"] = self._stm.compact_summary()
        state["stm_json"] = self._stm.to_json()

        gt_context = ""
        gt_objs = gt_state.get("ground_truth_objects")
        if isinstance(gt_objs, dict) and gt_objs:
            names = [str(n) for n in gt_objs.keys()][:32]
            gt_context = ", ".join(names)
        state["gt_context"] = gt_context
        return state

    def _perceive_physics(self, state: AgentState) -> AgentState:
        belief = state.get("belief_state") if isinstance(state.get("belief_state"), dict) else {}
        belief_objects = belief.get("objects") if isinstance(belief.get("objects"), dict) else {}

        object_ids: List[str] = [str(k) for k in belief_objects.keys()]
        trigger = state.get("trigger") if isinstance(state.get("trigger"), dict) else {}
        for key in ("all_interactables", "init_interactables", "changed_objects"):
            value = trigger.get(key)
            if isinstance(value, list):
                for item in value:
                    name = str(item or "").split("/")[-1].strip()
                    if name and name not in object_ids:
                        object_ids.append(name)

        # A2A dual-brain: pass frames + object IDs to Commander; Commander decides
        # whether/when to call Cosmos observer.
        state["tracked_object_ids"] = object_ids
        state["latest_observation"] = str(state.get("latest_observation") or "").strip()
        return state

    def _query_memory_inline(self, state: AgentState) -> List[str]:
        def _clip(s: str, n: int = 220) -> str:
            s = str(s or "").strip().replace("\n", " ")
            return s if len(s) <= n else (s[: n - 3] + "...")

        query = state.get("user_text") or state.get("latest_observation") or state.get("stm_summary") or ""
        items = self._ltm.retrieve(str(query), top_k=3)
        snippets: List[str] = []
        for item in items:
            meta = item.metadata or {}
            tag = meta.get("type") or meta.get("tag") or "mem"
            snippets.append(f"[{tag}] {_clip(item.text, 220)}")
        return snippets[:3]

    def _cognitive_reasoning(self, state: AgentState) -> AgentState:
        snippets = self._query_memory_inline(state)
        state["ltm_snippets"] = snippets

        commander_out = self._commander_fn(
            belief_state=state.get("belief_state") if isinstance(state.get("belief_state"), dict) else {},
            user_chat=str(state.get("user_text") or ""),
            latest_observation=str(state.get("latest_observation") or ""),
            long_memory_snippets=snippets,
            frames=state.get("frames") or [],
            object_ids=state.get("tracked_object_ids") or [],
            trigger=state.get("trigger") if isinstance(state.get("trigger"), dict) else {},
        )
        if not isinstance(commander_out, dict):
            commander_out = {}
        observer_text = str(commander_out.get("_observer_text") or "").strip()
        if observer_text:
            state["latest_observation"] = observer_text
        state["commander_output"] = commander_out
        return state

    def _merge_belief(self, state: AgentState) -> AgentState:
        commander = state.get("commander_output") if isinstance(state.get("commander_output"), dict) else {}
        belief_update = commander.get("belief_state_update")
        if not isinstance(belief_update, dict):
            belief_update = {}

        action = commander.get("action") if isinstance(commander.get("action"), dict) else {"type": "noop", "args": {}}
        reply = str(commander.get("reply_to_human") or "").strip()

        state["model_output"] = {
            "reply": reply,
            "action": action,
            "belief_update": belief_update,
            "stm_observation": str(state.get("latest_observation") or ""),
            "meta": {
                "pipeline": "dual_brain",
                "observer": "cosmos_reason2_via_commander" if commander.get("_used_cosmos") else "commander_direct",
                "num_images": len(state.get("frames") or []),
                "used_cosmos": bool(commander.get("_used_cosmos")),
            },
        }
        return state

    def _queue_action(self, state: AgentState) -> AgentState:
        model_output = state.get("model_output") if isinstance(state.get("model_output"), dict) else {}
        action = model_output.get("action") if isinstance(model_output.get("action"), dict) else None
        if not action:
            action = {"type": "noop", "args": {}}
        state["pending_action"] = action
        return state

    def _execute(self, state: AgentState) -> AgentState:
        return state

    def _reflect_store(self, state: AgentState) -> AgentState:
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

        trigger = state.get("trigger") or {}
        if trigger.get("type") == "gt_change" and trigger.get("changed_objects"):
            self._ltm.add(
                f"GT_CHANGE: objects={trigger['changed_objects']}, obs={_clip(state.get('latest_observation', ''), 180)}",
                {"type": "gt_event", "ts": state["ts"]},
            )
        return state

    def _fallback_run(self, state: AgentState) -> AgentState:
        state = self._check_trigger(state)
        if not state.get("should_proceed", True):
            return state
        state = self._context_builder(state)
        state = self._perceive_physics(state)
        state = self._cognitive_reasoning(state)
        state = self._merge_belief(state)
        state = self._queue_action(state)
        state = self._execute(state)
        return state
