"""assembly/sequence_runner.py

Loads asset_registry.yaml and gearbox_sequence.yaml, tracks TaskState,
and executes assembly steps via MagicAssemblyManager.

Public interface (called by ui.py slash commands):
    runner.load()                          -> bool
    runner.run_sequence(delay_s)           -> int   (steps succeeded)
    runner.step_next()                     -> StepResult | None
    runner.step_prev()                     -> bool
    runner.run_alias(alias_name)           -> list[StepResult]
    runner.jump_to_step(step_id)           -> StepResult | None
    runner.validate_state()                -> dict
    runner.status()                        -> dict
    runner.reset()                         -> None  (clears task state, keeps YAML loaded)
"""

import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # handled in load()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_id: str
    human_label: str
    success: bool
    failure_reason: Optional[str]
    assemblies_before: List[Tuple[str, str]]
    assemblies_after: List[Tuple[str, str]]
    executor_mode: str
    # Member step ids covered when this result represents a simultaneous group
    # executed as one logical step. Empty for ordinary single-step results.
    members: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskState:
    sequence_id: str
    total_steps: int
    current_index: int                   # index into _ordered_ids; -1 = not started
    completed_steps: List[str]           # step ids that succeeded
    failed_steps: List[str]             # step ids that were attempted and failed
    skipped_steps: List[str]            # verify=true steps bypassed by user
    blocked: bool
    block_reason: str
    history: List[StepResult]           # ordered execution log
    check_results: Dict[str, int] = field(default_factory=dict)  # check_pose step_id -> 1/0
    # When a check_pose step defers to an async prompt (e.g. the UI), this holds
    # {"step_id": ..., "child": ...} for the step awaiting an answer. The run is
    # paused until answer_pose() supplies a result. None when nothing is pending.
    pending_pose: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [r.to_dict() for r in self.history]
        return d


# ---------------------------------------------------------------------------
# SequenceRunner
# ---------------------------------------------------------------------------

_CASING_PARENTS = {"Casing_Top", "Casing_Base"}
_REGISTRY_PATH = Path(__file__).parent / "asset_registry.yaml"
_SEQUENCE_PATH = Path(__file__).parent / "gearbox_sequence.yaml"


class _PoseQueryPending(BaseException):
    """Control-flow signal: a check_pose step deferred to an async prompt.

    Inherits from BaseException (not Exception) so the broad ``except Exception``
    in ``_execute_step`` does not swallow it and mis-report the step as failed.
    Raised by ``_execute_step`` and caught by the run methods, which stop the
    run cleanly; the answer arrives later via ``answer_pose()``.
    """

    def __init__(self, step_id: str, child: Optional[str]):
        super().__init__(step_id)
        self.step_id = step_id
        self.child = child


class SequenceRunner:
    """Stateful runner for a single assembly sequence."""

    def __init__(
        self,
        magic_assembly,
        log_fn: Callable[[str, str], None],
        registry_path: Path = _REGISTRY_PATH,
        sequence_path: Path = _SEQUENCE_PATH,
    ):
        self._ma = magic_assembly
        self._log = log_fn          # log_fn(level, message) — e.g. log_line("INFO ", msg)
        self._registry_path = registry_path
        self._sequence_path = sequence_path
        self._lock = threading.Lock()

        # Loaded data
        self._registry: Dict[str, dict] = {}      # name -> part entry
        self._sequence_meta: dict = {}
        self._steps: List[dict] = []              # ordered step list (from YAML)
        self._step_map: Dict[str, dict] = {}      # step_id -> step dict
        self._ordered_ids: List[str] = []         # step_ids in YAML file order
        self._aliases: Dict[str, List[str]] = {}  # alias_name -> [step_id, ...]

        # Parallel work streams and interchangeable groups
        self._streams: Dict[str, dict] = {}              # stream_id -> stream spec
        self._phase_stream: Dict[str, str] = {}          # phase_id -> stream_id
        self._groups: Dict[str, dict] = {}               # group_id -> group spec
        self._step_group: Dict[str, str] = {}            # step_id -> group_id

        # Simultaneous groups: steps sharing a non-null `simultaneous` tag run
        # as one logical step (e.g. the 3 shaft stages; each M10 bolt+nut pair).
        self._simultaneous_groups: Dict[str, List[str]] = {}  # tag -> [step_id] (file order)
        self._step_simultaneous: Dict[str, str] = {}          # step_id -> tag

        self._task_state: Optional[TaskState] = None
        self._loaded = False

        # Pose resolver: called for a check_pose step with no recorded answer.
        # Returns True/False to answer inline, or None to defer to an async
        # prompt (pauses the run; answer supplied later via answer_pose()).
        # Default = blocking terminal prompt (assembly.pose_check.check_pose),
        # which never returns None, so non-UI runs never pause. The UI installs
        # its own resolver via set_pose_resolver().
        self._pose_resolver: Callable[[str, Optional[str]], Optional[bool]] = (
            self._default_pose_resolver
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load and validate YAML files. Must be called before any execution.
        Returns True on success, False on error."""
        if yaml is None:
            self._log("WARN ", "[SEQ] pyyaml not installed — cannot load sequence files")
            return False

        try:
            registry_data = yaml.safe_load(self._registry_path.read_text())
            sequence_data = yaml.safe_load(self._sequence_path.read_text())
        except FileNotFoundError as e:
            self._log("WARN ", f"[SEQ] load failed: {e}")
            return False
        except yaml.YAMLError as e:
            self._log("WARN ", f"[SEQ] YAML parse error: {e}")
            return False

        # Index registry by name
        parts = registry_data.get("parts", [])
        self._registry = {p["name"]: p for p in parts}

        # Index sequence steps
        steps = sequence_data.get("steps", [])
        if not steps:
            self._log("WARN ", "[SEQ] load: no steps found in sequence file")
            return False

        self._steps = steps
        self._step_map = {s["id"]: s for s in steps}
        self._ordered_ids = [s["id"] for s in steps]
        self._sequence_meta = sequence_data.get("meta", {})

        # Index shortcut aliases
        raw_aliases = sequence_data.get("shortcut_aliases", {})
        self._aliases = {
            name: data["steps"]
            for name, data in raw_aliases.items()
            if isinstance(data, dict) and "steps" in data
        }

        # Index parallel streams (meta.streams) and build phase→stream map
        stream_specs = self._sequence_meta.get("streams", []) or []
        self._streams = {s["id"]: s for s in stream_specs if "id" in s}
        self._phase_stream = {}
        for sid, spec in self._streams.items():
            for phase_id in spec.get("phases", []) or []:
                self._phase_stream[phase_id] = sid

        # Index interchangeable groups (top-level `groups:` section) and build
        # reverse map step_id→group_id (a step belongs to at most one group).
        group_specs = sequence_data.get("groups", []) or []
        self._groups = {g["id"]: g for g in group_specs if "id" in g}
        self._step_group = {}
        for gid, spec in self._groups.items():
            for step_id in spec.get("members", []) or []:
                self._step_group[step_id] = gid

        # Index simultaneous groups from each step's `simultaneous` tag (file
        # order preserved). Members of a tag execute together as one logical
        # step (see _execute_simultaneous_group).
        self._simultaneous_groups = {}
        self._step_simultaneous = {}
        for step in self._steps:
            tag = step.get("simultaneous")
            if not tag:  # None / null / empty -> not part of any group
                continue
            tag = str(tag)
            self._simultaneous_groups.setdefault(tag, []).append(step["id"])
            self._step_simultaneous[step["id"]] = tag

        # Validate
        errors = self._validate_schema()
        if errors:
            for e in errors:
                self._log("WARN ", f"[SEQ] schema warning: {e}")
            # Warn but don't abort — some parts (bearings) have verify=true

        self._loaded = True
        total = len(self._steps)
        seq_id = self._sequence_meta.get("assembly", "unknown")
        self._log(
            "INFO ",
            f"[SEQ] loaded '{seq_id}': {total} steps, {len(self._streams)} streams, "
            f"{len(self._groups)} groups, {len(self._aliases)} aliases",
        )
        return True

    def _validate_schema(self) -> List[str]:
        """Cross-check step part names against registry. Returns list of warnings."""
        errors = []
        for step in self._steps:
            for field_name in ("child", "parent"):
                name = step.get(field_name)
                if name and name not in self._registry:
                    if step.get("verify"):
                        errors.append(
                            f"step {step['id']}: {field_name}={name!r} not in registry "
                            f"(verify=true — confirm prim exists at runtime)"
                        )
                    else:
                        errors.append(
                            f"step {step['id']}: {field_name}={name!r} not in registry"
                        )
        return errors

    # ------------------------------------------------------------------
    # Task state management
    # ------------------------------------------------------------------

    def _ensure_task_state(self) -> TaskState:
        if self._task_state is None:
            seq_id = self._sequence_meta.get("assembly", "gearbox")
            self._task_state = TaskState(
                sequence_id=seq_id,
                total_steps=len(self._steps),
                current_index=-1,
                completed_steps=[],
                failed_steps=[],
                skipped_steps=[],
                blocked=False,
                block_reason="",
                history=[],
            )
        return self._task_state

    def reset(self) -> None:
        """Clear task state (history, progress). Keeps YAML data loaded."""
        with self._lock:
            self._task_state = None
        self._log("INFO ", "[SEQ] task state reset")

    # ------------------------------------------------------------------
    # Pose resolution (check_pose)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_pose_resolver(child: Optional[str], step_id: str) -> Optional[bool]:
        """Default resolver: delegate to assembly.pose_check.check_pose.

        Returns a bool (never None), so a run using this resolver never pauses —
        in a terminal it prompts via input(); headless it falls back to True.
        """
        from assembly.pose_check import check_pose
        return check_pose(child)

    def set_pose_resolver(
        self, resolver: Callable[[str, Optional[str]], Optional[bool]]
    ) -> None:
        """Install a pose resolver. ``resolver(child, step_id)`` returns True/False
        to answer a check_pose inline, or None to defer to an async prompt (the
        run pauses and answer_pose() supplies the result later)."""
        self._pose_resolver = resolver

    @property
    def pending_pose(self) -> Optional[Dict[str, str]]:
        """The check_pose step awaiting an async answer, or None.

        ``{"step_id": ..., "child": ...}`` while a run is paused on a deferred
        pose query; None otherwise."""
        ts = self._task_state
        return ts.pending_pose if ts is not None else None

    def answer_pose(self, value: bool) -> Optional[str]:
        """Supply the answer for a pending deferred pose query and clear the
        pause. Records 1/0 into check_results so the resumed run uses it without
        re-prompting. Returns the answered step_id, or None if nothing pending."""
        with self._lock:
            ts = self._task_state
            if ts is None or not ts.pending_pose:
                self._log("WARN ", "[SEQ] answer_pose: no pose query pending")
                return None
            step_id = ts.pending_pose["step_id"]
            ts.check_results[step_id] = 1 if value else 0
            ts.pending_pose = None
            self._log(
                "INFO ",
                f"[SEQ] pose answer for {step_id}: "
                f"{'correct' if value else 'incorrect'}",
            )
            return step_id

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def step_next(
        self,
        stream: Optional[str] = None,
        group: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Optional[StepResult]:
        """Execute the next ready step.

        Selection rules (in priority order):
        - If step_id is given: execute that specific step (only if its
          preconditions are met).
        - Else if group is given: pick the first ready step in that group.
        - Else if stream is given: pick the first ready step in that stream.
        - Else: pick the first ready step in YAML file order.

        A "ready" step has all preconditions completed and is not yet completed
        or failed. Returns the StepResult, or None if no ready step matches.
        """
        with self._lock:
            if not self._loaded:
                self._log("WARN ", "[SEQ] step_next: not loaded — call load() first")
                return None

            ts = self._ensure_task_state()

            if step_id is not None:
                chosen = step_id
                step = self._step_map.get(chosen)
                if step is None:
                    self._log("WARN ", f"[SEQ] step_next: unknown step {chosen!r}")
                    return None
                if chosen in ts.completed_steps:
                    self._log("INFO ", f"[SEQ] step_next: {chosen} already completed")
                    return None
                ok, unmet = self._check_preconditions(step, ts)
                if not ok:
                    ts.blocked = True
                    ts.block_reason = f"Unmet preconditions for {chosen}: {unmet}"
                    self._log("WARN ", f"[SEQ] step_next: {ts.block_reason}")
                    return StepResult(
                        step_id=chosen,
                        human_label=step.get("human_label", chosen),
                        success=False,
                        failure_reason=ts.block_reason,
                        assemblies_before=self._snapshot(),
                        assemblies_after=self._snapshot(),
                        executor_mode=step.get("executor_mode", "magic"),
                    )
            else:
                ready = self._ready_steps_locked(ts, stream=stream, group=group)
                if not ready:
                    filt = []
                    if stream: filt.append(f"stream={stream!r}")
                    if group:  filt.append(f"group={group!r}")
                    suffix = f" ({', '.join(filt)})" if filt else ""
                    self._log("INFO ", f"[SEQ] step_next: no ready steps{suffix}")
                    return None
                chosen = ready[0]
                step = self._step_map[chosen]

                # Simultaneous group: when the auto-selected step carries a
                # `simultaneous` tag, run the whole group as one logical step.
                # (Explicit step_id selection above stays single-step.)
                tag = self._step_simultaneous.get(chosen)
                if tag:
                    try:
                        result = self._execute_simultaneous_group(tag, ts)
                    except _PoseQueryPending as pause:
                        self._log(
                            "INFO ",
                            f"[SEQ] paused: awaiting pose answer for "
                            f"{pause.child!r} ([{pause.step_id}])",
                        )
                        return None
                    ts.blocked = False
                    ts.block_reason = ""
                    ts.history.append(result)  # one entry for the whole group
                    return result

            try:
                result = self._execute_step(step, ts)
            except _PoseQueryPending as pause:
                # check_pose deferred to an async prompt. Leave the step
                # un-completed; the run stops here (None reads as "no progress"
                # to the bulk-run loops, which break). answer_pose() + a re-run
                # resumes from this same step.
                self._log(
                    "INFO ",
                    f"[SEQ] paused: awaiting pose answer for "
                    f"{pause.child!r} ([{pause.step_id}])",
                )
                return None
            ts.blocked = False
            ts.block_reason = ""
            ts.history.append(result)

            if result.success:
                ts.completed_steps.append(chosen)
                # Advance current_index advisory marker to the latest position
                # of any completed step in file order.
                if chosen in self._ordered_ids:
                    idx = self._ordered_ids.index(chosen)
                    if idx > ts.current_index:
                        ts.current_index = idx
            else:
                ts.failed_steps.append(chosen)

            return result

    def _execute_simultaneous_group(self, tag: str, ts: TaskState) -> StepResult:
        """Execute all ready members of a simultaneous group as one logical step.

        Members run in intra-group dependency order (mini-Kahn): a member runs
        once its preconditions are met by the steps completed so far — including
        members just completed within this group — so e.g. an M10 bolt combines
        before its nut, while the 3 independent shaft stages all run in one pass.

        Members are recorded individually in completed_steps / failed_steps (so
        validate_state still checks each part); the returned StepResult covers
        the whole group, with its `members` field listing the executed ids. The
        caller appends it to history. Stops at the first failing member. Any
        member left blocked by an *external* precondition is deferred (logged)
        and retried when the group next triggers.
        """
        members = [
            m for m in self._simultaneous_groups.get(tag, [])
            if m not in ts.completed_steps and m not in ts.failed_steps
        ]
        before = self._snapshot()
        results: List[StepResult] = []
        executed: List[str] = []
        mode = "magic"
        failed = False

        remaining = list(members)
        while remaining and not failed:
            progressed = False
            for mid in list(remaining):
                step = self._step_map[mid]
                ok_pre, _unmet = self._check_preconditions(step, ts)
                if not ok_pre:
                    continue  # intra-group dep not yet met, or external block
                result = self._execute_step(step, ts)  # may raise _PoseQueryPending
                results.append(result)
                mode = result.executor_mode or mode
                remaining.remove(mid)
                progressed = True
                if result.success:
                    ts.completed_steps.append(mid)
                    if mid in self._ordered_ids:
                        idx = self._ordered_ids.index(mid)
                        if idx > ts.current_index:
                            ts.current_index = idx
                    executed.append(mid)
                else:
                    ts.failed_steps.append(mid)
                    failed = True
                    break
            if not progressed:
                break  # remaining members blocked by external preconditions

        if remaining and not failed:
            self._log(
                "WARN ",
                f"[SEQ] simultaneous {tag!r}: ran {len(executed)}, deferred "
                f"{len(remaining)} (external preconditions): {remaining}",
            )

        children = [self._step_map[m].get("child") for m in executed]
        label = (f"Simultaneous {tag} ({len(executed)}): "
                 + ", ".join(c for c in children if c))
        reason = None
        if failed:
            reason = "; ".join(
                f"{r.step_id}: {r.failure_reason}" for r in results if not r.success
            )
        after = self._snapshot()
        self._log(
            "INFO " if not failed else "WARN ",
            f"[SEQ] {'OK  ' if not failed else 'FAIL'} group [{tag}] "
            f"({len(executed)} step(s))",
        )
        return StepResult(
            step_id=tag,
            human_label=label,
            success=not failed,
            failure_reason=reason,
            assemblies_before=before,
            assemblies_after=after,
            executor_mode=mode,
            members=executed,
        )

    def step_prev(self) -> bool:
        """Undo the last completed step using separate(). Returns True on success."""
        with self._lock:
            if not self._loaded:
                self._log("WARN ", "[SEQ] step_prev: not loaded")
                return False

            ts = self._ensure_task_state()
            if not ts.completed_steps:
                self._log("INFO ", "[SEQ] step_prev: nothing to undo")
                return False

            last_id = ts.completed_steps[-1]

            # Simultaneous group: the last logical step was a group, so undo
            # every completed member as one unit (reverse intra-group order:
            # combine -> separate; stage/focus -> untrack only, matching the
            # single-step focus behavior below). History is an append-only log
            # (left intact, as for single-step undo).
            tag = self._step_simultaneous.get(last_id)
            if tag:
                members = [m for m in self._simultaneous_groups.get(tag, [])
                           if m in ts.completed_steps]
                for mid in reversed(members):
                    mstep = self._step_map.get(mid, {})
                    if mstep.get("action", "combine") == "combine" and mstep.get("child"):
                        if not self._ma.separate(mstep["child"]):
                            self._log(
                                "WARN ",
                                f"[SEQ] step_prev: separate({mstep['child']!r}) failed",
                            )
                    ts.completed_steps.remove(mid)
                ts.current_index = self._max_completed_index(ts)
                self._log(
                    "INFO ",
                    f"[SEQ] step_prev: undid simultaneous group {tag!r} "
                    f"({len(members)} member(s))",
                )
                return True

            step = self._step_map.get(last_id)
            if step is None:
                self._log("WARN ", f"[SEQ] step_prev: step {last_id!r} not found")
                return False

            action = step.get("action", "combine")
            child = step.get("child")

            if action == "combine" and child:
                ok = self._ma.separate(child)
                if ok:
                    ts.completed_steps.pop()
                    ts.current_index -= 1
                    self._log("INFO ", f"[SEQ] step_prev: undid {last_id} (separated {child!r})")
                else:
                    self._log("WARN ", f"[SEQ] step_prev: separate({child!r}) failed")
                return ok
            elif action == "focus":
                # Cannot meaningfully undo a focus; just back the index up
                ts.completed_steps.pop()
                ts.current_index -= 1
                self._log("INFO ", f"[SEQ] step_prev: skipped undo for focus step {last_id}")
                return True
            else:
                self._log("WARN ", f"[SEQ] step_prev: no undo defined for action={action!r}")
                return False

    def _max_completed_index(self, ts: TaskState) -> int:
        """Highest file-order index among completed steps (-1 if none). Used to
        re-derive the advisory current_index after a multi-member group undo."""
        idxs = [self._ordered_ids.index(c)
                for c in ts.completed_steps if c in self._ordered_ids]
        return max(idxs) if idxs else -1

    # ------------------------------------------------------------------
    # DAG queries: ready steps, streams, groups
    # ------------------------------------------------------------------

    def step_stream(self, step_id: str) -> Optional[str]:
        """Return the stream id that owns this step, via its phase."""
        step = self._step_map.get(step_id)
        if step is None:
            return None
        return self._phase_stream.get(step.get("phase"))

    def step_group(self, step_id: str) -> Optional[str]:
        """Return the group id this step belongs to, or None."""
        return self._step_group.get(step_id)

    def get_ready_steps(
        self,
        stream: Optional[str] = None,
        group: Optional[str] = None,
    ) -> List[str]:
        """Return step_ids whose preconditions are met and not yet completed
        or failed, in YAML file order. Optionally filter by stream or group."""
        with self._lock:
            ts = self._ensure_task_state()
            return self._ready_steps_locked(ts, stream=stream, group=group)

    def _ready_steps_locked(
        self,
        ts: TaskState,
        stream: Optional[str] = None,
        group: Optional[str] = None,
    ) -> List[str]:
        """Caller must hold self._lock. Returns ready steps filtered by stream/group."""
        completed = set(ts.completed_steps)
        failed = set(ts.failed_steps)
        ready: List[str] = []
        for sid in self._ordered_ids:
            if sid in completed or sid in failed:
                continue
            step = self._step_map[sid]
            preconditions = step.get("preconditions") or []
            if not all(p in completed for p in preconditions):
                continue
            if stream is not None:
                if self._phase_stream.get(step.get("phase")) != stream:
                    continue
            if group is not None:
                if self._step_group.get(sid) != group:
                    continue
            ready.append(sid)
        return ready

    def get_active_streams(self) -> List[str]:
        """Return ids of streams that currently have at least one ready step."""
        with self._lock:
            ts = self._ensure_task_state()
            active = []
            for sid in self._streams:
                if self._ready_steps_locked(ts, stream=sid):
                    active.append(sid)
            return active

    def stream_progress(self, stream_id: str) -> dict:
        """Return progress within a stream: completed/total/ready counts."""
        spec = self._streams.get(stream_id)
        if spec is None:
            return {"error": f"unknown stream {stream_id!r}"}
        phases = set(spec.get("phases", []) or [])
        members = [s["id"] for s in self._steps if s.get("phase") in phases]
        with self._lock:
            ts = self._ensure_task_state()
            completed = sum(1 for m in members if m in ts.completed_steps)
            failed = sum(1 for m in members if m in ts.failed_steps)
            ready = len(self._ready_steps_locked(ts, stream=stream_id))
        return {
            "stream": stream_id,
            "total": len(members),
            "completed": completed,
            "failed": failed,
            "ready": ready,
            "remaining": len(members) - completed,
            "can_parallel_with": spec.get("can_parallel_with", []),
        }

    # ------------------------------------------------------------------
    # Bulk execution: streams and groups
    # ------------------------------------------------------------------

    def run_stream(self, stream_id: str, delay_s: float = 0.1) -> List[StepResult]:
        """Run all ready+future steps in a single stream until none remain ready.
        Stops on first failure. Returns list of StepResults."""
        if not self._loaded:
            self._log("WARN ", f"[SEQ] run_stream({stream_id!r}): not loaded")
            return []
        if stream_id not in self._streams:
            self._log("WARN ", f"[SEQ] run_stream: unknown stream {stream_id!r}")
            return []

        self._log("INFO ", f"[SEQ] run_stream({stream_id!r}): begin")
        results: List[StepResult] = []
        while True:
            result = self.step_next(stream=stream_id)
            if result is None:
                break
            results.append(result)
            if not result.success:
                self._log(
                    "WARN ",
                    f"[SEQ] run_stream({stream_id!r}): stopped — {result.failure_reason}",
                )
                break
            if delay_s > 0:
                time.sleep(delay_s)
        ok = sum(1 for r in results if r.success)
        self._log("INFO ", f"[SEQ] run_stream({stream_id!r}): done — {ok}/{len(results)} succeeded")
        return results

    def run_group(self, group_id: str, delay_s: float = 0.1) -> List[StepResult]:
        """Run all ready members of a group until none remain ready. Stops on
        first failure. Useful for batches like '/run_group('m6_hub_bolts')'."""
        if not self._loaded:
            self._log("WARN ", f"[SEQ] run_group({group_id!r}): not loaded")
            return []
        if group_id not in self._groups:
            self._log("WARN ", f"[SEQ] run_group: unknown group {group_id!r}")
            return []

        self._log("INFO ", f"[SEQ] run_group({group_id!r}): begin")
        results: List[StepResult] = []
        while True:
            result = self.step_next(group=group_id)
            if result is None:
                break
            results.append(result)
            if not result.success:
                self._log(
                    "WARN ",
                    f"[SEQ] run_group({group_id!r}): stopped — {result.failure_reason}",
                )
                break
            if delay_s > 0:
                time.sleep(delay_s)
        ok = sum(1 for r in results if r.success)
        self._log("INFO ", f"[SEQ] run_group({group_id!r}): done — {ok}/{len(results)} succeeded")
        return results

    def run_sequence(self, delay_s: float = 0.3) -> int:
        """Run all remaining ready steps until none remain.
        Picks ready steps in YAML file order (canonical topological sort).
        delay_s: pause between steps (seconds) to allow sim to settle.
        Returns count of steps that succeeded."""
        if not self._loaded:
            self._log("WARN ", "[SEQ] run_sequence: not loaded — call load() first")
            return 0

        succeeded = 0
        ts = self._ensure_task_state()
        remaining = len(self._ordered_ids) - len(ts.completed_steps) - len(ts.failed_steps)
        self._log("INFO ", f"[SEQ] run_sequence: {remaining} steps remaining")

        while True:
            result = self.step_next()
            if result is None:
                break
            if result.success:
                succeeded += 1
            else:
                self._log(
                    "WARN ",
                    f"[SEQ] run_sequence: stopped at {result.step_id} — {result.failure_reason}",
                )
                break
            if delay_s > 0:
                time.sleep(delay_s)

        self._log("INFO ", f"[SEQ] run_sequence: done — {succeeded} steps succeeded")
        return succeeded

    def run_alias(self, alias_name: str, delay_s: float = 0.1) -> List[StepResult]:
        """Execute all steps referenced by a shortcut alias (e.g. 'combine_casing_base').
        Skips steps already completed. Returns list of StepResults."""
        if not self._loaded:
            self._log("WARN ", f"[SEQ] run_alias({alias_name!r}): not loaded")
            return []

        step_ids = self._aliases.get(alias_name)
        if step_ids is None:
            self._log("WARN ", f"[SEQ] run_alias: unknown alias {alias_name!r}")
            return []

        ts = self._ensure_task_state()
        results = []
        self._log("INFO ", f"[SEQ] run_alias({alias_name!r}): {len(step_ids)} steps")

        for step_id in step_ids:
            if step_id in ts.completed_steps:
                self._log("INFO ", f"[SEQ] run_alias: {step_id} already done, skipping")
                continue

            step = self._step_map.get(step_id)
            if step is None:
                self._log("WARN ", f"[SEQ] run_alias: step {step_id!r} not in sequence")
                continue

            with self._lock:
                try:
                    result = self._execute_step(step, ts)
                except _PoseQueryPending as pause:
                    self._log(
                        "INFO ",
                        f"[SEQ] run_alias({alias_name!r}): paused — awaiting pose "
                        f"answer for {pause.child!r} ([{pause.step_id}])",
                    )
                    break
                # Advance current_index to cover this step
                if step_id in self._ordered_ids:
                    idx = self._ordered_ids.index(step_id)
                    if idx > ts.current_index:
                        ts.current_index = idx
                ts.history.append(result)
                if result.success:
                    ts.completed_steps.append(step_id)
                else:
                    ts.failed_steps.append(step_id)

            results.append(result)
            if not result.success:
                self._log(
                    "WARN ",
                    f"[SEQ] run_alias: stopped — {result.failure_reason}",
                )
                break
            if delay_s > 0:
                time.sleep(delay_s)

        ok = sum(1 for r in results if r.success)
        self._log("INFO ", f"[SEQ] run_alias({alias_name!r}): {ok}/{len(results)} succeeded")
        return results

    def jump_to_step(self, step_id: str) -> Optional[StepResult]:
        """Execute a specific step by id regardless of current position.
        Does NOT update current_index sequencing — use for one-off execution."""
        if not self._loaded:
            self._log("WARN ", f"[SEQ] jump_to_step: not loaded")
            return None

        step = self._step_map.get(step_id)
        if step is None:
            self._log("WARN ", f"[SEQ] jump_to_step: unknown step {step_id!r}")
            return None

        with self._lock:
            ts = self._ensure_task_state()
            try:
                result = self._execute_step(step, ts)
            except _PoseQueryPending as pause:
                self._log(
                    "INFO ",
                    f"[SEQ] jump_to_step: paused — awaiting pose answer for "
                    f"{pause.child!r} ([{pause.step_id}])",
                )
                return None
            ts.history.append(result)
            if result.success and step_id not in ts.completed_steps:
                ts.completed_steps.append(step_id)
            elif not result.success:
                ts.failed_steps.append(step_id)

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_state(self) -> dict:
        """Compare actual assembly state (from MagicAssemblyManager) against
        expected state based on completed steps. Returns a validation report."""
        if not self._loaded:
            return {"error": "not loaded"}

        ts = self._ensure_task_state()
        actual = dict(self._snapshot())  # child -> parent

        expected_attachments: Dict[str, str] = {}
        for step_id in ts.completed_steps:
            step = self._step_map.get(step_id, {})
            if step.get("action") == "combine":
                child = step.get("child")
                parent = step.get("parent")
                if child and parent:
                    expected_attachments[child] = parent

        missing = []    # expected but not in actual
        wrong_parent = []  # attached to wrong parent
        unexpected = []    # in actual but not expected

        for child, expected_parent in expected_attachments.items():
            actual_parent = actual.get(child)
            if actual_parent is None:
                missing.append({"child": child, "expected_parent": expected_parent})
            elif actual_parent != expected_parent:
                wrong_parent.append({
                    "child": child,
                    "expected_parent": expected_parent,
                    "actual_parent": actual_parent,
                })

        for child, actual_parent in actual.items():
            if child not in expected_attachments:
                unexpected.append({"child": child, "actual_parent": actual_parent})

        ok = not missing and not wrong_parent
        report = {
            "ok": ok,
            "completed_steps": len(ts.completed_steps),
            "expected_attachments": len(expected_attachments),
            "actual_attachments": len(actual),
            "missing": missing,
            "wrong_parent": wrong_parent,
            "unexpected": unexpected,
        }

        level = "INFO " if ok else "WARN "
        self._log(level, f"[SEQ] validate_state: ok={ok}, "
                         f"missing={len(missing)}, wrong_parent={len(wrong_parent)}")
        if missing:
            for m in missing:
                self._log("WARN ", f"[SEQ]   MISSING: {m['child']} -> {m['expected_parent']}")
        if wrong_parent:
            for w in wrong_parent:
                self._log("WARN ", f"[SEQ]   WRONG: {w['child']} -> {w['actual_parent']} "
                                   f"(expected {w['expected_parent']})")
        return report

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current TaskState as a plain dict (safe to log/display).
        Uses DAG semantics: 'next_step' is the first ready step in file order,
        and per-stream ready counts are included."""
        if self._task_state is None:
            return {
                "loaded": self._loaded,
                "started": False,
                "total_steps": len(self._steps),
                "streams": list(self._streams.keys()),
                "groups": list(self._groups.keys()),
            }

        with self._lock:
            ts = self._task_state
            ready_ids = self._ready_steps_locked(ts)
            next_step = None
            if ready_ids:
                sid = ready_ids[0]
                s = self._step_map[sid]
                next_step = {
                    "id": sid,
                    "human_label": s.get("human_label", sid),
                    "stream": self._phase_stream.get(s.get("phase")),
                    "group": self._step_group.get(sid),
                }

            # Per-stream readiness summary
            stream_ready: Dict[str, int] = {}
            for stream_id in self._streams:
                stream_ready[stream_id] = len(
                    self._ready_steps_locked(ts, stream=stream_id)
                )

            # Logical step accounting: each simultaneous group counts as one
            # logical step (it executes in a single tick). Sub-steps = members.
            grouped_ids = set(self._step_simultaneous)
            completed_set = set(ts.completed_steps)
            logical_total = (
                (len(self._steps) - len(grouped_ids)) + len(self._simultaneous_groups)
            )
            logical_completed = (
                len([c for c in ts.completed_steps if c not in grouped_ids])
                + sum(1 for members in self._simultaneous_groups.values()
                      if all(m in completed_set for m in members))
            )

            return {
                "loaded": self._loaded,
                "started": ts.current_index >= 0 or bool(ts.completed_steps),
                "sequence_id": ts.sequence_id,
                "total_steps": ts.total_steps,
                "logical_total": logical_total,
                "logical_completed": logical_completed,
                "current_index": ts.current_index,
                "completed": len(ts.completed_steps),
                "failed": len(ts.failed_steps),
                "skipped": len(ts.skipped_steps),
                "ready_count": len(ready_ids),
                "blocked": ts.blocked,
                "block_reason": ts.block_reason,
                "next_step": next_step,
                "stream_ready_counts": stream_ready,
                "active_streams": [s for s, n in stream_ready.items() if n > 0],
                "progress_pct": round(
                    100 * len(ts.completed_steps) / max(ts.total_steps, 1), 1
                ),
            }

    def format_status(self) -> str:
        """Return a human-readable status string for UI display."""
        s = self.status()
        if not s["loaded"]:
            return "Sequence not loaded"
        if not s["started"]:
            return f"Ready — {s['total_steps']} steps"
        lines = [
            f"Progress: {s['completed']}/{s['total_steps']} steps ({s['progress_pct']}%)",
            f"Ready: {s.get('ready_count', 0)} steps  |  Active streams: {', '.join(s.get('active_streams', [])) or '(none)'}",
        ]
        if s.get("logical_total", s["total_steps"]) != s["total_steps"]:
            lines.insert(
                1,
                f"Logical: {s['logical_completed']}/{s['logical_total']} "
                f"({s['completed']}/{s['total_steps']} sub-steps; "
                f"simultaneous groups collapsed)",
            )
        if s["next_step"]:
            ns = s["next_step"]
            stream_tag = f" [{ns.get('stream')}]" if ns.get("stream") else ""
            group_tag = f" ({ns.get('group')})" if ns.get("group") else ""
            lines.append(f"Next: [{ns['id']}]{stream_tag}{group_tag} {ns['human_label']}")
        if s["blocked"]:
            lines.append(f"BLOCKED: {s['block_reason']}")
        if s["failed"]:
            lines.append(f"Failed steps: {s['failed']}")
        return "  |  ".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_step(self, step: dict, ts: TaskState) -> StepResult:
        """Execute a single step. Preconditions must be checked by the caller."""
        step_id = step["id"]
        action = step.get("action", "combine")
        child = step.get("child")
        parent = step.get("parent")
        plug = step.get("plug")
        socket = step.get("socket")
        mode = step.get("executor_mode", "magic")
        label = step.get("human_label", step_id)

        before = self._snapshot()
        ok = False
        reason = None

        # Conditional step gate (e.g. a flip that only fires when a prior
        # check_pose reported the part is mis-posed). The condition references a
        # check_pose step by id and the value it must equal. If the recorded
        # result doesn't match, the step is skipped as a success no-op (not a
        # failure) so the sequence keeps advancing.
        condition = step.get("condition")
        if condition:
            check_id = condition.get("check")
            want = condition.get("equals")
            got = ts.check_results.get(check_id)
            if got != want:
                self._log(
                    "INFO ",
                    f"[SEQ] SKIP [{step_id}] {label} — condition not met "
                    f"({check_id}={got}, needs {want})",
                )
                return StepResult(
                    step_id=step_id,
                    human_label=label,
                    success=True,
                    failure_reason=f"skipped: condition {check_id}={got} != {want}",
                    assemblies_before=before,
                    assemblies_after=before,
                    executor_mode=mode,
                )

        self._log("INFO ", f"[SEQ] execute: [{step_id}] {label}")

        try:
            if action == "combine":
                if not child or not parent:
                    reason = "missing child or parent"
                else:
                    # Ensure casing attachment assets exist before combining
                    if parent in _CASING_PARENTS:
                        try:
                            self._ma.ensure_case_attachment_assets()
                        except Exception:
                            pass  # non-fatal; combine will fail if truly missing
                    ok = self._ma.combine(child, parent, plug, socket)
                    if not ok:
                        reason = f"MagicAssemblyManager.combine failed"

            elif action == "separate":
                if not child:
                    reason = "missing child for separate"
                else:
                    ok = self._ma.separate(child)
                    if not ok:
                        reason = "MagicAssemblyManager.separate failed"

            elif action == "focus":
                if not child:
                    reason = "missing child for focus"
                else:
                    ok = self._ma.focus(child)
                    if not ok:
                        reason = "MagicAssemblyManager.focus failed"

            elif action == "unfocus":
                if not child:
                    reason = "missing child for unfocus"
                else:
                    unfocus_fn = getattr(self._ma, "unfocus", None)
                    if unfocus_fn is None:
                        ok = True
                        reason = "MagicAssemblyManager.unfocus not implemented — no-op"
                    else:
                        ok = unfocus_fn(child)
                        if not ok:
                            reason = "MagicAssemblyManager.unfocus failed"

            elif action == "upright":
                if not child:
                    reason = "missing child for upright"
                elif mode == "robot":
                    # Spec: upright is not robot-executable.
                    reason = "upright action is not valid for executor_mode=robot"
                    ok = False
                else:
                    upright_fn = getattr(self._ma, "upright", None)
                    if upright_fn is None:
                        ok = True
                        reason = "MagicAssemblyManager.upright not implemented — no-op"
                    else:
                        axis = str(step.get("axis", "x")).lower()
                        # Re-seat onto the table after the 90° rotation so the
                        # part rests on its new down-face (matches /upright slash
                        # command behavior).
                        ok = upright_fn(child, axis, seat=True)
                        if not ok:
                            reason = "MagicAssemblyManager.upright failed"

            elif action == "flip":
                if not child:
                    reason = "missing child for flip"
                else:
                    flip_fn = getattr(self._ma, "flip", None)
                    if flip_fn is None:
                        ok = True
                        reason = "MagicAssemblyManager.flip not implemented — no-op"
                    else:
                        axis = str(step.get("axis", "x")).lower()
                        # Re-seat onto the table after the 180° flip so the part
                        # rests on its new down-face (matches /flip slash command).
                        ok = flip_fn(child, axis, seat=True)
                        if not ok:
                            reason = "MagicAssemblyManager.flip failed"

            elif action == "check_pose":
                # Non-manipulation: evaluate the part's pose and record a 1/0
                # result for any conditional step that references this step id.
                # A pre-recorded answer (e.g. supplied via answer_pose() after an
                # async UI prompt) is used as-is. Otherwise the pose resolver is
                # consulted: it returns True/False to answer inline, or None to
                # defer to an async prompt — in which case we pause the run by
                # raising _PoseQueryPending (a BaseException, so the broad
                # except below does not catch it).
                if step_id in ts.check_results:
                    result = ts.check_results[step_id]
                else:
                    answer = self._pose_resolver(child, step_id)
                    if answer is None:
                        ts.pending_pose = {"step_id": step_id, "child": child or ""}
                        raise _PoseQueryPending(step_id, child)
                    result = 1 if answer else 0
                    ts.check_results[step_id] = result
                ok = True
                reason = f"pose ok={bool(result)}"

            elif action == "stage":
                # Pre-position the part directly above its target socket (lifted
                # by hover_m) so the later `combine` drops it straight down. Uses
                # MagicAssemblyManager.stage when available; otherwise falls back
                # to a plain hover (lift in place) or a logged no-op.
                try:
                    hover_m = float(step.get("hover_m") or 0.15)
                except (TypeError, ValueError):
                    hover_m = 0.15
                stage_fn = getattr(self._ma, "stage", None)
                if child and parent and stage_fn is not None:
                    ok = stage_fn(child, parent, plug, socket, hover_m=hover_m)
                    reason = "staged above socket" if ok else "stage failed"
                else:
                    hover_fn = getattr(self._ma, "hover", None)
                    if child and hover_fn is not None:
                        try:
                            hover_fn(child)
                        except Exception:
                            pass  # cosmetic only
                    ok = True
                    reason = "staged (hover-in-place fallback)"

            elif action == "inspect":
                # inspect steps always succeed — they're validation triggers
                ok = True

            else:
                reason = f"unknown action: {action!r}"

        except Exception as exc:
            reason = f"exception: {exc}"
            ok = False

        after = self._snapshot()

        level = "INFO " if ok else "WARN "
        self._log(level, f"[SEQ] {'OK  ' if ok else 'FAIL'} [{step_id}]"
                         + (f" — {reason}" if reason else ""))

        return StepResult(
            step_id=step_id,
            human_label=label,
            success=ok,
            failure_reason=reason,
            assemblies_before=before,
            assemblies_after=after,
            executor_mode=mode,
        )

    def _check_preconditions(
        self, step: dict, ts: TaskState
    ) -> Tuple[bool, List[str]]:
        """Returns (all_met, list_of_unmet_step_ids)."""
        preconditions = step.get("preconditions") or []
        unmet = [p for p in preconditions if p not in ts.completed_steps]
        return (len(unmet) == 0), unmet

    def _snapshot(self) -> List[Tuple[str, str]]:
        """Return current assembly attachment list from MagicAssemblyManager."""
        try:
            return self._ma.list_assemblies()
        except Exception:
            return []
