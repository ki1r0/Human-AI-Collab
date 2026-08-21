#!/usr/bin/env python3
"""Offline playback test for generated instances through SequenceRunner.

Drives a generated instance YAML (default: canonical_grouped) end-to-end through
SequenceRunner.run_sequence() against a fake MagicAssemblyManager that just
records calls. No Isaac Sim / pxr required — this validates the new runner
semantics (stage, check_pose, conditional flip skipping) and precondition
ordering, not the actual USD reparenting.

    python3 tools/test_instance_playback.py
    # or, inside the container, identical result:
    docker compose run --rm hac tools/run_tool.sh tools/test_instance_playback.py
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from assembly.sequence_runner import SequenceRunner  # noqa: E402

_INSTANCES = Path(_REPO_ROOT) / "assembly" / "instances"


class FakeMagic:
    """Records calls; every operation succeeds. Tracks parent of each child."""

    def __init__(self):
        self.calls = {
            "combine": [], "separate": [], "focus": [], "unfocus": [],
            "upright": [], "flip": [], "stage": [], "hover": [],
        }
        self._attach = {}  # child -> parent

    def combine(self, child, parent, plug, socket):
        self.calls["combine"].append((child, parent, plug, socket))
        self._attach[child] = parent
        return True

    def separate(self, child):
        self.calls["separate"].append(child)
        self._attach.pop(child, None)
        return True

    def focus(self, child):
        self.calls["focus"].append(child)
        return True

    def unfocus(self, child):
        self.calls["unfocus"].append(child)
        return True

    def upright(self, child, axis="x", seat=False):
        self.calls["upright"].append((child, axis, seat))
        return True

    def flip(self, child, axis="x", seat=False):
        self.calls["flip"].append((child, axis, seat))
        return True

    def hover(self, child, surface="table", margin=0.005):
        self.calls["hover"].append((child, surface))
        return True

    def stage(self, child, parent, plug=None, socket=None, hover_m=0.15):
        self.calls["stage"].append((child, parent, plug, socket, hover_m))
        return True

    def list_assemblies(self):
        return list(self._attach.items())

    def ensure_case_attachment_assets(self):
        return {}


def _silent(level, msg):
    pass


def _load(variant):
    ma = FakeMagic()
    runner = SequenceRunner(
        magic_assembly=ma,
        log_fn=_silent,
        sequence_path=_INSTANCES / f"{variant}.yaml",
    )
    assert runner.load(), f"load() failed for {variant}"
    # Deterministic, non-interactive pose answer (mirrors the historical stub:
    # always "correct"). The production default resolver prompts via input(),
    # which would block these offline tests on a terminal. Individual tests
    # override this to exercise the deferred / pause path.
    runner.set_pose_resolver(lambda child, step_id: True)
    return runner, ma


def _run_until_group(runner, tag):
    """step_next until the last result is the simultaneous group `tag`; return it."""
    while True:
        r = runner.step_next()
        if r is None:
            return None
        if r.members and r.step_id == tag:
            return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_canonical_73_steps():
    runner, _ = _load("canonical_grouped")
    assert len(runner._steps) == 73, len(runner._steps)


def test_full_playback_completes_all_steps():
    """run_sequence drives every one of the 73 steps to success (conditional
    flips are skipped-as-success, not failures)."""
    runner, _ = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    ts = runner._task_state
    assert ts is not None
    assert len(ts.failed_steps) == 0, ts.failed_steps
    assert len(ts.completed_steps) == 73, (
        f"completed {len(ts.completed_steps)}/73; "
        f"blocked={ts.blocked} reason={ts.block_reason!r}"
    )


def test_combine_called_for_every_combine_step():
    """49 combine actions in canonical: 6 covers + 24 M6 + 3 shafts + 1 mate
    + 6 M10 bolts + 6 nuts + 3 accessories."""
    runner, ma = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    assert len(ma.calls["combine"]) == 49, len(ma.calls["combine"])


def test_conditional_flips_are_skipped_with_stub_check():
    """check_pose stub returns True -> result 1; conditional flips want
    equals 0, so none fire. The fake's flip must never be called."""
    runner, ma = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    assert len(ma.calls["flip"]) == 0, ma.calls["flip"]


def test_unconditional_upright_runs():
    """inst_056 upright has no condition -> executes once, with seat=True."""
    runner, ma = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    assert len(ma.calls["upright"]) == 1, ma.calls["upright"]
    assert ma.calls["upright"][0][2] is True  # seat=True


def test_check_pose_results_recorded():
    """7 check_pose steps; each stores result 1 (stub True) in task state."""
    runner, _ = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    results = runner._task_state.check_results
    assert len(results) == 7, results
    assert all(v == 1 for v in results.values()), results


def test_stage_steps_position_above_socket_without_combine():
    """3 stage steps run via ma.stage(child, parent, plug, socket, hover_m);
    they position above the target socket and must NOT reparent (no combine)."""
    runner, ma = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    stage_ids = [s["id"] for s in runner._steps if s.get("action") == "stage"]
    assert len(stage_ids) == 3, stage_ids
    completed = set(runner._task_state.completed_steps)
    assert all(sid in completed for sid in stage_ids)
    # stage children are the 3 shafts; combine still happens later via insert ops
    staged = {s["child"] for s in runner._steps if s.get("action") == "stage"}
    assert staged == {"Input_Shaft", "Output_Shaft", "Transfer_Shaft"}, staged
    # Each stage targets its socket on Casing_Base with a positive hover.
    assert len(ma.calls["stage"]) == 3, ma.calls["stage"]
    targets = {(c, p, s) for (c, p, _plug, s, _h) in ma.calls["stage"]}
    assert targets == {
        ("Input_Shaft", "Casing_Base", "socket_gear_input"),
        ("Output_Shaft", "Casing_Base", "socket_gear_output"),
        ("Transfer_Shaft", "Casing_Base", "socket_gear_transfer"),
    }, targets
    assert all(isinstance(h, float) and h > 0 for (*_x, h) in ma.calls["stage"])


def test_all_variants_complete():
    """Every generated variant plays back to full completion."""
    for variant in ("canonical_grouped", "interleaved_covers", "random_topo"):
        runner, _ = _load(variant)
        runner.run_sequence(delay_s=0)
        ts = runner._task_state
        assert len(ts.completed_steps) == 73, (
            f"{variant}: {len(ts.completed_steps)}/73 "
            f"blocked={ts.blocked} {ts.block_reason!r}"
        )
        assert len(ts.failed_steps) == 0, f"{variant}: {ts.failed_steps}"


def test_deferred_resolver_pauses_run():
    """A resolver returning None (defer to async prompt) pauses run_sequence at
    the first check_pose: pending_pose is set and the run is incomplete."""
    runner, _ = _load("canonical_grouped")
    runner.set_pose_resolver(lambda child, step_id: None)
    runner.run_sequence(delay_s=0)
    ts = runner._task_state
    assert ts.pending_pose is not None, "expected a pending pose query"
    assert "step_id" in ts.pending_pose and "child" in ts.pending_pose
    assert len(ts.completed_steps) < 73, "run should have paused, not completed"
    # The paused check_pose step is NOT recorded as completed or failed.
    paused_id = ts.pending_pose["step_id"]
    assert paused_id not in ts.completed_steps
    assert paused_id not in ts.failed_steps


def test_answer_pose_resumes_and_fires_conditional_flip():
    """answer_pose(False) records result 0; re-running resumes and the gated
    flip (condition equals 0) fires for that part. Remaining check_poses answer
    True so only the one flip occurs and the run completes."""
    runner, ma = _load("canonical_grouped")
    runner.set_pose_resolver(lambda child, step_id: None)
    runner.run_sequence(delay_s=0)
    paused_id = runner.pending_pose["step_id"]
    answered = runner.answer_pose(False)
    assert answered == paused_id, (answered, paused_id)
    assert runner.pending_pose is None, "answer_pose must clear the pause"
    # Remaining check_poses answer 'correct' so only the answered one flips.
    runner.set_pose_resolver(lambda child, step_id: True)
    runner.run_sequence(delay_s=0)
    ts = runner._task_state
    assert len(ts.completed_steps) == 73, (
        f"{len(ts.completed_steps)}/73 blocked={ts.blocked} {ts.block_reason!r}"
    )
    assert len(ts.failed_steps) == 0, ts.failed_steps
    assert len(ma.calls["flip"]) == 1, ma.calls["flip"]


def test_answer_pose_without_pending_is_noop():
    runner, _ = _load("canonical_grouped")
    assert runner.answer_pose(True) is None


def test_reset_clears_check_results():
    runner, _ = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    assert runner._task_state.check_results  # populated
    runner.reset()
    # fresh task state on next access has empty check_results
    ts = runner._ensure_task_state()
    assert ts.check_results == {}


def test_simultaneous_run_sequence_logical_count():
    """73 sub-steps collapse to 63 logical steps (stage_shafts 3->1,
    insert_shafts 3->1, m10_pair_01..06 2->1 x6). Members still all tracked."""
    runner, _ = _load("canonical_grouped")
    n = runner.run_sequence(delay_s=0)
    ts = runner._task_state
    assert n == 63, n
    assert len(ts.history) == 63, len(ts.history)
    assert len(ts.completed_steps) == 73, len(ts.completed_steps)
    assert len(ts.failed_steps) == 0, ts.failed_steps


def test_simultaneous_groups_are_single_history_entries():
    runner, _ = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    groups = [r for r in runner._task_state.history if r.members]
    tags = {r.step_id for r in groups}
    assert tags == {"stage_shafts", "insert_shafts"} | {
        f"m10_pair_0{i}" for i in range(1, 7)
    }, tags
    by_tag = {r.step_id: r for r in groups}
    assert len(by_tag["stage_shafts"].members) == 3
    assert len(by_tag["insert_shafts"].members) == 3
    assert len(by_tag["m10_pair_01"].members) == 2


def test_simultaneous_m10_bolt_combined_before_nut():
    """Intra-group order: each nut combines after its bolt."""
    runner, ma = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    order = [c[0] for c in ma.calls["combine"]]  # child names, call order
    for i in range(1, 7):
        bolt, nut = f"M10_Casing_Bolt_0{i}", f"M10_Casing_Nut_0{i}"
        assert order.index(bolt) < order.index(nut), (i, order)


def test_simultaneous_step_next_runs_whole_group_at_once():
    runner, ma = _load("canonical_grouped")
    r = _run_until_group(runner, "stage_shafts")
    assert r is not None and r.success
    assert set(r.members) == {
        "inst_049_stage_input_shaft",
        "inst_050_stage_output_shaft",
        "inst_051_stage_transfer_shaft",
    }, r.members
    assert len(ma.calls["stage"]) == 3, ma.calls["stage"]


def test_step_prev_undoes_whole_simultaneous_group():
    runner, _ = _load("canonical_grouped")
    r = _run_until_group(runner, "stage_shafts")
    members = list(r.members)
    assert all(m in runner._task_state.completed_steps for m in members)
    assert runner.step_prev() is True
    assert all(m not in runner._task_state.completed_steps for m in members)


def test_step_prev_undoes_m10_pair_separating_both_in_reverse():
    runner, ma = _load("canonical_grouped")
    r = _run_until_group(runner, "m10_pair_01")
    assert set(r.members) == {
        "inst_057_combine_m10_casing_bolt_01",
        "inst_063_combine_m10_casing_nut_01",
    }, r.members
    ma.calls["separate"].clear()
    assert runner.step_prev() is True
    # reverse intra-group order: nut detaches before bolt
    assert ma.calls["separate"] == ["M10_Casing_Nut_01", "M10_Casing_Bolt_01"], \
        ma.calls["separate"]


def test_jump_to_step_runs_single_member_not_group():
    """Explicit /run_step on a group member is a granular override: just that
    one step runs (no group), proving simultaneity is auto-select-only."""
    runner, ma = _load("canonical_grouped")
    r = runner.jump_to_step("inst_049_stage_input_shaft")
    assert r is not None and r.success
    assert r.members == []
    assert runner._task_state.completed_steps == ["inst_049_stage_input_shaft"]
    assert len(ma.calls["stage"]) == 1


def test_status_reports_logical_step_counts():
    runner, _ = _load("canonical_grouped")
    runner.run_sequence(delay_s=0)
    s = runner.status()
    assert s["total_steps"] == 73
    assert s["logical_total"] == 63, s["logical_total"]
    assert s["logical_completed"] == 63, s["logical_completed"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
