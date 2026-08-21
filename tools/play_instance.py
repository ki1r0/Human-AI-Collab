#!/usr/bin/env python3
"""Headless real-USD playback of a generated assembly instance.

Opens the scene USD, builds a real MagicAssemblyManager (no Kit), points a
SequenceRunner at assembly/instances/<variant>.yaml, runs the whole sequence,
then reports validate_state(). This is the "does it actually assemble" check
that complements the dependency-free tools/test_instance_playback.py.

    docker compose run --rm hac tools/run_tool.sh tools/play_instance.py
    docker compose run --rm hac tools/run_tool.sh tools/play_instance.py --variant random_topo
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from tools._bootstrap import ensure_pxr_paths  # noqa: E402

ensure_pxr_paths()

from pxr import Usd  # noqa: E402

from assembly.sequence_runner import SequenceRunner  # noqa: E402
from runtime.magic_assembly import MagicAssemblyManager  # noqa: E402

_INSTANCES = Path(_REPO_ROOT) / "assembly" / "instances"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="canonical_grouped")
    ap.add_argument("--scene", default=os.path.join(_REPO_ROOT, "assets", "simple_room_scene.usd"))
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between steps")
    ap.add_argument("--verbose", action="store_true", help="print every runner log line")
    ap.add_argument("--pose", choices=("correct", "incorrect"), default="correct",
                    help="non-interactive check_pose answer for all check_pose steps "
                         "('incorrect' fires the gated conditional flips)")
    args = ap.parse_args()

    inst_path = _INSTANCES / f"{args.variant}.yaml"
    if not inst_path.exists():
        avail = ", ".join(sorted(p.stem for p in _INSTANCES.glob("*.yaml")))
        print(f"[ERROR] unknown variant {args.variant!r}. Available: {avail}")
        return 2

    print(f"\n{'='*72}\n  play instance — variant={args.variant!r} "
          f"scene={os.path.basename(args.scene)}\n{'='*72}")

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        print("[ERROR] could not open stage")
        return 1

    def log(level, msg):
        if args.verbose or level.strip() == "WARN":
            print(f"  {level.strip():4s} {msg}")

    ma = MagicAssemblyManager(stage_fn=lambda: stage, use_omni_commands=False,
                              logger=lambda m: None)
    runner = SequenceRunner(magic_assembly=ma, log_fn=log, sequence_path=inst_path)
    # Headless: answer all check_pose steps non-interactively (the default
    # resolver would prompt via input() on a TTY and stall this tool).
    _pose_ok = (args.pose == "correct")
    runner.set_pose_resolver(lambda child, step_id: _pose_ok)
    if not runner.load():
        print("[ERROR] runner.load() failed")
        return 1

    total = len(runner._steps)
    n = runner.run_sequence(delay_s=args.delay)
    ts = runner._task_state
    completed = len(ts.completed_steps)
    failed = len(ts.failed_steps)

    report = runner.validate_state()
    print(f"\n  steps: {completed}/{total} completed, {failed} failed "
          f"({n} reported by run_sequence)")
    print(f"  validate_state: ok={report['ok']} "
          f"missing={len(report['missing'])} wrong_parent={len(report['wrong_parent'])}")
    if failed:
        print(f"  FAILED steps: {ts.failed_steps}")
    if report["missing"]:
        for m in report["missing"][:10]:
            print(f"    MISSING {m['child']} -> {m['expected_parent']}")

    ok = (failed == 0 and completed == total and report["ok"])
    print(f"\n{'='*72}\n  RESULT: {'PASS' if ok else 'FAIL'}\n{'='*72}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
