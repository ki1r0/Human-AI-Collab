#!/usr/bin/env python3
"""Standalone test for the check_pose stub. Run: python3 tools/test_pose_check.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assembly.pose_check import check_pose


def test_check_pose_returns_true_for_any_object():
    assert check_pose("Casing_Base") is True
    assert check_pose("anything") is True


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"[PASS] {name}")
            except Exception as e:
                fails += 1; print(f"[FAIL] {name}: {e}")
    sys.exit(1 if fails else 0)
