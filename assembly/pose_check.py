"""assembly/pose_check.py — interactive runtime pose evaluator.

`check_pose(object)` returns whether `object` is already in the correct pose for
the operation that gated it. The conditional flips injected by the instantiator
fire when this returns False.

Current implementation prompts the human operator: it asks whether the part's
pose is correct and returns True/False from a Y/N answer, re-asking on any
other input. When stdin is not interactive (headless playback, automated
tests), it falls back to True ("correct") so non-interactive runs do not block
or crash — matching the original stub behavior.

Replace the prompt with a real test (GT/USD orientation, later VLM) when one is
available.
"""
from __future__ import annotations

_YES = {"y", "yes"}
_NO = {"n", "no"}


def check_pose(obj: str) -> bool:
    """Ask the operator whether ``obj`` is in the correct pose.

    Returns True for yes, False for no. Re-prompts on any other input. Falls
    back to True if stdin is unavailable (non-interactive session).
    """
    prompt = f"Is the pose of '{obj}' correct? [Y/N]: "
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            # Non-interactive stdin (headless playback / tests): default to
            # "correct" so paired conditional flips are skipped, as before.
            return True
        if answer in _YES:
            return True
        if answer in _NO:
            return False
        print(f"Please answer Y (yes) or N (no); got '{answer}'.")
