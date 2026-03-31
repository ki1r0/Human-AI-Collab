from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.paths import usd_python_paths  # noqa: E402


def ensure_pxr_paths() -> list[str]:
    added: list[str] = []
    for path in usd_python_paths():
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    return added
