"""Check recorded source STL bounds against the files in the repository."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from .constrained_tasks import DEFAULT_CONFIG, ROOT, load_config


def stl_bbox(path: Path) -> tuple[float, float, float]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"STL too short: {path}")
    triangles = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + 50 * triangles
    if abs(expected - len(raw)) > 4:
        raise ValueError(f"only binary STL is supported by this audit: {path}")
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for index in range(triangles):
        offset = 84 + 50 * index + 12
        for vertex in range(3):
            point = struct.unpack_from("<3f", raw, offset + 12 * vertex)
            for axis, value in enumerate(point):
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
    return tuple(maximum - minimum for minimum, maximum in zip(mins, maxs))


def audit(config: dict, repo_root: Path = ROOT, tolerance_mm: float = 0.02) -> list[dict]:
    rows = []
    for name, asset in config["source_assets"].items():
        path = repo_root / asset["measurement"]
        observed = stl_bbox(path)
        expected = tuple(float(value) for value in asset["bbox_mm"])
        errors = tuple(abs(a - b) for a, b in zip(observed, expected))
        if max(errors) > tolerance_mm:
            raise RuntimeError(f"{name}: recorded bbox {expected} differs from STL {observed}: {errors}")
        rows.append({"part": name, "bbox_mm": list(observed), "max_abs_error_mm": max(errors)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    rows = audit(load_config(args.config))
    print(json.dumps({"status": "PASS", "parts": rows}, indent=2))
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Source STL geometry audit",
            "",
            "Recorded task-pack dimensions match the repository STL bounds within 0.02 mm.",
            "",
            "| Part | Measured bbox (mm) | Max absolute error (mm) |",
            "|---|---|---:|",
        ]
        for row in rows:
            b = ", ".join(f"{v:.3f}" for v in row["bbox_mm"])
            lines.append(f"| {row['part']} | `{b}` | {row['max_abs_error_mm']:.6f} |")
        args.write_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
