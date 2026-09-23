"""Audit generated USD fixture layers with a USD-capable Python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(scene_dir: Path) -> list[dict]:
    from pxr import Usd, UsdPhysics

    rows = []
    for path in sorted(scene_dir.glob("*.usda")):
        stage = Usd.Stage.Open(str(path))
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise RuntimeError(f"invalid/default prim missing: {path}")
        collisions = sum(1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
        references = sum(1 for prim in stage.Traverse() if prim.HasAuthoredReferences())
        if collisions < 1 or references < 1:
            raise RuntimeError(f"scene lacks source references or collision geometry: {path}")
        rows.append({
            "scene": path.name,
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "collision_prims": collisions,
            "source_reference_prims": references,
            "prim_count": sum(1 for _ in stage.Traverse()),
        })
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 generated scenes, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=Path(__file__).resolve().parents[1] / "scenes" / "generated")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    rows = audit(args.scene_dir)
    print(json.dumps({"status": "PASS", "scenes": rows}, indent=2))
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Generated USD fixture audit",
            "",
            "This checks USD composition, source references, and collision API presence. It is not Isaac physics calibration.",
            "",
            "| Scene | Default prim | Collision prims | Source refs | Total prims |",
            "|---|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(f"| {row['scene']} | `{row['default_prim']}` | {row['collision_prims']} | {row['source_reference_prims']} | {row['prim_count']} |")
        args.write_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
