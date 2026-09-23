"""Emit auditable scene-authoring recipes from the five-task contract.

The recipes are deliberately data-only.  An Isaac scene builder can consume them
to create USD layers, but cannot silently choose different visual and collision
parameters.  This keeps CAD authoring separate from the offline relation oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constrained_tasks import DEFAULT_CONFIG, load_config, validate_config


def make_recipe(config: dict[str, Any], task: dict[str, Any], variant: str) -> dict[str, Any]:
    variant_cfg = task["variants"][variant]
    return {
        "schema_version": "0.1.0-mvp",
        "task_id": task["task_id"],
        "variant": variant,
        "source_visual_parts": [config["source_assets"][p]["visual"] for p in task["parts"]],
        "source_measurements": {
            p: config["source_assets"][p]["bbox_mm"] for p in task["parts"]
        },
        "stage": {
            "meters_per_unit": config["units"]["stage_meters_per_unit"],
            "up_axis": config["units"]["stage_up_axis"],
            "source_to_stage_scale": 0.001
        },
        "shared_mutation_parameters": task["geometry"],
        "variant_parameters": variant_cfg,
        "render_collision_contract": {
            "parameter_source": "shared_mutation_parameters",
            "render_feature": "generated from the recipe's cutout/slot/pocket primitive",
            "collision_feature": "the exact same primitive dimensions and transform, decomposed only for solver stability",
            "forbidden": ["solid convex hull over an opening", "hidden blocker", "teleport/snap attachment"]
        },
        "calibration": {
            "tolerance_mm": config["shared_contract"]["calibration_tolerance_mm"],
            "required_margin_mm": config["shared_contract"]["minimum_intervention_margin_mm"],
            "required_checks": [
                "canonical A>B succeeds in both variants",
                "reverse B>A fails for HARD and succeeds for COMMUTABLE",
                "zero penetration at start and terminal stability",
                "camera close view exposes the changed feature",
                "same controller and seeds are used for paired variants"
            ]
        }
    }


def write_recipes(config: dict[str, Any], out_dir: Path) -> None:
    errors = validate_config(config)
    if errors:
        raise ValueError("invalid task config:\n- " + "\n- ".join(errors))
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in config["tasks"]:
        for variant in ("HARD", "COMMUTABLE"):
            path = out_dir / f"{task['task_id'].lower()}_{variant.lower()}.json"
            path.write_text(json.dumps(make_recipe(config, task, variant), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "scenes" / "recipes")
    args = parser.parse_args()
    write_recipes(load_config(args.config), args.out_dir)
    print(f"wrote 10 scene recipes to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
