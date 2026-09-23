"""Generate anonymous public/evaluator manifests for the five-task MVP."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .constrained_tasks import DEFAULT_CONFIG, load_config, validate_config


def _anonymous_ids(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    ids: list[str] = []
    while len(ids) < n:
        candidate = "D-" + "".join(rng.choice(alphabet) for _ in range(6))
        if candidate not in ids:
            ids.append(candidate)
    return ids


def generate(config: dict[str, Any], out_dir: Path, seed: int = 1205) -> None:
    errors = validate_config(config)
    if errors:
        raise ValueError("invalid task config:\n- " + "\n- ".join(errors))
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = _anonymous_ids(10, seed)
    public_lines: list[dict[str, Any]] = []
    evaluator_lines: list[dict[str, Any]] = []
    index = 0
    for task in config["tasks"]:
        for variant in ("HARD", "COMMUTABLE"):
            domain_id = domains[index]
            pair_id = f"P-{index // 2 + 1:02d}"
            index += 1
            public = {
                "domain_id": domain_id,
                "pair_id": pair_id,
                "media": {"canonical_video": f"media/canonical/{domain_id}.mp4"},
                "goal": task["goal"],
                "action_a": task["action_a"],
                "action_b": task["action_b"],
                "candidate_procedures": [["A", "B"], ["B", "A"], ["A"], ["B"]],
                "model_request": {
                    "goal": task["goal"],
                    "initial_state": task["initial_state"],
                    "action_a": task["action_a"]["text"],
                    "action_b": task["action_b"]["text"],
                    "demonstrated_order": ["A", "B"]
                }
            }
            oracle = {
                "domain_id": domain_id,
                "source_task_id": task["task_id"],
                "variant": variant,
                "relation": task["variants"][variant]["relation"],
                "can_b_before_a": task["variants"][variant]["can_b_before_a"],
                "feasible_procedures": {
                    "A>B": True,
                    "B>A": task["variants"][variant]["can_b_before_a"],
                    "A": False,
                    "B": False
                },
                "intervention": task["variants"][variant],
                "oracle_evidence": f"outputs/oracle/{domain_id}.json"
            }
            public_lines.append(public)
            evaluator_lines.append(oracle)
    for filename, lines in (("public_manifest.jsonl", public_lines), ("evaluator_manifest.jsonl", evaluator_lines)):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tasks" / "manifests")
    parser.add_argument("--seed", type=int, default=1205)
    args = parser.parse_args()
    generate(load_config(args.config), args.out_dir, args.seed)
    print(f"wrote manifests to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
