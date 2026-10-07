#!/usr/bin/env python3
"""Check that the isolated physics trials cover every canonical combine step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical_combine_ids(path: Path) -> list[str]:
    current = None
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- id: "):
            current = line[len("- id: "):].strip()
        elif line.strip() == "action: combine" and current is not None:
            result.append(current)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="assembly/instances/canonical_grouped.yaml")
    parser.add_argument("--config", default="assembly/physics_validation.json")
    args = parser.parse_args(argv)
    sequence = canonical_combine_ids(Path(args.sequence))
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    covered = [
        step
        for trial in payload["trials"]
        for step in trial.get("canonical_steps", [])
    ]
    duplicates = sorted({step for step in covered if covered.count(step) > 1})
    missing = [step for step in sequence if step not in covered]
    unknown = [step for step in covered if step not in sequence]
    result = {
        "canonical_combine_steps": len(sequence),
        "covered_step_references": len(covered),
        "unique_covered_steps": len(set(covered)),
        "missing": missing,
        "unknown": unknown,
        "duplicates": duplicates,
        "interface_trials": len(payload["trials"]),
    }
    print(json.dumps(result, indent=2))
    return 0 if not missing and not unknown and len(set(covered)) == len(sequence) else 1


if __name__ == "__main__":
    raise SystemExit(main())
