"""Run the offline five-task contract and print a compact relation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constrained_tasks import DEFAULT_CONFIG, expected_table, load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    rows = []
    for row in expected_table(config):
        rows.append({
            "task_id": row["task_id"],
            "variant": row["variant"],
            "A>B": row["AB"]["valid"],
            "B>A": row["BA"]["valid"],
            "A-only": row["A_only"]["valid"],
            "B-only": row["B_only"]["valid"],
        })
    result = {"status": "PASS", "domains": rows}
    print(json.dumps(result, indent=2))
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Five-task offline contract report",
            "",
            "This report is pre-Isaac validation. It does not replace collision/contact calibration.",
            "",
            "| Task | Variant | A→B | B→A | A only | B only |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(f"| {row['task_id']} | {row['variant']} | {str(row['A>B'])} | {str(row['B>A'])} | {str(row['A-only'])} | {str(row['B-only'])} |")
        args.write_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
