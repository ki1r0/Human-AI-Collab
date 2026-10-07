#!/usr/bin/env python3
"""Summarize per-canonical-ID collision-on mating logs.

Logs are applied in argument order; a later run supersedes an earlier run for
the same canonical ID.  This makes repair/recheck iterations explicit without
silently counting an obsolete failed attempt as the final result.
"""

from __future__ import annotations

import argparse
import json
import os
import re


PASS_RE = re.compile(r"PHYSICS (PASS|FAIL) (?P<trial>[^:]+):")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="assembly/physics_validation.json")
    ap.add_argument("--log", action="append", required=True,
                    help="log file; later occurrences supersede earlier ones")
    ap.add_argument("--out", help="optional JSON summary path")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        payload = json.load(stream)
    expected = [step for trial in payload["trials"]
                for step in trial.get("canonical_steps", [])]
    records: dict[str, dict] = {}
    for log_path in args.log:
        with open(log_path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = PASS_RE.search(line)
                if not match or "__" not in match.group("trial"):
                    continue
                step_id = match.group("trial").split("__", 1)[1]
                records[step_id] = {
                    "status": match.group(1),
                    "trial": match.group("trial"),
                    "log": log_path,
                }
    missing = sorted(set(expected) - records.keys())
    unknown = sorted(set(records) - set(expected))
    failures = sorted(step for step in expected
                      if records.get(step, {}).get("status") != "PASS")
    summary = {
        "canonical_steps": len(expected),
        "observed_unique_steps": len(records),
        "pass": sum(record["status"] == "PASS" for record in records.values()),
        "fail": sum(record["status"] == "FAIL" for record in records.values()),
        "missing": missing,
        "unknown": unknown,
        "failed_final_ids": failures,
        "records": [records[step] for step in expected if step in records],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return 0 if not missing and not unknown and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
