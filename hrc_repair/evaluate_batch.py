"""Serial held-out batch entry point with failure-preserving output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .run import parse_range, run_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/repair_m0.yaml")
    parser.add_argument("--methods", nargs="+", default=["auto", "fixed_retry", "repair"])
    parser.add_argument("--seeds", default="100:110")
    parser.add_argument("--scenario", choices=("nominal", "blocked"), default="nominal")
    args = parser.parse_args(argv)
    config_path, config = load_config(args.config)
    results = []
    for method in args.methods:
        for seed in parse_range(args.seeds):
            try:
                results.append(run_episode(config_path, config, method=method, scenario=args.scenario, seed=seed))
            except Exception as exc:  # preserve a failed episode and continue
                results.append({"method": method, "scenario": args.scenario, "seed": seed, "termination": "system_error", "error_type": type(exc).__name__})
    out = Path(config.get("experiment", {}).get("output_root", "runs/repair_m0")) / f"batch_{args.scenario}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
