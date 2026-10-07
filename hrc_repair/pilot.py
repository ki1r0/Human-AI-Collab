"""Small serial pilot runner; every episode keeps its own artifact folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .run import run_episode


def parse_range(value: str) -> list[int]:
    if ":" not in value:
        return [int(value)]
    left, right = value.split(":", 1)
    return list(range(int(left), int(right)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/repair_m0.yaml")
    parser.add_argument("--seeds", default="0:1")
    parser.add_argument("--method", choices=("auto", "fixed_retry", "repair"), default="fixed_retry")
    parser.add_argument("--scenario", choices=("nominal", "blocked"), default="nominal")
    args = parser.parse_args(argv)
    config_path, config = load_config(args.config)
    results = []
    for seed in parse_range(args.seeds):
        results.append(run_episode(config_path, config, method=args.method, scenario=args.scenario, seed=seed))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
