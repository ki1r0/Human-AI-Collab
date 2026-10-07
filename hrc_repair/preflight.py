"""Run and persist P0 binding checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/repair_m0.yaml")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    config_path, config = load_config(args.config)
    result = preflight(config_path, config, strict=False)
    out = Path(args.out) if args.out else config_path.parent / "repair_m0_preflight.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if args.strict and result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
