#!/usr/bin/env python3
"""tools/center_usd_assets.py — Rigorous CAD-origin centering for USD part files.

Permanently recenter the local origin of each USD asset to the exact geometric
centroid of its mesh geometry using OpenUSD's BBoxCache (axis-aligned bounding
box, computed from actual vertex data).

MATH
----
For a USD file with structure:

    World [Xform, translate=(0,0,0)]
      node_ [Xform, translate=(0,0,0)]     ← target prim
        mesh_ [Mesh]                       ← vertex centroid at (Cx, Cy, Cz)

After centering:

    node_.translate = existing_translate − (Cx, Cy, Cz)

Proof: world position of geometry centroid
    = node_.translate_new + (Cx, Cy, Cz)
    = (0 − C) + C = (0, 0, 0)  ✓

When this file is later referenced into a scene and the reference prim is
placed at position P by MagicAssemblyManager, the geometry's centroid appears
exactly at P.

USAGE
-----
# Process every .usd in assets/parts/ (IN-PLACE):
python3 tools/center_usd_assets.py assets/parts/

# Process a single file (in-place):
python3 tools/center_usd_assets.py "assets/parts/Casing Base.usd"

# Write to a separate output directory (original files untouched):
python3 tools/center_usd_assets.py assets/parts/ --out assets/parts_centered/

# Preview only (dry run, no files written):
python3 tools/center_usd_assets.py assets/parts/ --dry-run

# Explicit target prim path (overrides auto-detection):
python3 tools/center_usd_assets.py "assets/parts/Casing Base.usd" --target-prim /World/node_

Run from the repository root:

    PYTHONPATH=<usd_lib_path> <isaacsim_python> tools/center_usd_assets.py assets/parts/

The script discovers pxr from the active USD / Isaac Sim environment.
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — make runtime.asset_utils importable from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools._bootstrap import ensure_pxr_paths

ensure_pxr_paths()

# Also look for pxr in common Isaac Sim packman cache locations.
from runtime.asset_utils import center_stage_file, CenterResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_usd_files(path: str) -> list[str]:
    """Return all .usd / .usda / .usdc files under path (file or directory)."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    result = []
    for root, _dirs, files in os.walk(path):
        for fname in sorted(files):
            if fname.lower().endswith((".usd", ".usda", ".usdc")):
                result.append(os.path.join(root, fname))
    return result


def _make_output_path(input_path: str, out_dir: Optional[str], suffix: str) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, os.path.basename(input_path))
    if suffix:
        base, ext = os.path.splitext(input_path)
        return base + suffix + ext
    return input_path  # in-place


# mypy stub
from typing import Optional


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="center_usd_assets.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to a .usd file or a directory containing .usd files.",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Output directory.  If omitted, files are overwritten in-place.",
    )
    parser.add_argument(
        "--suffix",
        metavar="SUFFIX",
        default="",
        help="Suffix appended to the output filename before the extension "
             "(e.g. '_centered').  Ignored when --out is given.",
    )
    parser.add_argument(
        "--target-prim",
        metavar="PATH",
        default=None,
        help="Override auto-detection: explicit USD prim path to center "
             "(e.g. /World/node_).  Applied to every processed file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen but do not write any files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed diagnostic messages.",
    )

    args = parser.parse_args(argv)

    files = _collect_usd_files(args.input)
    if not files:
        print(f"[WARN] No USD files found at: {args.input}")
        return 1

    results: list[CenterResult] = []
    n_ok = 0
    n_fail = 0

    for usd_file in files:
        out_path = _make_output_path(usd_file, args.out, args.suffix)
        rel = os.path.relpath(usd_file, _REPO_ROOT)

        if args.dry_run:
            print(f"[DRY-RUN] Would center: {rel}  →  {os.path.relpath(out_path, _REPO_ROOT)}")
            continue

        logger = print if args.verbose else (lambda _: None)
        result = center_stage_file(
            input_path=usd_file,
            output_path=out_path,
            target_prim_path=args.target_prim,
            logger=logger,
        )
        results.append(result)

        if result.success:
            c = result.centroid_before
            print(
                f"  OK   {rel}"
                f"  target={result.target_prim_path}"
                f"  centroid_before=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})"
            )
            n_ok += 1
        else:
            print(f"  FAIL {rel}  — {result.message}")
            n_fail += 1

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  Processed {n_ok + n_fail} files: {n_ok} OK, {n_fail} failed.")
        print(f"{'='*60}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
