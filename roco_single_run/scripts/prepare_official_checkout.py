#!/usr/bin/env python3
"""Apply audited runtime compatibility patches to a pinned RoCo checkout.

The official checkout remains the source of truth.  This script refuses an
unexpected commit and only performs exact, idempotent substitutions for defects
observed in Isaac Sim 5.1.  It also supplies neutral maps for two texture
references whose files are absent from the public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import zlib
from pathlib import Path


PINNED_COMMIT = "094a1f76d18c207caec198315f23b1a60dbca94f"


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _patch_robot_bundle(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    r1_line = "ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_BUNDLE"
    if re.search(rf"^{re.escape(r1_line)}$", source, flags=re.MULTILINE):
        return "already-r1"

    r1_lite_pattern = r"^ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_LITE_BUNDLE$"
    patched, count = re.subn(r1_lite_pattern, r1_line, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not identify active robot assignment in {path}")
    path.write_text(patched, encoding="utf-8")
    return "selected-r1"


def _patch_collision_offsets(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    negative_original = (
        "collision_props=sim_utils.CollisionPropertiesCfg("
        "contact_offset=0.0, rest_offset=-0.0005)"
    )
    negative_patched = (
        "collision_props=sim_utils.CollisionPropertiesCfg("
        "contact_offset=0.0001, rest_offset=-0.0005)"
    )
    carrier_original = (
        "collision_props=sim_utils.CollisionPropertiesCfg("
        "contact_offset=0.0, rest_offset=0.0005)"
    )
    carrier_patched = (
        "collision_props=sim_utils.CollisionPropertiesCfg("
        "contact_offset=0.001, rest_offset=0.0005)"
    )

    def active_count(fragment: str) -> int:
        return len(re.findall(rf"^        {re.escape(fragment)}", source, flags=re.MULTILINE))

    negative_original_count = active_count(negative_original)
    negative_patched_count = active_count(negative_patched)
    if negative_original_count not in (0, 3) or negative_patched_count not in (0, 3):
        raise RuntimeError(
            "Unexpected negative-rest collision offset layout: "
            f"original={negative_original_count}, patched={negative_patched_count}"
        )
    if negative_original_count == 3:
        source = source.replace(negative_original, negative_patched)

    carrier_original_count = active_count(carrier_original)
    carrier_patched_count = active_count(carrier_patched)
    if carrier_original_count not in (0, 1) or carrier_patched_count not in (0, 1):
        raise RuntimeError(
            "Unexpected carrier collision offset layout: "
            f"original={carrier_original_count}, patched={carrier_patched_count}"
        )
    if carrier_original_count == 1:
        source = source.replace(carrier_original, carrier_patched)

    if (
        len(re.findall(rf"^        {re.escape(negative_patched)}", source, flags=re.MULTILINE)) != 3
        or len(re.findall(rf"^        {re.escape(carrier_patched)}", source, flags=re.MULTILINE)) != 1
    ):
        raise RuntimeError("Collision offset postcondition failed")
    path.write_text(source, encoding="utf-8")
    return {
        "negative_rest_assets_patched": negative_original_count,
        "carrier_assets_patched": carrier_original_count,
    }


def _png_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Encode a tiny deterministic RGB PNG using only the standard library."""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    pixels = row * height
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels, level=9)) + chunk(b"IEND", b"")


def _supply_missing_table_textures(table_dir: Path) -> dict:
    texture_dir = table_dir / "Textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "OakTable_N.png": _png_rgb(4, 4, (128, 128, 255)),
        "OakTable_R.png": _png_rgb(4, 4, (128, 128, 128)),
    }
    manifest = {}
    for name, payload in payloads.items():
        target = texture_dir / name
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite unexpected texture: {target}")
        target.write_bytes(payload)
        manifest[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": "neutral normal" if name.endswith("_N.png") else "neutral roughness",
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to the official gearboxAssembly checkout")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    head = _git_head(checkout)
    if head != PINNED_COMMIT:
        raise RuntimeError(f"Expected RoCo commit {PINNED_COMMIT}, found {head}")

    extension_root = checkout / "source" / "Galaxea_Lab_External"
    package_root = extension_root / "Galaxea_Lab_External"
    result = {
        "checkout": str(checkout),
        "commit": head,
        "robot_bundle": _patch_robot_bundle(package_root / "robots" / "robot_bundles.py"),
        "collision_offsets": _patch_collision_offsets(package_root / "robots" / "gears_assets.py"),
        "table_textures": _supply_missing_table_textures(extension_root / "assets" / "Props" / "table"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
