"""Render a compact MP4 from an Isaac M1 state trace.

This is deliberately a *state-trace visualization*, not an RGB-camera replay:
it draws the measured Hub root, gripper-link poses, and calibrated socket in
top/side views.  It is useful when RTX camera rendering is unavailable or too
slow, while the JSON trace remains the source of physical evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
try:
    import imageio_ffmpeg  # type: ignore

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    # A system ffmpeg may be available outside the Isaac container.
    pass


def _interp(a: list[float], b: list[float], fraction: float) -> list[float]:
    return [x + (y - x) * fraction for x, y in zip(a, b)]


def _frames(records: list[dict], per_segment: int) -> list[tuple[dict, float]]:
    result: list[tuple[dict, float]] = []
    for index in range(len(records) - 1):
        start, end = records[index], records[index + 1]
        for step in range(per_segment):
            result.append(({"start": start, "end": end}, step / per_segment))
    result.append(({"start": records[-1], "end": records[-1]}, 0.0))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-segment", type=int, default=12)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    data = json.loads(args.metrics.read_text())
    records = data["records"]
    frames = _frames(records, max(2, args.per_segment))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "M1 direct-policy state trace"},
    )
    socket_x, socket_y, casing_z = 0.55, 0.08683718, 1.0
    target_z = casing_z + 0.062

    with writer.saving(fig, str(args.output), dpi=110):
        for pair, fraction in frames:
            start, end = pair["start"], pair["end"]
            hub = _interp(start["hub_root_state"][:3], end["hub_root_state"][:3], fraction)
            link1 = _interp(start["bodies"]["left_gripper_link1"][:3], end["bodies"]["left_gripper_link1"][:3], fraction)
            link2 = _interp(start["bodies"]["left_gripper_link2"][:3], end["bodies"]["left_gripper_link2"][:3], fraction)
            link6 = _interp(start["bodies"]["left_arm_link6"][:3], end["bodies"]["left_arm_link6"][:3], fraction)
            label = start["label"] if fraction < 0.5 else end["label"]
            top.clear()
            side.clear()

            # Top view: casing footprint, output socket, annular Hub, and the
            # two measured gripper fingers.  The Hub is drawn as a schematic
            # annulus; its pose/contacts are the measured values in the trace.
            top.add_patch(Rectangle((0.33, -0.277), 0.44, 0.554, facecolor="#f4a261", alpha=0.35, edgecolor="#b5651d", lw=2))
            top.add_patch(Circle((socket_x, socket_y), 0.13, facecolor="#d62828", alpha=0.7, edgecolor="#7f0000", lw=2))
            top.add_patch(Circle((socket_x, socket_y), 0.065, facecolor="#f7f7f7", edgecolor="#7f0000", lw=2))
            top.add_patch(Circle((hub[0], hub[1]), 0.13, facecolor="#d62828", alpha=0.85, edgecolor="#550000", lw=2))
            top.add_patch(Circle((hub[0], hub[1]), 0.065, facecolor="#f7f7f7", edgecolor="#550000", lw=2))
            top.plot([link6[0], link1[0]], [link6[1], link1[1]], color="#264653", lw=3)
            top.plot([link6[0], link2[0]], [link6[1], link2[1]], color="#264653", lw=3)
            top.scatter([link1[0], link2[0]], [link1[1], link2[1]], c=["#1d3557", "#457b9d"], s=35, zorder=5)
            top.scatter([link6[0]], [link6[1]], c="#111111", s=25, zorder=5)
            top.set_xlim(0.18, 0.82)
            top.set_ylim(-0.32, 0.52)
            top.set_aspect("equal")
            top.set_title("Top view (XY)")
            top.set_xlabel("world X [m]")
            top.set_ylabel("world Y [m]")
            top.grid(alpha=0.2)

            # Side view: casing top plane and the Hub's measured vertical drop.
            side.add_patch(Rectangle((0.33, 0.98), 0.44, 0.08, facecolor="#f4a261", alpha=0.45, edgecolor="#b5651d", lw=2))
            side.add_patch(Rectangle((hub[0] - 0.13, hub[2] - 0.014), 0.26, 0.028, facecolor="#d62828", alpha=0.85, edgecolor="#550000", lw=2))
            side.plot([link6[0], link1[0]], [link6[2], link1[2]], color="#264653", lw=3)
            side.plot([link6[0], link2[0]], [link6[2], link2[2]], color="#264653", lw=3)
            side.scatter([link1[0], link2[0]], [link1[2], link2[2]], c=["#1d3557", "#457b9d"], s=35, zorder=5)
            side.axhline(target_z, color="#2a9d8f", ls="--", lw=1.5, label="seated root z")
            side.set_xlim(0.18, 0.82)
            side.set_ylim(0.95, 1.58)
            side.set_aspect("equal")
            side.set_title("Side view (XZ)")
            side.set_xlabel("world X [m]")
            side.set_ylabel("world Z [m]")
            side.grid(alpha=0.2)

            force = end.get("hub_casing_force_norm", 0.0)
            contacts = end.get("gripper_to_hub_contact_force_norm_N", {})
            c1 = contacts.get("left_gripper_link1_contact", 0.0)
            c2 = contacts.get("left_gripper_link2_contact", 0.0)
            fig.suptitle(
                "M1 direct-policy state trace (not RGB camera)\n"
                f"stage={label}  Hub=({hub[0]:.3f}, {hub[1]:.3f}, {hub[2]:.3f}) m  "
                f"finger contacts={float(c1):.1f}/{float(c2):.1f} N  casing={float(force):.1f} N",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            writer.grab_frame()

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
