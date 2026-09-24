"""Render an auditable gear-geometry demonstration.

This is deliberately a diagnostic renderer, not an Isaac simulation.  It reads
the source ``Output Gear.usd`` visual mesh and its ``collision_gear`` proxy,
renders both from the same coordinates, and writes a high-resolution image and
short rotation video.  The output is meant to make the current fidelity claim
falsifiable: the visual CAD contains teeth, while the collision proxy is a
lower-resolution toothed annulus and gear-to-gear running contact is still not
validated.

Run with the USD-capable environment::

    conda run -n lingbot-va python -m pilot_12pair.oracle.render_geometry_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mesh_data(stage_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return visual and collision triangles from the Output Gear asset."""

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {stage_path}")
    meshes: dict[str, tuple[np.ndarray, np.ndarray, str, int]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(counts):
            continue
        # The imported visual mesh is triangular, while the collision proxy may
        # use quads.  A fan triangulation preserves the authored polygon without
        # pretending that this is a watertight simulation mesh.
        faces: list[list[int]] = []
        offset = 0
        for count in counts.tolist():
            count = int(count)
            polygon = indices[offset : offset + count].tolist()
            offset += count
            if count < 3:
                continue
            for j in range(1, count - 1):
                faces.append([int(polygon[0]), int(polygon[j]), int(polygon[j + 1])])
        triangles = points[np.asarray(faces, dtype=np.int64)]
        path = str(prim.GetPath())
        key = "collision" if "collision_gear" in path.lower() else "visual"
        meshes[key] = (points, triangles, path, int(len(counts)))

    missing = {key for key in ("visual", "collision") if key not in meshes}
    if missing:
        raise RuntimeError(f"missing meshes {sorted(missing)} in {stage_path}")
    visual_points, visual_triangles, visual_path, visual_polygons = meshes["visual"]
    collision_points, collision_triangles, collision_path, collision_polygons = meshes["collision"]
    meta = {
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "visual_path": visual_path,
        "collision_path": collision_path,
        "visual_points": int(len(visual_points)),
        "visual_faces": int(visual_polygons),
        "collision_points": int(len(collision_points)),
        "collision_faces": int(collision_polygons),
        "collision_render_triangles": int(len(collision_triangles)),
    }
    return visual_triangles, collision_triangles, meta


def _normalise(triangles: np.ndarray) -> np.ndarray:
    """Center XY coordinates but preserve Z for face colouring."""

    centered = triangles.copy()
    center = np.mean(centered.reshape(-1, 3), axis=0)
    centered[:, :, 0] -= center[0]
    centered[:, :, 1] -= center[1]
    return centered


def _top_faces(triangles: np.ndarray) -> np.ndarray:
    """Keep cap-like faces for a legible top view while retaining tooth detail."""

    z = triangles[:, :, 2]
    zmax = float(np.max(z))
    zmin = float(np.min(z))
    span = max(zmax - zmin, 1e-9)
    # Keep faces close to either cap.  If an imported mesh has no clean caps,
    # fall back to every face so the demonstration never silently disappears.
    cap = np.mean(z, axis=1)
    keep = (cap >= zmax - 0.04 * span) | (cap <= zmin + 0.04 * span)
    return triangles[keep] if int(np.count_nonzero(keep)) >= 20 else triangles


def _add_mesh(ax, triangles: np.ndarray, color: str, alpha: float, label: str):
    from matplotlib.collections import PolyCollection

    top = _top_faces(triangles)
    # Projection to XY is valid for this source asset: its 27-unit thickness is
    # along Z, while the 161.157-unit toothed annulus lies in XY.
    polys = top[:, :, :2]
    collection = PolyCollection(polys, facecolors=color, edgecolors="none", alpha=alpha)
    ax.add_collection(collection)
    ax.plot([], [], color=color, linewidth=8, alpha=min(1.0, alpha + 0.25), label=label)


def _rotate_xy(triangles: np.ndarray, angle_deg: float) -> np.ndarray:
    radians = np.deg2rad(angle_deg)
    c, s = float(np.cos(radians)), float(np.sin(radians))
    xy = triangles[:, :, :2]
    rotated = triangles.copy()
    rotated[:, :, 0] = c * xy[:, :, 0] - s * xy[:, :, 1]
    rotated[:, :, 1] = s * xy[:, :, 0] + c * xy[:, :, 1]
    return rotated


def _limits(visual: np.ndarray, collision: np.ndarray) -> tuple[float, float]:
    all_xy = np.concatenate([visual.reshape(-1, 3)[:, :2], collision.reshape(-1, 3)[:, :2]])
    radius = float(np.max(np.linalg.norm(all_xy, axis=1)))
    return -radius * 1.08, radius * 1.08


def _style_axis(ax, title: str, lim: tuple[float, float]):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("source X (asset units)", fontsize=9)
    ax.set_ylabel("source Y (asset units)", fontsize=9)
    ax.grid(True, linewidth=0.35, alpha=0.25)


def _new_figure(visual: np.ndarray, collision: np.ndarray, meta: dict[str, object], angle: float = 0.0):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10), dpi=120, facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.78])
    ax_v = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_o = fig.add_subplot(grid[1, 0])
    ax_t = fig.add_subplot(grid[1, 1])

    lim = _limits(visual, collision)
    visual_rot = _rotate_xy(visual, angle)
    collision_rot = _rotate_xy(collision, angle)

    _add_mesh(ax_v, visual_rot, "#2563eb", 0.72, "imported visual CAD")
    _style_axis(ax_v, "A. Source visual mesh (actual teeth)", lim)
    ax_v.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _add_mesh(ax_c, collision_rot, "#f59e0b", 0.80, "collision_gear proxy")
    _style_axis(ax_c, "B. Current collision proxy", lim)
    ax_c.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _add_mesh(ax_o, visual_rot, "#2563eb", 0.24, "visual CAD")
    _add_mesh(ax_o, collision_rot, "#f59e0b", 0.33, "collision proxy")
    _style_axis(ax_o, "C. Overlay: shape evidence, not contact proof", lim)
    ax_o.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax_t.axis("off")
    visual_faces = int(meta["visual_faces"])
    collision_faces = int(meta["collision_faces"])
    text = (
        "WHAT THIS DEMO ESTABLISHES\n"
        "• The source visual asset contains a toothed annulus.\n"
        f"• Visual mesh: {meta['visual_points']:,} points / {visual_faces:,} triangular faces.\n"
        f"• Collision proxy: {meta['collision_points']:,} points / {collision_faces:,} authored polygons.\n"
        "• Both meshes are loaded from the same Output Gear.usd.\n\n"
        "WHAT IT DOES NOT ESTABLISH\n"
        "• Tooth-by-tooth gear-to-gear clearance.\n"
        "• A no-penetration PhysX trajectory under rotation.\n"
        "• That the five mutation fixtures are booleans in the CAD parts.\n\n"
        "NEXT PHYSICS CHECK\n"
        "Compose the real gear/shaft poses, sweep relative phase and center\n"
        "distance, then run collision-on PhysX with contact/penetration logs."
    )
    ax_t.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.42,
        family="DejaVu Sans",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    fig.suptitle(
        f"Output Gear geometry audit — diagnostic rotation {angle:05.1f}°",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Source: assets/parts/Output Gear.usd | rendered in source asset coordinates; no Isaac simulation is implied",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    return fig


def _figure_frame(fig) -> np.ndarray:
    import cv2

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def render(stage_path: Path, out_dir: Path, video_frames: int = 48) -> dict[str, object]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visual, collision, meta = _mesh_data(stage_path)
    visual = _normalise(visual)
    collision = _normalise(collision)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "output_gear_visual_vs_collision.png"
    fig = _new_figure(visual, collision, meta, 0.0)
    fig.savefig(image_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # A short, self-contained MP4 lets the viewer rotate the same two meshes.
    first = _new_figure(visual, collision, meta, 0.0)
    frame = _figure_frame(first)
    height, width = frame.shape[:2]
    video_path = out_dir / "output_gear_visual_vs_collision.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    if not writer.isOpened():
        plt.close(first)
        raise RuntimeError(f"could not open video writer for {video_path}")
    writer.write(frame)
    plt.close(first)
    for i in range(1, max(1, video_frames)):
        angle = 360.0 * i / max(1, video_frames - 1)
        fig = _new_figure(visual, collision, meta, angle)
        writer.write(_figure_frame(fig))
        plt.close(fig)
    writer.release()

    # Keep a machine-readable sidecar without introducing a new experiment gate.
    sidecar = {
        **meta,
        "source_asset": str(stage_path),
        "image": str(image_path),
        "video": str(video_path),
        "video_frames": int(max(1, video_frames)),
        "interpretation": "diagnostic source-mesh comparison; not a physics validation",
    }
    (out_dir / "manifest.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path("assets/parts/Output Gear.usd"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pilot_12pair/outputs/geometry_demo"),
    )
    parser.add_argument("--video-frames", type=int, default=48)
    args = parser.parse_args()
    result = render(args.stage, args.out_dir, max(1, args.video_frames))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
