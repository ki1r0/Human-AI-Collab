#!/usr/bin/env python3
"""Render recorded physics telemetry, explicitly not a live Isaac scene video."""
import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run/"trace.jsonl").read_text().splitlines()]
    config = json.loads((args.run/"configuration.json").read_text())
    time = np.array([r["sim_time_s"] for r in rows])
    state = np.array([r["cover_root_state"] for r in rows])
    initial_z = config["preinsert_xyz_m"][2]
    target_z = config["casing_xyz_m"][2]+config["nominal_relative_xyz_m"][2]
    target_xy = np.array(config["casing_xyz_m"][:2])+config["nominal_relative_xyz_m"][:2]
    force = np.linalg.norm([r["contact_force_world_N"] for r in rows], axis=1)
    penetration = np.array([max(0., -(r["min_separation_m"] or 0.)) for r in rows])*1000
    series = [("Insertion depth (mm)", (initial_z-state[:, 2])*1000, 45., False),
              ("Axial target error (mm)", (state[:, 2]-target_z)*1000, 45., False),
              ("Radial alignment error (mm)", np.linalg.norm(state[:, :2]-target_xy, axis=1)*1000, 1., False),
              ("PhysX speed (mm/s)", np.linalg.norm(state[:, 7:10], axis=1)*1000, 120., False),
              ("Contact API force (N, log scale)", force, 10000., True),
              ("Reported penetration (mm)", penetration, 1., False)]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        heading = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font = heading = ImageFont.load_default()
    duration = float(time[-1])
    frames = 0
    with imageio.get_writer(args.run/"telemetry_playback.mp4", fps=10, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=None) as writer:
        for now in np.arange(0., duration+.05, .1):
            index = min(int(np.searchsorted(time, now)), len(rows)-1)
            image = Image.new("RGB", (1280, 720), "#101823")
            draw = ImageDraw.Draw(image)
            draw.text((25, 15), "ISOLATED PHYSICS TEST - NO ROBOT", font=heading, fill="white")
            draw.text((25, 52), "TELEMETRY PLAYBACK, NOT A LIVE ISAAC SCENE RECORDING", font=font, fill="#ffce70")
            draw.text((25, 80), f"t={now:05.2f}s | actuator {'ON' if now < 26 else 'OFF'} | collisions ON | no runtime pose commands", font=font, fill="white")
            for panel, (name, values, limit, logarithmic) in enumerate(series):
                column, row = panel % 3, panel // 3
                x, y = 35+column*420, 145+row*270
                width, height = 375, 185
                draw.text((x, y-30), name, font=font, fill="white")
                draw.rectangle((x, y, x+width, y+height), outline="#708090")
                off_x = x+width*26/duration
                draw.line((off_x, y, off_x, y+height), fill="#a28652", width=1)
                # ponytail: display decimated to 20 Hz; full 240 Hz trace retains short spikes.
                sample = np.arange(0, index+1, 12)
                sample = np.unique(np.append(sample, index))
                scaled = np.log1p(np.maximum(values[sample], 0))/math.log1p(limit) if logarithmic else values[sample]/limit
                points = [(x+width*time[i]/duration, y+height*(1-np.clip(v, 0., 1.)))
                          for i, v in zip(sample, scaled)]
                if len(points) > 1:
                    draw.line(points, fill="#54d6b2", width=2)
                draw.text((x, y+height+8), f"0s          {values[index]:.4f}          32s", font=font, fill="#c9d4df")
            draw.text((25, 695), "Contact-force telemetry is uncalibrated; failure classification is in the diagnostic report.", font=font, fill="#ffce70")
            writer.append_data(np.asarray(image))
            frames += 1
        image.save(args.run/"telemetry_summary.png")
    metadata = {"type": "recorded_telemetry_playback_not_live_scene", "source": str(args.run/"trace.jsonl"),
                "source_rows": len(rows), "frames": frames, "fps": 10, "physics_duration_s": duration}
    (args.run/"telemetry_video.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
