"""Draw simple, unlabeled task-layout images for LingBot-VA interface smoke tests.

These are not benchmark renders and must never be used as a task success score. They
only let us exercise the official model input contract before Isaac camera rendering is
available. The final benchmark inputs must be rendered from the generated USD scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


W, H = 320, 224
NAMES = (
    "observation.images.cam_high.png",
    "observation.images.cam_left_wrist.png",
    "observation.images.cam_right_wrist.png",
)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), (45, 48, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, W - 8, H - 8), fill=(108, 91, 71), outline=(190, 170, 135), width=2)
    return image, draw


def _draw_feature(draw: ImageDraw.ImageDraw, task_id: str, variant: str, cx: int, cy: int) -> None:
    dark = (36, 40, 44)
    metal = (176, 185, 190)
    accent = (86, 135, 164)
    if task_id == "HCF-01":
        draw.rectangle((cx - 42, cy - 32, cx + 42, cy + 32), fill=metal, outline=(35, 40, 45), width=3)
        draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=dark)
        if variant == "COMMUTABLE":
            draw.rectangle((cx - 7, cy - 32, cx + 7, cy - 16), fill=dark)
            draw.rectangle((cx + 16, cy - 7, cx + 42, cy + 7), fill=dark)
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), outline=accent, width=2)
    elif task_id == "WSG-01":
        draw.ellipse((cx - 62, cy - 62, cx + 62, cy + 62), fill=metal, outline=(35, 40, 45), width=3)
        draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=dark)
        if variant == "COMMUTABLE":
            draw.rectangle((cx + 16, cy - 7, cx + 62, cy + 7), fill=dark)
        draw.ellipse((cx - 35, cy - 35, cx + 35, cy + 35), outline=accent, width=2)
    elif task_id == "KEY-01":
        draw.rectangle((cx - 58, cy - 34, cx + 58, cy + 34), fill=metal, outline=(35, 40, 45), width=3)
        if variant == "HARD":
            draw.rectangle((cx - 18, cy - 5, cx + 18, cy + 5), fill=dark)
        else:
            draw.rectangle((cx - 18, cy - 30, cx + 18, cy + 30), fill=dark)
        draw.line((cx - 58, cy, cx + 58, cy), fill=accent, width=2)
    elif task_id == "CAS-01":
        draw.rectangle((cx - 62, cy - 38, cx + 62, cy + 38), fill=metal, outline=(35, 40, 45), width=3)
        draw.rectangle((cx - 13, cy - 13, cx + 13, cy + 13), fill=dark)
        if variant == "COMMUTABLE":
            draw.rectangle((cx - 30, cy - 9, cx - 13, cy + 9), fill=dark)
            draw.rectangle((cx + 13, cy - 9, cx + 30, cy + 9), fill=dark)
        draw.line((cx - 62, cy, cx + 62, cy), fill=accent, width=2)
    elif task_id == "DOW-01":
        draw.rectangle((cx - 62, cy - 38, cx + 62, cy + 38), fill=metal, outline=(35, 40, 45), width=3)
        if variant == "COMMUTABLE":
            draw.rectangle((cx - 22, cy - 17, cx + 22, cy + 17), fill=dark)
        else:
            draw.rectangle((cx - 8, cy - 17, cx + 8, cy + 17), fill=dark)
        draw.line((cx - 62, cy, cx + 62, cy), fill=accent, width=2)


def make_images(task_id: str, variant: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera_index, filename in enumerate(NAMES):
        image, draw = _canvas()
        if camera_index == 0:
            _draw_feature(draw, task_id, variant, 155, 108)
            draw.ellipse((245, 25, 275, 55), fill=(204, 196, 178), outline=(25, 25, 25), width=2)
        elif camera_index == 1:
            _draw_feature(draw, task_id, variant, 142, 112)
            draw.rectangle((245, 130, 290, 170), fill=(210, 170, 80), outline=(30, 30, 30), width=2)
        else:
            _draw_feature(draw, task_id, variant, 177, 100)
            draw.ellipse((38, 150, 84, 196), fill=(210, 170, 80), outline=(30, 30, 30), width=2)
        image.save(out_dir / filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant", choices=("HARD", "COMMUTABLE"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    make_images(args.task_id, args.variant, args.out_dir)
    print(f"wrote synthetic observation smoke images to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
