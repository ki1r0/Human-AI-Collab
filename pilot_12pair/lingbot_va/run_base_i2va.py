"""Run one official LingBot-VA base image-to-video-action smoke episode.

The script mutates only the imported official config in memory.  The upstream
checkout and checkpoint remain outside this repository; this file is the
reproducible adapter and provenance record for the experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


EXPECTED_IMAGES = (
    "observation.images.cam_high.png",
    "observation.images.cam_left_wrist.png",
    "observation.images.cam_right_wrist.png",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--save-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--video-steps", type=int, default=5)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"LingBot-VA checkpoint directory not found: {args.model_path}")
    missing = [name for name in EXPECTED_IMAGES if not (args.input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"input directory missing required images: {missing}")
    if args.num_chunks < 1 or args.video_steps < 1 or args.action_steps < 1:
        raise ValueError("num-chunks and inference steps must be positive")


def main() -> int:
    args = _parse_args()
    _validate(args)
    args.save_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "robbyant/lingbot-va-base",
        "checkpoint_path": str(args.model_path),
        "official_repo": os.environ.get("LINGBOT_VA_REPO", ""),
        "input_dir": str(args.input_dir),
        "prompt": args.prompt,
        "num_chunks": args.num_chunks,
        "video_inference_steps": args.video_steps,
        "action_inference_steps": args.action_steps,
        "enable_offload": True,
        "native_action_layout": "Franka config: selected Cartesian EEF/gripper channels in 30-dim layout",
        "note": "Base-model interface smoke only; no custom task adaptation or completion score.",
    }
    (args.save_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # The upstream module imports flash-attn even when its inference path uses
    # PyTorch SDPA.  Keep the official `attn_mode="torch"` path usable on a
    # clean environment without compiling an unused flash-attn extension.
    try:
        import flash_attn  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("flash_attn")
        def _unused_flash_attn(*_args, **_kwargs):
            raise RuntimeError("flash-attn stub called; inference must use attn_mode='torch'")
        stub.flash_attn_func = _unused_flash_attn
        sys.modules["flash_attn"] = stub

    # The official server requires torch.distributed even for one process.  This
    # entry point is therefore invoked under torch.distributed.run by the shell
    # launcher; importing the upstream config here lets us avoid editing it.
    from wan_va.configs import VA_CONFIGS  # type: ignore
    from wan_va.wan_va_server import init_logger, run  # type: ignore

    config = VA_CONFIGS["franka_i2va"]
    config.wan22_pretrained_model_name_or_path = str(args.model_path)
    config.input_img_path = str(args.input_dir)
    config.save_root = str(args.save_root)
    config.prompt = args.prompt
    config.num_chunks_to_infer = args.num_chunks
    config.num_inference_steps = args.video_steps
    config.action_num_inference_steps = args.action_steps
    config.enable_offload = True
    config.infer_mode = "i2va"

    class RunArgs:
        config_name = "franka_i2va"
        port = None
        save_root = str(args.save_root)

    init_logger()
    run(RunArgs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
