#!/usr/bin/env python3
"""Compare RoCo ACT preprocessing paths on one saved live observation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from roco_policy import POLICY_CONFIG, RocoActPolicy
from act.policy import ACTPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.observation) as package:
        images = np.stack(
            [package[name] for name in ("head_rgb", "left_hand_rgb", "right_hand_rgb")],
            axis=1,
        )
        qpos = package["qpos"].copy()
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    with args.stats.open("rb") as stream:
        stats = pickle.load(stream)

    device = torch.device(args.device)
    model = ACTPolicy(dict(POLICY_CONFIG)).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    q_raw = torch.from_numpy(qpos).to(device=device, dtype=torch.float32)
    image_raw = torch.from_numpy(images).to(device).permute(0, 1, 4, 2, 3)
    image_01 = image_raw.to(torch.float32) / 255.0
    q_norm = (
        q_raw - torch.from_numpy(stats["qpos_mean"]).to(device=device, dtype=torch.float32)
    ) / torch.from_numpy(stats["qpos_std"]).to(device=device, dtype=torch.float32)

    def infer(label: str, q: torch.Tensor, image: torch.Tensor) -> None:
        record: dict[str, object] = {"case": label}
        try:
            with torch.inference_mode():
                output = model(q, image)
            record.update(
                {
                    "shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "min": float(output.min().item()),
                    "max": float(output.max().item()),
                    "mean": float(output.mean().item()),
                    "std": float(output.std().item()),
                    "first_action": output[0, 0].detach().cpu().tolist(),
                }
            )
        except Exception as exc:  # diagnostic comparison should show the failure mode
            record["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(record, sort_keys=True))

    infer("official_VLA_raw_qpos_raw_rgb", q_raw, image_raw)
    infer("official_deploy_raw_qpos_rgb_0_1", q_raw, image_01)
    infer("training_faithful_norm_qpos_rgb_0_1", q_norm, image_01)

    adapter = RocoActPolicy(
        args.checkpoint,
        args.stats,
        device=args.device,
        expected_checkpoint_sha256="a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1",
        expected_stats_sha256="4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e",
    )
    trace = adapter.predict_trace(qpos, images)
    print(
        json.dumps(
            {
                "case": "current_RocoActPolicy",
                "normalized_qpos": trace["normalized_qpos"][0].detach().cpu().tolist(),
                "raw_first_action": trace["raw_chunk"][0, 0].detach().cpu().tolist(),
                "denormalized_action": trace["denormalized_policy_action"][0]
                .detach()
                .cpu()
                .tolist(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
