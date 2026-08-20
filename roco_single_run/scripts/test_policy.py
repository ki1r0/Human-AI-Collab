#!/usr/bin/env python3
"""Stage D: load the real ACT checkpoint and infer on a saved live observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from roco_policy import CAMERA_NAMES, POLICY_CONFIG, RocoActPolicy, policy_to_environment_order


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--observation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument(
    "--expected-checkpoint-sha256",
    default="a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1",
)
parser.add_argument(
    "--expected-stats-sha256",
    default="4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e",
)
parser.add_argument("--candidate-id", default="yjsm1203/roco_model_act_2")
parser.add_argument("--candidate-revision", default="52344a203e0739638cb2c7b11ea632e7b2eb2608")
args = parser.parse_args()


def _summary(value: torch.Tensor) -> dict:
    tensor = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
    }


def _values(value: torch.Tensor) -> list:
    return value.detach().float().cpu().tolist()


def main() -> None:
    result = {
        "stage": "D",
        "classification": "confirmatory",
        "status": "RUNNING",
        "checkpoint": str(args.checkpoint.resolve()),
        "stats": str(args.stats.resolve()),
        "observation": str(args.observation.resolve()),
        "device": args.device,
        "camera_order": list(CAMERA_NAMES),
        "policy_config": POLICY_CONFIG,
        "temporal_decay": 0.1,
        "candidate_id": args.candidate_id,
        "candidate_revision": args.candidate_revision,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "expected_stats_sha256": args.expected_stats_sha256,
    }
    try:
        with np.load(args.observation) as package:
            images = np.stack([package[name] for name in CAMERA_NAMES], axis=1)
            qpos = package["qpos"].copy()

        adapter = RocoActPolicy(
            args.checkpoint,
            args.stats,
            device=args.device,
            temporal_decay=0.1,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_stats_sha256=args.expected_stats_sha256,
        )
        first = adapter.predict_trace(qpos, images)
        second = adapter.predict_trace(qpos, images)
        second_timestep = adapter.timestep
        adapter.reset()
        reset_state = {"timestep": adapter.timestep, "chunk_count": len(adapter._chunks)}
        after_reset = adapter.predict_trace(qpos, images)

        first_action = first["denormalized_policy_action"]
        second_action = second["denormalized_policy_action"]
        reset_action = after_reset["denormalized_policy_action"]
        arms = torch.cat((first_action[:, :6], first_action[:, 7:13]), dim=-1)
        grippers = torch.cat((first_action[:, 6:7], first_action[:, 13:14]), dim=-1)
        reorder_sentinel = policy_to_environment_order(
            torch.arange(14, device=adapter.device).reshape(1, 14)
        )

        tensor_fields = (
            "normalized_qpos",
            "normalized_images",
            "raw_chunk",
            "aggregated_normalized_action",
            "denormalized_policy_action",
            "environment_action",
        )
        all_traces_finite = all(
            bool(torch.isfinite(trace[field]).all().item())
            for trace in (first, second, after_reset)
            for field in tensor_fields
        )
        checks = {
            "strict_state_dict_load": not adapter.missing_keys and not adapter.unexpected_keys,
            "normalization_shapes_14": all(
                value.shape == (14,) for value in adapter.stats_numpy.values()
            ),
            "normalization_std_positive": bool(
                np.all(adapter.stats_numpy["qpos_std"] > 0)
                and np.all(adapter.stats_numpy["action_std"] > 0)
            ),
            "actual_roco_observation_shape": qpos.shape == (1, 14)
            and images.shape == (1, 3, 240, 320, 3),
            "rgb_scaled_once_to_0_1": float(first["normalized_images"].min()) >= 0.0
            and float(first["normalized_images"].max()) <= 1.0,
            "normalized_qpos_finite": bool(torch.isfinite(first["normalized_qpos"]).all()),
            "raw_chunk_shape_1x100x14": first["raw_chunk"].shape == (1, 100, 14),
            "all_trace_tensors_finite": all_traces_finite,
            "temporal_aggregation_advances": second_timestep == 2
            and second["aggregation_weights"].numel() == 2,
            "temporal_aggregation_changes_action": not torch.allclose(
                first_action, second_action, atol=1e-7, rtol=0
            ),
            "reset_clears_temporal_state": reset_state == {"timestep": 0, "chunk_count": 0},
            "reset_reproduces_first_action": torch.allclose(
                first_action, reset_action, atol=1e-6, rtol=1e-6
            ),
            "exact_reorder_sentinel": reorder_sentinel.cpu().tolist()
            == [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]],
            "broad_physical_action_bounds": bool(
                torch.all(torch.abs(arms) <= 2 * torch.pi)
                and torch.all(grippers >= -0.02)
                and torch.all(grippers <= 0.08)
            ),
        }
        result.update(
            {
                "checkpoint_sha256": adapter.checkpoint_sha256,
                "stats_sha256": adapter.stats_sha256,
                "checkpoint_tensor_count": adapter.checkpoint_tensor_count,
                "parameter_count": adapter.parameter_count,
                "missing_keys": adapter.missing_keys,
                "unexpected_keys": adapter.unexpected_keys,
                "stats_summaries": {
                    key: {
                        "shape": list(value.shape),
                        "min": float(value.min()),
                        "max": float(value.max()),
                        "mean": float(value.mean()),
                    }
                    for key, value in adapter.stats_numpy.items()
                },
                "input_qpos_raw": qpos.tolist(),
                "input_images_raw": {
                    "shape": list(images.shape),
                    "dtype": str(images.dtype),
                    "min": int(images.min()),
                    "max": int(images.max()),
                },
                "normalized_qpos": _values(first["normalized_qpos"]),
                "normalized_qpos_summary": _summary(first["normalized_qpos"]),
                "normalized_images_summary": _summary(first["normalized_images"]),
                "raw_chunk_summary": _summary(first["raw_chunk"]),
                "raw_network_first_action_normalized": _values(first["raw_chunk"][:, 0, :]),
                "raw_network_second_action_normalized": _values(first["raw_chunk"][:, 1, :]),
                "first_aggregated_action_normalized": _values(
                    first["aggregated_normalized_action"]
                ),
                "second_aggregation_weights_oldest_to_newest": _values(
                    second["aggregation_weights"]
                ),
                "second_aggregated_action_normalized": _values(
                    second["aggregated_normalized_action"]
                ),
                "first_denormalized_policy_action": _values(first_action),
                "first_environment_ordered_action": _values(first["environment_action"]),
                "second_denormalized_policy_action": _values(second_action),
                "reset_denormalized_policy_action": _values(reset_action),
                "reorder_sentinel": _values(reorder_sentinel),
                "checks": checks,
            }
        )
        result["status"] = "PASS" if all(checks.values()) else "FAIL"
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("ROCO_STAGE_D_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
