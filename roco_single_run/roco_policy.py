"""Training-faithful adapter around the official RoCo ACT implementation."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from Galaxea_Lab_External.VLA.ACT.act.policy import ACTPolicy


CHECKPOINT_SHA256 = "a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1"
STATS_SHA256 = "4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e"
CAMERA_NAMES = ("head_rgb", "left_hand_rgb", "right_hand_rgb")
POLICY_CONFIG = {
    "num_queries": 100,
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": list(CAMERA_NAMES),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_to_environment_order(action: torch.Tensor) -> torch.Tensor:
    """Map [L6,Lgrip,R6,Rgrip] to [L6,R6,Lgrip,Rgrip]."""
    if action.shape[-1] != 14:
        raise ValueError(f"Expected 14-D policy action, got {tuple(action.shape)}")
    return torch.cat(
        (action[..., :6], action[..., 7:13], action[..., 6:7], action[..., 13:14]),
        dim=-1,
    )


class RocoActPolicy:
    """Strict checkpoint loader and stateful ACT inference adapter."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        stats_path: str | Path,
        *,
        device: str = "cuda:0",
        temporal_decay: float = 0.1,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.stats_path = Path(stats_path).resolve()
        self.device = torch.device(device)
        self.temporal_decay = float(temporal_decay)
        self.num_queries = int(POLICY_CONFIG["num_queries"])

        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.stats_sha256 = file_sha256(self.stats_path)
        if self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Checkpoint hash mismatch: {self.checkpoint_sha256} != {CHECKPOINT_SHA256}"
            )
        if self.stats_sha256 != STATS_SHA256:
            raise RuntimeError(f"Stats hash mismatch: {self.stats_sha256} != {STATS_SHA256}")

        with self.stats_path.open("rb") as stream:
            stats = pickle.load(stream)
        required = {"qpos_mean", "qpos_std", "action_mean", "action_std"}
        missing = required - set(stats)
        if missing:
            raise RuntimeError(f"Missing normalization arrays: {sorted(missing)}")
        self.stats_numpy = {
            key: np.asarray(stats[key], dtype=np.float32).copy() for key in sorted(required)
        }
        for key, value in self.stats_numpy.items():
            if value.shape != (14,) or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid {key}: shape={value.shape}, finite={np.isfinite(value).all()}")
        if np.any(self.stats_numpy["qpos_std"] <= 0) or np.any(
            self.stats_numpy["action_std"] <= 0
        ):
            raise RuntimeError("Normalization standard deviations must be positive")

        self.qpos_mean = torch.from_numpy(self.stats_numpy["qpos_mean"]).to(self.device)
        self.qpos_std = torch.from_numpy(self.stats_numpy["qpos_std"]).to(self.device)
        self.action_mean = torch.from_numpy(self.stats_numpy["action_mean"]).to(self.device)
        self.action_std = torch.from_numpy(self.stats_numpy["action_std"]).to(self.device)

        self.policy = ACTPolicy(dict(POLICY_CONFIG))
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise RuntimeError("Checkpoint is not a nonempty state dictionary")
        load_status = self.policy.load_state_dict(checkpoint, strict=True)
        self.missing_keys = list(load_status.missing_keys)
        self.unexpected_keys = list(load_status.unexpected_keys)
        self.checkpoint_tensor_count = len(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        self.reset()

    def reset(self) -> None:
        self.timestep = 0
        self._chunks: list[tuple[int, torch.Tensor]] = []

    def preprocess(
        self, qpos: torch.Tensor | np.ndarray, images: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos_tensor = torch.as_tensor(qpos, device=self.device)
        image_tensor = torch.as_tensor(images, device=self.device)
        if qpos_tensor.shape != (1, 14):
            raise ValueError(f"Expected qpos (1,14), got {tuple(qpos_tensor.shape)}")
        if image_tensor.shape == (1, 3, 240, 320, 3):
            image_tensor = image_tensor.permute(0, 1, 4, 2, 3)
        if image_tensor.shape != (1, 3, 3, 240, 320):
            raise ValueError(
                "Expected images (1,3,240,320,3) or (1,3,3,240,320), "
                f"got {tuple(image_tensor.shape)}"
            )
        if image_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected raw uint8 RGB, got {image_tensor.dtype}")
        image_tensor = image_tensor.to(torch.float32) / 255.0
        qpos_tensor = qpos_tensor.to(torch.float32)
        normalized_qpos = (qpos_tensor - self.qpos_mean) / self.qpos_std
        if not torch.isfinite(normalized_qpos).all() or not torch.isfinite(image_tensor).all():
            raise RuntimeError("Non-finite preprocessed policy input")
        return normalized_qpos, image_tensor

    def _aggregate(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._chunks.append((self.timestep, chunk.detach()))
        current = []
        for start, prior_chunk in self._chunks:
            offset = self.timestep - start
            if 0 <= offset < self.num_queries:
                current.append(prior_chunk[:, offset, :])
        actions = torch.cat(current, dim=0)
        weights = torch.exp(
            -self.temporal_decay
            * torch.arange(len(current), device=self.device, dtype=torch.float32).flip(0)
        )
        weights = weights / weights.sum()
        aggregated = (actions.T @ weights).unsqueeze(0)
        return aggregated, weights

    def predict_trace(
        self, qpos: torch.Tensor | np.ndarray, images: torch.Tensor | np.ndarray
    ) -> dict[str, torch.Tensor | int]:
        normalized_qpos, normalized_images = self.preprocess(qpos, images)
        with torch.inference_mode():
            raw_chunk = self.policy(normalized_qpos, normalized_images)
            if raw_chunk.shape != (1, self.num_queries, 14):
                raise RuntimeError(f"Unexpected ACT output shape: {tuple(raw_chunk.shape)}")
            if not torch.isfinite(raw_chunk).all():
                raise RuntimeError("ACT produced non-finite output")
            aggregated, weights = self._aggregate(raw_chunk)
            denormalized = aggregated * self.action_std + self.action_mean
            environment_action = policy_to_environment_order(denormalized)
        trace = {
            "timestep": self.timestep,
            "normalized_qpos": normalized_qpos,
            "normalized_images": normalized_images,
            "raw_chunk": raw_chunk,
            "aggregated_normalized_action": aggregated,
            "aggregation_weights": weights,
            "denormalized_policy_action": denormalized,
            "environment_action": environment_action,
        }
        self.timestep += 1
        return trace
