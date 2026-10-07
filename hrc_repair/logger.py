"""Append-only M0 artifact logger with public/GT separation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import stable_hash


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"api_key", "token", "secret", "password", "gt", "ground_truth", "scenario", "scenario_private", "fault_profile", "hub_pose", "casing_pose", "penetration"}:
                result[str(key)] = "<PRIVATE>"
            else:
                result[str(key)] = _public(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value]
    return _jsonable(value)


class RepairLogger:
    def __init__(self, out_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._files = {
            "events": (self.out_dir / "events.jsonl").open("a", encoding="utf-8"),
            "observations": (self.out_dir / "public_observations.jsonl").open("a", encoding="utf-8"),
            "model_calls": (self.out_dir / "model_calls.jsonl").open("a", encoding="utf-8"),
            "gt": (self.out_dir / "gt_trajectory.jsonl").open("a", encoding="utf-8"),
            "trajectory": (self.out_dir / "robot_trajectory.jsonl").open("a", encoding="utf-8"),
        }
        os.chmod(self._files["gt"].fileno(), 0o600)
        self._index = 0
        self.manifest = _public(manifest)
        self.manifest.setdefault("logger", {}).update({"format": "repair-m0-jsonl-v1", "public_events": "events.jsonl", "private_gt": "gt_trajectory.jsonl"})
        (self.out_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def event(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {"event_index": self._index, "time": time.time(), "kind": str(kind), "payload": _public(payload or {})}
        self._index += 1
        self._write("events", record)

    def observation(self, observation: Any) -> None:
        self._write("observations", _public(observation))

    def model_call(self, payload: Mapping[str, Any]) -> None:
        self._write("model_calls", _public(payload))

    def trajectory(self, payload: Mapping[str, Any]) -> None:
        self._write("trajectory", _public(payload))

    def gt(self, payload: Mapping[str, Any]) -> None:
        self._write("gt", _jsonable(payload))

    def metrics(self, payload: Mapping[str, Any]) -> None:
        value = _public(payload)
        value["metrics_hash"] = stable_hash(value)
        (self.out_dir / "metrics.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write(self, name: str, value: Any) -> None:
        handle = self._files[name]
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()

    def close(self) -> None:
        for handle in self._files.values():
            if not handle.closed:
                handle.flush()
                handle.close()

    def __enter__(self) -> "RepairLogger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["RepairLogger"]
