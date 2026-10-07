"""Small append-only run logger for M1.

The logger deliberately keeps evaluator-only data in a separate file.  Public
events are JSONL so an interrupted Kit process still leaves a useful trace.
It is not a replacement for a database or experiment tracker.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import stable_hash


_SECRET_WORDS = ("api_key", "token", "secret", "password", "authorization")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float, bool)):
        return _jsonable(value.value)
    return value


def _public(value: Any) -> Any:
    """Redact likely credentials before a value reaches events.jsonl."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).lower()
            if any(word in key_text for word in _SECRET_WORDS):
                result[str(key)] = "<REDACTED>"
            else:
                result[str(key)] = _public(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_public(child) for child in value]
    return _jsonable(value)


class EventLogger:
    """Write a reproducible manifest, public events and private evaluator data."""

    def __init__(self, out_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "frames").mkdir(exist_ok=True)
        self._events = (self.out_dir / "events.jsonl").open("a", encoding="utf-8")
        self._private = (self.out_dir / "evaluator_private.jsonl").open("a", encoding="utf-8")
        os.chmod(self._private.fileno(), 0o600)
        self._event_index = 0
        public_manifest = _public(manifest)
        public_manifest.setdefault("logger", {})
        public_manifest["logger"].update({"format": "m1-jsonl-v1", "events": "events.jsonl"})
        self.manifest = public_manifest
        (self.out_dir / "manifest.json").write_text(
            json.dumps(public_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def event(self, kind: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "event_index": self._event_index,
            "utc_time": time.time(),
            "monotonic_time": time.monotonic(),
            "kind": str(kind),
            "payload": _public(payload or {}),
        }
        self._event_index += 1
        self._events.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        self._events.flush()
        return record

    def evaluator_event(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {
            "event_index": self._event_index,
            "utc_time": time.time(),
            "kind": str(kind),
            "payload": _jsonable(payload or {}),
        }
        self._private.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        self._private.flush()

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        public_metrics = _public(metrics)
        public_metrics["metrics_hash"] = stable_hash(public_metrics)
        (self.out_dir / "metrics.json").write_text(
            json.dumps(public_metrics, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        if not self._events.closed:
            self._events.flush()
            self._events.close()
        if not self._private.closed:
            self._private.flush()
            self._private.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["EventLogger"]
