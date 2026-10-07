"""VLM verification boundary with an explicit UNKNOWN outcome."""

from __future__ import annotations

import json
import os
import base64
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import Observation


class VLMVerdict(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Verification:
    verdict: VLMVerdict
    evidence: str
    response_hash: str | None = None
    prompt_hash: str | None = None


class ObserverVerifier:
    def __init__(self, endpoint: str | None = None, model: str | None = None, *, api_key_env: str = "HRC_M1_VLM_API_KEY", timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)

    def verify(self, observation: Observation, *, expected_outcome: str | None = None) -> Verification:
        if not self.endpoint or not self.model:
            return Verification(VLMVerdict.UNKNOWN, "VLM endpoint is not configured")
        key = os.environ.get(self.api_key_env)
        if not key:
            return Verification(VLMVerdict.UNKNOWN, f"VLM credential {self.api_key_env} is not configured")
        if not observation.frames:
            return Verification(VLMVerdict.UNKNOWN, "no RGB frames are available for visual verification")
        instruction = (
            "Compare the current RGB images with the expected outcome below. "
            "Use only visible evidence; do not infer hidden physics. If the outcome "
            "is ambiguous or occluded, use UNKNOWN. Return SUCCESS only when every "
            "condition in the expected outcome has direct visual evidence; proximity "
            "alone does not prove contact, grasp, seating, or release. Return JSON with keys verdict "
            "(SUCCESS, FAILED, or UNKNOWN) and assessment (one concise sentence).\n"
            f"Expected outcome: {expected_outcome}"
            if expected_outcome
            else "Classify the visible assembly state as SUCCESS, FAILED, or UNKNOWN. Do not infer hidden physics."
        )
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "text", "text": instruction}, *[{"type": "image_url", "image_url": {"url": _frame_data_uri(frame)}} for frame in observation.frames.values()]]}]}
        prompt_hash = _hash_response(payload)
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_bytes = response.read()
            raw = json.loads(raw_bytes.decode("utf-8"))
            content = str(raw["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
            return Verification(VLMVerdict.UNKNOWN, "VLM request or response failed")
        verdict, evidence = _parse_assessment(content)
        import hashlib
        return Verification(verdict, evidence, hashlib.sha256(raw_bytes).hexdigest(), prompt_hash)


def _parse_assessment(content: str) -> tuple[VLMVerdict, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        raw_verdict = str(decoded.get("verdict", "")).upper()
        evidence = str(decoded.get("assessment", "")).strip() or text
    else:
        match = re.search(r"\b(SUCCESS|FAILED|UNKNOWN)\b", text.upper())
        raw_verdict = match.group(1) if match else "UNKNOWN"
        evidence = text
    try:
        verdict = VLMVerdict(raw_verdict)
    except ValueError:
        verdict = VLMVerdict.UNKNOWN
    return verdict, evidence[:1600]


def _frame_data_uri(value: str) -> str:
    if value.startswith("data:image/"):
        return value
    path = Path(value)
    if path.is_file():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
        return f"data:{mime};base64,{encoded}"
    return value


def _hash_response(value: Any) -> str:
    return __import__("hashlib").sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


__all__ = ["ObserverVerifier", "Verification", "VLMVerdict"]
