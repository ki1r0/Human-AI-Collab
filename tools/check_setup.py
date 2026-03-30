#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation_logs"


@dataclass
class CheckResult:
    level: str
    title: str
    detail: str


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _mask(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "<empty>"
    if len(text) <= 6:
        return "*" * len(text)
    return f"{text[:2]}***{text[-4:]}"


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _status_line(result: CheckResult) -> str:
    return f"[{result.level}] {result.title}: {result.detail}"


def _append_command_result(results: List[CheckResult], title: str, cmd: Sequence[str], required: bool) -> subprocess.CompletedProcess[str]:
    completed = _run(cmd)
    output = completed.stdout.strip() or completed.stderr.strip() or f"exit={completed.returncode}"
    if completed.returncode == 0:
        results.append(CheckResult("PASS", title, output))
    else:
        level = "FAIL" if required else "WARN"
        results.append(CheckResult(level, title, f"{output} | cmd={_fmt_cmd(cmd)}"))
    return completed


def run_checks() -> List[CheckResult]:
    results: List[CheckResult] = []

    system_name = platform.system()
    if system_name == "Linux":
        results.append(CheckResult("PASS", "Host OS", platform.platform()))
    else:
        results.append(CheckResult("FAIL", "Host OS", f"Expected Linux host, found {platform.platform()}"))

    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        values = _parse_env_file(env_path)
        base_image = values.get("ISAACLAB_BASE_IMAGE", "").strip()
        commander_key = values.get("COMMANDER_API_KEY", "").strip()
        cosmos_url = values.get("COSMOS_CHAT_COMPLETIONS_URL", "").strip() or values.get("COSMOS_BASE_URL", "").strip()
        display = values.get("DISPLAY", "").strip()
        xauthority_host = values.get("XAUTHORITY_HOST_PATH", "").strip()

        if base_image:
            results.append(CheckResult("PASS", ".env ISAACLAB_BASE_IMAGE", base_image))
        else:
            results.append(CheckResult("FAIL", ".env ISAACLAB_BASE_IMAGE", "Missing required base image tag in .env"))

        if commander_key:
            results.append(CheckResult("PASS", ".env COMMANDER_API_KEY", _mask(commander_key)))
        else:
            results.append(CheckResult("FAIL", ".env COMMANDER_API_KEY", "Missing required commander key in .env"))

        if cosmos_url:
            results.append(CheckResult("PASS", ".env Cosmos endpoint", cosmos_url))
        else:
            results.append(CheckResult("WARN", ".env Cosmos endpoint", "Not configured; commander-only path will be used"))

        if display:
            results.append(CheckResult("PASS", ".env DISPLAY", display))
        else:
            results.append(CheckResult("WARN", ".env DISPLAY", "DISPLAY is empty in .env"))
        if xauthority_host:
            if Path(xauthority_host).exists():
                results.append(CheckResult("PASS", ".env XAUTHORITY_HOST_PATH", xauthority_host))
            else:
                results.append(CheckResult("WARN", ".env XAUTHORITY_HOST_PATH", f"Configured path does not exist: {xauthority_host}"))
        else:
            results.append(CheckResult("WARN", ".env XAUTHORITY_HOST_PATH", "Not set; X11 auth may rely on xhost"))
    else:
        results.append(CheckResult("FAIL", ".env file", "Missing .env; copy from .env.example first"))

    runtime_env_path = PROJECT_ROOT / "config" / "runtime_env.env"
    if runtime_env_path.is_file():
        results.append(CheckResult("PASS", "Runtime defaults", str(runtime_env_path)))
        runtime_values = _parse_env_file(runtime_env_path)
        tracked_secret_keys = []
        for key in ("COMMANDER_API_KEY", "COSMOS_API_KEY", "MEM0_API_KEY"):
            if runtime_values.get(key, "").strip():
                tracked_secret_keys.append(key)
        if tracked_secret_keys:
            joined = ", ".join(tracked_secret_keys)
            results.append(CheckResult("WARN", "Tracked runtime secrets", f"config/runtime_env.env contains non-empty secret-like keys: {joined}"))
    else:
        results.append(CheckResult("FAIL", "Runtime defaults", "Missing config/runtime_env.env"))

    local_runtime_path = PROJECT_ROOT / "config" / "runtime_env.local.env"
    if local_runtime_path.is_file():
        results.append(CheckResult("PASS", "Local runtime overrides", str(local_runtime_path)))
    else:
        results.append(CheckResult("WARN", "Local runtime overrides", "config/runtime_env.local.env not present"))

    _append_command_result(results, "Docker CLI", ["docker", "--version"], required=True)
    docker_compose = _append_command_result(results, "Docker Compose", ["docker", "compose", "version"], required=True)
    if docker_compose.returncode != 0:
        results.append(CheckResult("WARN", "Compose hint", "Install Docker Compose v2 or enable the compose plugin"))

    nvidia = _append_command_result(results, "NVIDIA driver", ["nvidia-smi"], required=False)
    if nvidia.returncode == 0:
        pass
    else:
        results.append(CheckResult("WARN", "GPU hint", "Isaac Lab requires a working NVIDIA driver stack on the host"))

    socket_path = Path("/var/run/docker.sock")
    if socket_path.exists():
        stat = socket_path.stat()
        results.append(
            CheckResult(
                "PASS",
                "Docker socket",
                f"mode={oct(stat.st_mode & 0o777)} uid={stat.st_uid} gid={stat.st_gid}",
            )
        )
    else:
        results.append(CheckResult("FAIL", "Docker socket", "/var/run/docker.sock is missing"))

    docker_ps = _append_command_result(results, "Docker daemon access", ["docker", "ps"], required=True)
    if docker_ps.returncode != 0:
        results.append(
            CheckResult(
                "WARN",
                "Docker daemon hint",
                "If this is a permissions issue, check your access to /var/run/docker.sock or use a shell with working Docker privileges.",
            )
        )

    if env_path.is_file() and docker_ps.returncode == 0:
        base_image = _parse_env_file(env_path).get("ISAACLAB_BASE_IMAGE", "").strip()
        if base_image:
            _append_command_result(results, "Configured base image", ["docker", "image", "inspect", base_image], required=False)

    display = os.getenv("DISPLAY", "").strip()
    x11_path = Path("/tmp/.X11-unix")
    if display:
        results.append(CheckResult("PASS", "Host DISPLAY", display))
    else:
        results.append(CheckResult("WARN", "Host DISPLAY", "DISPLAY is not set in the current shell"))
    if x11_path.exists():
        results.append(CheckResult("PASS", "X11 socket", str(x11_path)))
    else:
        results.append(CheckResult("WARN", "X11 socket", "/tmp/.X11-unix is missing"))

    accidental_venv = PROJECT_ROOT / "python=3.12"
    if accidental_venv.exists():
        results.append(CheckResult("WARN", "Accidental local venv", f"Found local artifact directory: {accidental_venv}"))

    return results


def summarize(results: Sequence[CheckResult]) -> int:
    failures = sum(1 for item in results if item.level == "FAIL")
    warnings = sum(1 for item in results if item.level == "WARN")
    print(f"Human-AI-Collab setup check for {PROJECT_ROOT}")
    print(f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()
    for item in results:
        print(_status_line(item))
    print()
    print(f"Summary: {failures} fail, {warnings} warn, {len(results) - failures - warnings} pass")
    return 1 if failures else 0


def maybe_write_log(results: Sequence[CheckResult], enabled: bool) -> Path | None:
    if not enabled:
        return None
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    log_path = VALIDATION_DIR / f"setup_check_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [
        f"Human-AI-Collab setup check for {PROJECT_ROOT}",
        f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
    ]
    lines.extend(_status_line(item) for item in results)
    lines.append("")
    failures = sum(1 for item in results if item.level == "FAIL")
    warnings = sum(1 for item in results if item.level == "WARN")
    passes = len(results) - failures - warnings
    lines.append(f"Summary: {failures} fail, {warnings} warn, {passes} pass")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local setup for Human-AI-Collab")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Print results only and do not write validation_logs/setup_check_*.log",
    )
    args = parser.parse_args()

    results = run_checks()
    exit_code = summarize(results)
    log_path = maybe_write_log(results, enabled=not args.no_log)
    if log_path is not None:
        print(f"Log written to {log_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
