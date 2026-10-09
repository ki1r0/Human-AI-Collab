#!/usr/bin/env python3
"""Run the existing pick-only smoke and preserve native signed contact tensors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_REPO = Path("/workspace/Human-AI-Collab")
IMAGE = "human-ai-collab:latest"
sys.path.insert(0, str(REPO_ROOT))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _py(value: Any) -> Any:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [_py(item) for item in value]
    if callable(getattr(value, "item", None)):
        value = value.item()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finalize_existing_capture(output_dir: Path, docker_exit_code: int | None = None) -> int:
    raw_path = output_dir / "native_contact_force_samples.jsonl"
    runner_summary_path = output_dir / "pick_smoke/smoke_summary.json"
    capture_path = output_dir / "force_matrix_capture.json"
    launch_path = output_dir / "launch_summary.json"
    runner = _read_json(runner_summary_path)
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with raw_path.open(encoding="utf-8") as raw_file:
            for line_number, line in enumerate(raw_file, 1):
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("sample is not a JSON object")
                    samples.append(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"raw line {line_number}: {type(exc).__name__}: {exc}")
    except OSError as exc:
        errors.append(f"raw trace unavailable: {type(exc).__name__}: {exc}")

    finger_stats: dict[str, dict[str, Any]] = {
        side: {
            "contact_samples": 0,
            "contact_points": 0,
            "negative_scalar_points": 0,
            "positive_scalar_points": 0,
            "zero_scalar_points": 0,
            "scalar_min_n": None,
            "scalar_max_n": None,
            "max_point_sum_to_pair_matrix_residual_n": 0.0,
        }
        for side in ("right", "left")
    }
    net_residuals: list[float] = []
    bilateral_samples = 0
    required_keys = {
        "raw_pair_force_matrix_w_n",
        "raw_net_contact_forces_on_sensor_w_n",
        "filters",
        "pair_force_matrix_sum_w_n",
        "net_minus_pair_matrix_sum_w_n",
    }

    for sample_index, sample in enumerate(samples):
        if not required_keys.issubset(sample):
            errors.append(f"sample {sample_index} is missing native force fields")
            continue
        try:
            residual = sample["net_minus_pair_matrix_sum_w_n"]
            net_residuals.append(_norm([float(value) for value in residual]))
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"sample {sample_index} has invalid net/pair residual: {exc}")
        active_sides = set()
        for item in sample["filters"]:
            path = str(item.get("filter_prim_path", ""))
            side = "right" if "panda_rightfinger" in path else "left" if "panda_leftfinger" in path else None
            if side is None:
                continue
            stats = finger_stats[side]
            contacts = item.get("raw_contacts", [])
            if int(item.get("reported_count", 0)) > 0:
                stats["contact_samples"] += 1
                active_sides.add(side)
            try:
                pair_residual = float(item["pair_minus_point_sum_norm_n"])
                stats["max_point_sum_to_pair_matrix_residual_n"] = max(
                    stats["max_point_sum_to_pair_matrix_residual_n"], pair_residual
                )
            except (TypeError, ValueError, KeyError):
                errors.append(f"sample {sample_index} missing {side} pair/point residual")
            for contact in contacts:
                stats["contact_points"] += 1
                force_raw = contact.get("force_raw", [])
                try:
                    scalar = float(force_raw[0])
                except (TypeError, ValueError, IndexError):
                    errors.append(f"sample {sample_index} has invalid {side} force scalar")
                    continue
                if scalar < 0:
                    stats["negative_scalar_points"] += 1
                elif scalar > 0:
                    stats["positive_scalar_points"] += 1
                else:
                    stats["zero_scalar_points"] += 1
                stats["scalar_min_n"] = scalar if stats["scalar_min_n"] is None else min(
                    stats["scalar_min_n"], scalar
                )
                stats["scalar_max_n"] = scalar if stats["scalar_max_n"] is None else max(
                    stats["scalar_max_n"], scalar
                )
        if active_sides == {"right", "left"}:
            bilateral_samples += 1

    buffer_summary = runner.get("private_bolt_contact_buffer", {})
    runner_sample_count = runner.get("private_bolt_contact_samples")
    if isinstance(runner_sample_count, list):
        runner_sample_count = len(runner_sample_count)
    if isinstance(runner_sample_count, int) and runner_sample_count != len(samples):
        errors.append(
            f"runner/raw sample count mismatch: {runner_sample_count} vs {len(samples)}"
        )
    if buffer_summary.get("capacity_reached_or_truncation_possible"):
        errors.append("runner reports contact-buffer saturation or unknown counts")
    if int(buffer_summary.get("raw_report_validation_error_count", 0)) != 0:
        errors.append("runner reports invalid raw contact records")

    capture_complete = bool(samples) and not errors
    capture_summary = {
        "run_kind": "bolt_pick_signed_contact_force_capture_only",
        "capture_scope": "existing pick stage and transport; not insertion/task-success evaluation",
        "raw_samples_path": str(raw_path),
        "runner_summary_path": str(runner_summary_path),
        "raw_sample_count": len(samples),
        "runner_private_sample_count": runner_sample_count,
        "raw_capture_complete": capture_complete,
        "capture_errors": errors,
        "bilateral_filter_contact_samples": bilateral_samples,
        "finger_signed_scalar_and_pair_matrix_stats": finger_stats,
        "net_minus_filtered_pair_matrix_residual_norm_n": {
            "min": min(net_residuals) if net_residuals else None,
            "median": sorted(net_residuals)[len(net_residuals) // 2] if net_residuals else None,
            "max": max(net_residuals) if net_residuals else None,
            "exact_zero_samples": sum(value == 0.0 for value in net_residuals),
        },
        "runner_smoke_passed": runner.get("smoke_passed"),
        "pick_result": runner.get("pick_result"),
        "transport_result": runner.get("transport_result"),
        "task_success": None,
        "final_success_claimed": False,
        "task_success_claimed": False,
    }
    _write_json(capture_path, capture_summary)

    smoke_passed = runner.get("smoke_passed")
    runner_result_code = None if smoke_passed is None else (0 if smoke_passed else 2)
    capture_exit_code = 0 if capture_complete else 1
    wrapper_exit_code = capture_exit_code if not capture_complete else (runner_result_code or 0)
    launch_summary = {
        "run_kind": "bolt_pick_signed_contact_force_capture_only",
        "capture_status": "capture_complete" if capture_complete else "capture_incomplete",
        "output_dir": str(output_dir),
        "stdout_path": str(output_dir / "stdout.log"),
        "command_path": str(output_dir / "command.json"),
        "docker_process_exit_code": docker_exit_code,
        "runner_result_code_from_saved_smoke_status": runner_result_code,
        "capture_exit_code": capture_exit_code,
        "wrapper_exit_code": wrapper_exit_code,
        "docker_exit_masked_runner_failure": bool(
            docker_exit_code == 0 and runner_result_code not in (None, 0)
        ),
        "capture": capture_summary,
        "task_success_claimed": False,
    }
    _write_json(launch_path, launch_summary)
    print(
        f"Pick-force diagnostic capture={'complete' if capture_complete else 'incomplete'}; "
        f"runner result={runner.get('pick_result')}; wrapper exit={wrapper_exit_code}; "
        f"summary={launch_path}",
        flush=True,
    )
    return wrapper_exit_code


def _vec3_sum(contacts: list[dict[str, Any]]) -> list[float]:
    result = [0.0, 0.0, 0.0]
    for contact in contacts:
        raw = contact["force_raw"]
        normal = contact["normal_w"]
        scalar = float(raw[0])
        for axis in range(3):
            result[axis] += scalar * float(normal[axis])
    return result


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _capture_native_sample(env: Any, source: dict[str, Any], stage: str) -> dict[str, Any]:
    view = source["contact_physx_view"]
    dt_s = float(source["dt_s"])
    forces, points, normals, separations, counts, starts = view.get_contact_data(dt_s)
    pair_matrix = _py(view.get_contact_force_matrix(dt_s))
    net_forces = _py(view.get_net_contact_forces(dt_s))
    count_rows = _py(counts)
    start_rows = _py(starts)
    filters = []
    pair_sum = [0.0, 0.0, 0.0]

    for item in source["filter_map"]:
        filter_index = int(item["filter_index"])
        count = int(count_rows[0][filter_index])
        start = int(start_rows[0][filter_index])
        contacts = []
        for point_index in range(start, start + count):
            contacts.append(
                {
                    "buffer_index": point_index,
                    "force_raw": _py(forces[point_index]),
                    "point_w": _py(points[point_index]),
                    "normal_w": _py(normals[point_index]),
                    "separation_raw": _py(separations[point_index]),
                }
            )
        vector_sum = _vec3_sum(contacts)
        matrix_vector = [float(value) for value in pair_matrix[0][filter_index]]
        for axis in range(3):
            pair_sum[axis] += matrix_vector[axis]
        filters.append(
            {
                **item,
                "reported_count": count,
                "reported_start_index": start,
                "raw_contacts": contacts,
                "point_force_sum_scalar_times_normal_w_n": vector_sum,
                "pair_force_matrix_w_n": matrix_vector,
                "pair_minus_point_sum_w_n": _subtract(matrix_vector, vector_sum),
                "pair_minus_point_sum_norm_n": _norm(_subtract(matrix_vector, vector_sum)),
            }
        )

    net_sensor = [float(value) for value in net_forces[0]]
    return {
        "sim_timestamp_s": float(env._robot._data._sim_timestamp),
        "stage": stage,
        "dt_s": dt_s,
        "sensor_body_names": list(source["sensor_body_names"]),
        "filter_map": source["filter_map"],
        "raw_pair_force_matrix_w_n": pair_matrix,
        "raw_net_contact_forces_on_sensor_w_n": net_forces,
        "filters": filters,
        "pair_force_matrix_sum_w_n": pair_sum,
        "net_minus_pair_matrix_sum_w_n": _subtract(net_sensor, pair_sum),
    }


def _load_runner():
    path = REPO_ROOT / "scripts/run_bolt_harness.py"
    spec = importlib.util.spec_from_file_location("bolt_harness_runner_for_force_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load existing runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_inside_container(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "native_contact_force_samples.jsonl"
    capture_path = output_dir / "force_matrix_capture.json"
    runner_output_dir = output_dir / "pick_smoke"
    runner_summary_path = runner_output_dir / "smoke_summary.json"
    callback_count = 0
    sample_errors: list[str] = []
    started = _now().isoformat()
    runner_code = 1

    try:
        runner = _load_runner()
        original_recorder = runner._record_private_bolt_contacts
        with raw_path.open("x", encoding="utf-8") as raw_file:
            def traced_recorder(env, source, summary, stage):
                nonlocal callback_count
                sample = None
                try:
                    sample = _capture_native_sample(env, source, stage)
                except Exception as exc:
                    sample_errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    original_recorder(env, source, summary, stage)
                finally:
                    if sample is not None:
                        raw_file.write(json.dumps(_json_safe(sample), sort_keys=True, allow_nan=False) + "\n")
                        raw_file.flush()
                        callback_count += 1

            runner._record_private_bolt_contacts = traced_recorder
            sys.argv = [
                str(REPO_ROOT / "scripts/run_bolt_harness.py"),
                "--config",
                args.config,
                "--stage",
                "pick",
                "--output-dir",
                str(runner_output_dir),
                "--headless",
                "--device",
                "cuda:0",
            ]
            runner_code = int(runner.main())
    except BaseException as exc:
        sample_errors.append(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, SystemExit):
            runner_code = int(exc.code or 0)
        else:
            runner_code = 1
            raise
    finally:
        runner_summary = {}
        try:
            runner_summary = json.loads(runner_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        capture_summary = {
            "run_kind": "bolt_pick_signed_contact_force_capture_only",
            "capture_scope": "existing pick stage and transport; not insertion/task-success evaluation",
            "started_at_utc": started,
            "finished_at_utc": _now().isoformat(),
            "raw_samples_path": str(raw_path),
            "runner_summary_path": str(runner_summary_path),
            "raw_sample_count": callback_count,
            "raw_capture_complete": callback_count > 0 and not sample_errors,
            "capture_errors": sample_errors,
            "runner_exit_code": runner_code,
            "pick_result": runner_summary.get("pick_result"),
            "transport_result": runner_summary.get("transport_result"),
            "task_success": None,
            "final_success_claimed": False,
            "task_success_claimed": False,
        }
        _write_json(capture_path, capture_summary)
        if runner_summary:
            runner_summary["native_contact_force_capture"] = capture_summary
            _write_json(runner_summary_path, runner_summary)
        print(
            "BOLT_PICK_FORCE_CAPTURE="
            + json.dumps(
                {
                    "raw_sample_count": callback_count,
                    "raw_capture_complete": capture_summary["raw_capture_complete"],
                    "runner_exit_code": runner_code,
                    "task_success_claimed": False,
                    "raw_samples_path": str(raw_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return runner_code


def _host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config/bolt_insertion.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--analyze-existing", help="finalize an existing capture folder without launching simulation")
    return parser


def _container_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def _launch(args: argparse.Namespace) -> int:
    from tools.calibrate_bolt_contact import _docker_preflight

    run_id = f"gpu2-pick-force-{_now().strftime('%Y%m%dT%H%M%S')}-{_now().microsecond:06d}Z"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "artifacts/bolt_insertion/contact_calibration" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight = _docker_preflight()
    relative_output = output_dir.relative_to(REPO_ROOT)
    in_container_script = CONTAINER_REPO / "tools/capture_bolt_pick_force_matrix.py"
    in_container_output = CONTAINER_REPO / relative_output
    in_container_config = Path(args.config).resolve()
    if not in_container_config.is_relative_to(REPO_ROOT):
        raise ValueError(f"config must be inside the mounted repository: {in_container_config}")
    config_path = CONTAINER_REPO / in_container_config.relative_to(REPO_ROOT)
    container_name = f"bolt-pick-force-{run_id}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus=device=2",
        "--shm-size=16g",
        "-w",
        str(CONTAINER_REPO),
        "--entrypoint",
        "/isaac-sim/python.sh",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "-e",
        "OMNI_ENV_PRIVACY_CONSENT=YES",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        f"PYTHONPATH={CONTAINER_REPO}",
        "-v",
        f"{REPO_ROOT}:{CONTAINER_REPO}:rw",
        IMAGE,
        str(in_container_script),
        "--container-run",
        "--output-dir",
        str(in_container_output),
        "--config",
        str(config_path),
    ]
    _write_json(
        output_dir / "command.json",
        {
            "created_at_utc": _now().isoformat(),
            "docker_argv": command,
            "docker_command_display": shlex.join(command),
            "container_name": container_name,
            "gpu_request": "device=2 only; in-container CUDA device is cuda:0",
            "repo_mount": f"{REPO_ROOT}:{CONTAINER_REPO}:rw",
            "config_path": str(config_path),
            "stage": "pick",
            "gpu_preflight": preflight,
        },
    )
    summary_path = output_dir / "launch_summary.json"
    _write_json(
        summary_path,
        {
            "run_kind": "bolt_pick_signed_contact_force_capture_only",
            "capture_status": "container_not_started",
            "output_dir": str(output_dir),
            "gpu_preflight": preflight,
            "task_success_claimed": False,
        },
    )
    stdout_path = output_dir / "stdout.log"
    with stdout_path.open("wb") as stdout_file:
        process = subprocess.run(command, stdout=stdout_file, stderr=subprocess.STDOUT, check=False)
    return _finalize_existing_capture(output_dir, int(process.returncode))


def main() -> int:
    mode = argparse.ArgumentParser(add_help=False)
    mode.add_argument("--container-run", action="store_true")
    parsed, _ = mode.parse_known_args()
    if parsed.container_run:
        return _run_inside_container(_container_parser().parse_args())
    args = _host_parser().parse_args()
    if args.analyze_existing:
        folder = Path(args.analyze_existing).expanduser().resolve()
        previous = _read_json(folder / "launch_summary.json")
        docker_code = previous.get("docker_process_exit_code")
        return _finalize_existing_capture(folder, docker_code if isinstance(docker_code, int) else None)
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
