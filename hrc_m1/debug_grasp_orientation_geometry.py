"""Report which world axis the R1 jaw pair closes along for candidate poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hrc_m1.roco_adapter import RocoTaskAdapter  # noqa: E402
from hrc_m1.roco_env import make_env_classes  # noqa: E402


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def axis_quat(axis: str, degrees: float, device: torch.device) -> torch.Tensor:
    theta = math.radians(degrees) / 2.0
    vector = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    return torch.tensor([[math.cos(theta), *(math.sin(theta) * value for value in vector)]], device=device)


def vec(value: torch.Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def main() -> int:
    cfg_cls, env_cls = make_env_classes()
    cfg = cfg_cls()
    cfg.recompute_pose_ik_each_step = True
    env = env_cls(cfg)
    adapter = RocoTaskAdapter(env)
    try:
        names = list(env.robot.body_names)
        ids = [names.index("left_gripper_link1"), names.index("left_gripper_link2")]
        base = torch.tensor([[0.0, 0.0, 0.7071067812, -0.7071067812]], device=env.device)
        candidates = [("central_q", torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device))]
        for axis, degrees in (("x", -90.0), ("x", 90.0), ("y", -90.0), ("y", 90.0), ("z", 180.0)):
            candidates.append((f"base_{axis}{int(degrees):+d}", qmul(base, axis_quat(axis, degrees, env.device))))
        result = []
        for label, quat in candidates:
            adapter.reset(label, 1201, "nominal")
            hub = env.root_states()["hub"][0, :3].clone()
            # Direct link6 target matches the historical grasp probes; this
            # isolates orientation from the adapter's task-level TCP offset.
            env.set_gripper(0.04)
            env.set_pose_target("left", hub.unsqueeze(0) + torch.tensor([[0.0, 0.0, 0.048]], device=env.device), quat)
            adapter._step(60)
            bodies = env.robot.data.body_state_w[0, ids, :7]
            delta = bodies[0, :3] - bodies[1, :3]
            result.append({"label": label, "quat_wxyz": vec(quat[0]), "finger1_xyz": vec(bodies[0, :3]), "finger2_xyz": vec(bodies[1, :3]), "finger_delta_m": vec(delta), "finger_gap_m": float(torch.linalg.vector_norm(delta).item())})
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        adapter.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
