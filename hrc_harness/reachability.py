"""Offline USD arm IK audit; not runtime part feedback or physics validation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def audit(robot_usd, reference, wrist_z=1.27089, centers=None, quaternion=(0, 1, 0, 0), pair_offset=.085):
    stage = Usd.Stage.Open(str(robot_usd))
    cache = UsdGeom.XformCache()
    joints = {p.GetName(): UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
              if p.IsA(UsdPhysics.RevoluteJoint)}
    setup = json.loads((reference / "robot_setup.json").read_text())
    initial = json.loads((reference / "public_samples.jsonl").open().readline())
    def matrix(position, quaternion):
        value = Gf.Matrix4d(1)
        value.SetRotate(Gf.Rotation(Gf.Quatd(quaternion)))
        value.SetTranslateOnly(Gf.Vec3d(position))
        return np.asarray(value).T

    arms = {}
    for side in ("left", "right"):
        chain = [joints[f"{side}_arm_joint{i}"] for i in range(1, 7)]
        parent = stage.GetPrimAtPath(chain[0].GetBody0Rel().GetTargets()[0])
        base = np.asarray(cache.GetLocalToWorldTransform(parent)).T
        axes = np.array([{"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[j.GetAxisAttr().Get()]
                         for j in chain])
        origins = [matrix(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()) for j in chain]
        children = [np.linalg.inv(matrix(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())) for j in chain]
        indices = [setup["joint_names"].index(f"{side}_arm_joint{i}") for i in range(1, 7)]
        bounds = np.array([setup["joint_limits"][i] for i in indices])
        seed = np.array([initial["qpos"][i] for i in indices])
        def fk(q, base=base, origins=origins, children=children, axes=axes):
            result = base.copy()
            for angle, origin, child, axis in zip(q, origins, children, axes):
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                result = result @ origin @ turn @ child
            return result
        arms[side] = (fk, bounds, seed)

    left = arms["left"][0](arms["left"][2])
    tcp = left[:3, 3] + left[:3, :3] @ np.array([.0079, 0, .09089])
    fk_error = math.dist(tcp, initial["tcp_xyz_m"])
    if fk_error > .001:
        raise ValueError(f"USD FK does not match public robot telemetry: {fk_error} m")
    w, x, y, z = quaternion
    target_rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    results = []
    if centers is None:
        centers = [(x, y) for x in (.22, .25, .27, .3, .35, .4, .5, .55)
                   for y in (-.22, -.2, -.18, -.17, -.15, -.1, 0, .1, .15, .17, .18, .2, .22)]
    for x, y in centers:
        row = {"grasp_center_xy_m": [x, y], "arms": {}}
        for side, sign in (("left", 1), ("right", -1)):
            fk, bounds, seed = arms[side]
            target = np.array([x, y + sign * pair_offset, wrist_z])
            def residual(q):
                pose = fk(q)
                return np.r_[pose[:3, 3] - target,
                             .1 * Rotation.from_matrix(pose[:3, :3].T @ target_rotation).as_rotvec()]
            candidates = [least_squares(residual, np.clip(q, bounds[:, 0]+1e-6, bounds[:, 1]-1e-6),
                                         bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=150)
                          for q in (seed, bounds.mean(axis=1))]
            best = min(candidates, key=lambda result: np.linalg.norm(result.fun))
            error = residual(best.x)
            row["arms"][side] = {"position_error_m": float(np.linalg.norm(error[:3])),
                "orientation_error_deg": math.degrees(float(np.linalg.norm(error[3:]))/.1),
                "joint_positions_rad": best.x.tolist()}
        row["reachable"] = all(a["position_error_m"] < .003 and a["orientation_error_deg"] < 2
                              for a in row["arms"].values())
        results.append(row)
    return {"role": "offline_kinematics_not_physics", "reference": str(reference),
            "fk_telemetry_error_m": fk_error, "wrist_z_m": wrist_z,
            "wrist_quat_wxyz": list(quaternion), "pair_offset_m": pair_offset, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wrist-z", type=float, default=1.27089)
    parser.add_argument("--cover-xy", type=float, nargs=2, action="append")
    parser.add_argument("--quat", type=float, nargs=4, default=[0, 1, 0, 0])
    parser.add_argument("--pair-offset", type=float, default=.085)
    args = parser.parse_args()
    result = audit(args.robot_usd, args.reference, args.wrist_z, args.cover_xy, args.quat, args.pair_offset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    print("reachable", [r["grasp_center_xy_m"] for r in result["results"] if r["reachable"]])


if __name__ == "__main__":
    main()
