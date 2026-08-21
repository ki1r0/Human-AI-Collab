"""assembly/instantiate.py — expand the generic gearbox DAG into concrete,
ordered step instances bound to asset_registry.yaml.

Pure offline (stdlib + pyyaml). No pxr/Isaac imports. See
docs/superpowers/specs/2026-06-05-dag-instantiator-design.md
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

import yaml

_HERE = Path(__file__).resolve().parent
DEFAULT_DAG = _HERE / "gearbox_dag.yaml"
DEFAULT_REGISTRY = _HERE / "asset_registry.yaml"
DEFAULT_OUT_DIR = _HERE / "instances"
VARIANTS = ["canonical_grouped", "interleaved_covers", "random_topo"]


def load_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_registry_index(registry: dict) -> Dict[str, dict]:
    return {p["name"]: p for p in registry.get("parts", [])}


def resolve_set(query: dict, parts: Dict[str, dict]) -> List[str]:
    """Return names of registry parts matching the query, sorted by name."""
    out = []
    for name, p in parts.items():
        if p.get("role") != query.get("role"):
            continue
        if "parent" in query:
            pc = p.get("parent_connection") or {}
            if pc.get("parent") != query["parent"]:
                continue
        if "name_prefix" in query and not name.startswith(query["name_prefix"]):
            continue
        out.append(name)
    return sorted(out)


def slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "scene").lower()).strip("_")


def _new_step(op, action, child, **kw) -> dict:
    st = {
        "key": f"{op['id']}:{action}:{child}",
        "op": op["id"],
        "phase": op["phase"],
        "stream": op["stream"],
        "action": action,
        "child": child,
        "parent": kw.get("parent"),
        "plug": kw.get("plug"),
        "socket": kw.get("socket"),
        "axis": kw.get("axis"),
        "hover_m": kw.get("hover_m"),
        "condition": kw.get("condition"),
        "simultaneous": None,
        "pre_keys": [],
        "cover_key": None,
        "is_paired_nut": kw.get("is_paired_nut", False),
        "injected": kw.get("injected", False),
    }
    return st


def _nut_for_bolt(bolt_name: str, parts: Dict[str, dict]):
    for name, p in parts.items():
        pc = p.get("parent_connection") or {}
        if pc.get("parent") == bolt_name and name.startswith("M10_Casing_Nut"):
            return name, (p.get("plug") or pc.get("plug")), pc.get("socket")
    raise ValueError(f"no paired nut found for {bolt_name}")


def make_member_steps(op: dict, parts: Dict[str, dict], sets: Dict[str, list]) -> List[dict]:
    """Expand one op into its member steps (no preconditions/pose/simultaneous yet)."""
    action = op["action"]
    out: List[dict] = []

    # Single-target ops: inspect/focus/unfocus/upright, and the 1-member mate combine.
    if "child_set" not in op:
        child = op.get("target")
        parent = op.get("parent")
        plug = socket = None
        if action == "combine" and parent:  # op_mate_top
            pc = parts[child].get("parent_connection") or {}
            plug, socket = pc.get("plug"), pc.get("socket")
        out.append(_new_step(op, action, child, parent=parent, plug=plug,
                             socket=socket, axis=op.get("axis")))
        return out

    # Set ops: one step per member, bound from the registry parent_connection.
    for member in sets[op["child_set"]]:
        pc = parts[member].get("parent_connection") or {}
        out.append(_new_step(op, action, member, parent=op["parent"],
                             plug=pc.get("plug"), socket=pc.get("socket"),
                             hover_m=op.get("hover_m")))
        if op.get("emit_paired_nut"):
            nut, nut_plug, nut_socket = _nut_for_bolt(member, parts)
            out.append(_new_step(op, "combine", nut, parent=member,
                                 plug=nut_plug, socket=nut_socket, is_paired_nut=True))
    return out


def _cover_name_for_bolt(bolt_part: dict, cbb: dict, parts: Dict[str, dict]) -> str:
    socket = (bolt_part.get("parent_connection") or {}).get("socket", "")
    n = int(socket.rsplit("_", 1)[1])
    parent = (bolt_part.get("parent_connection") or {}).get("parent")
    for cover_name, idxs in cbb.items():
        if n in idxs and (parts[cover_name].get("parent_connection") or {}).get("parent") == parent:
            return cover_name
    raise ValueError(f"no cover binds socket {n} on {parent}")


def wire_preconditions(steps, ops, op_member_keys, op_check_keys, op_flip_keys,
                       parts, cbb) -> None:
    """Populate pre_keys on every step. op_check_keys/op_flip_keys come from
    pose-check injection (empty dicts if injection is skipped)."""
    # name -> member key (each scene part appears once as a member step)
    member_key_by_child = {
        s["child"]: k for k, s in steps.items()
        if not s["injected"] and not s["is_paired_nut"]
    }
    for op in ops:
        oid = op["id"]
        op_pre = []
        for p in (op.get("preconditions") or []):
            op_pre.extend(op_member_keys.get(p, []))
        # check steps depend on the op's upstream member steps
        for ck in op_check_keys.get(oid, []):
            steps[ck]["pre_keys"] = list(op_pre)
        # flip steps already point at their check (set during injection)
        # member base precondition: flips if pose-gated, else upstream members
        member_base = list(op_flip_keys.get(oid, [])) if op.get("pose_check") else list(op_pre)
        for k in op_member_keys[oid]:
            st = steps[k]
            if op.get("bind_each_to_cover"):
                cover_name = _cover_name_for_bolt(parts[st["child"]], cbb, parts)
                st["cover_key"] = member_key_by_child[cover_name]
                st["pre_keys"] = [st["cover_key"]]
            elif st["is_paired_nut"]:
                st["pre_keys"] = [f"{oid}:combine:{st['parent']}"]
            else:
                st["pre_keys"] = list(member_base)


def inject_pose_checks(steps, ops):
    """For each op with non-empty pose_check, add a check_pose + conditional flip
    step per object. Returns (op_check_keys, op_flip_keys)."""
    op_check_keys = collections.defaultdict(list)
    op_flip_keys = collections.defaultdict(list)
    for op in ops:
        for obj in (op.get("pose_check") or []):
            ck = f"{op['id']}:check_pose:{obj}"
            fk = f"{op['id']}:flip:{obj}"
            steps[ck] = _new_step(op, "check_pose", obj, injected=True)
            steps[fk] = _new_step(op, "flip", obj, axis="x", injected=True,
                                  condition={"check_key": ck, "equals": 0})
            steps[fk]["pre_keys"] = [ck]
            op_check_keys[op["id"]].append(ck)
            op_flip_keys[op["id"]].append(fk)
    return op_check_keys, op_flip_keys


def tag_simultaneous(steps, ops, op_member_keys) -> None:
    for op in ops:
        sim = op.get("simultaneous")
        if not sim:
            continue
        if sim is True:
            group = slug(op["id"].replace("op_", ""))  # e.g. "stage_shafts"
            for k in op_member_keys[op["id"]]:
                steps[k]["simultaneous"] = group
        elif sim == "pairs":
            for k in op_member_keys[op["id"]]:
                st = steps[k]
                bolt = st["parent"] if st["is_paired_nut"] else st["child"]
                num = bolt.rsplit("_", 1)[1]  # "01".."06"
                st["simultaneous"] = f"m10_pair_{num}"


def expand(dag: dict, parts: Dict[str, dict]) -> Dict[str, dict]:
    """Full variant-independent expansion: members + pose injection + wiring + simultaneous."""
    sets = {n: resolve_set(q, parts) for n, q in dag["sets"].items()}
    empty = [n for n, members in sets.items() if not members]
    if empty:
        raise ValueError(f"set query resolved to no registry parts: {empty}")
    ops = dag["operations"]
    steps: Dict[str, dict] = {}
    op_member_keys = collections.defaultdict(list)
    for op in ops:
        for st in make_member_steps(op, parts, sets):
            steps[st["key"]] = st
            op_member_keys[op["id"]].append(st["key"])
    op_check_keys, op_flip_keys = inject_pose_checks(steps, ops)
    wire_preconditions(steps, ops, op_member_keys, op_check_keys, op_flip_keys,
                       parts, dag["cover_bolt_binding"])
    tag_simultaneous(steps, ops, op_member_keys)
    return steps


def kahn(steps_by_key: Dict[str, dict], select: Callable[[list], str]) -> List[str]:
    indeg = {k: 0 for k in steps_by_key}
    adj = collections.defaultdict(list)
    for k, s in steps_by_key.items():
        for p in s["pre_keys"]:
            adj[p].append(k); indeg[k] += 1
    ready = [k for k in steps_by_key if indeg[k] == 0]
    order = []
    while ready:
        k = select(ready)
        ready.remove(k)
        order.append(k)
        for m in adj[k]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(order) != len(steps_by_key):
        raise ValueError("cycle detected during linearization")
    return order


def _op_groups(steps_by_key, op_id):
    """Return (checks, flips, members) keys for an op, members sorted by child."""
    checks, flips, members = [], [], []
    for k, s in steps_by_key.items():
        if s["op"] != op_id:
            continue
        if s["action"] == "check_pose":
            checks.append(k)
        elif s["injected"] and s["action"] == "flip":
            flips.append(k)
        else:
            members.append(k)
    members.sort(key=lambda k: steps_by_key[k]["child"])
    return checks, flips, members


def _covers_op_for_bolts(ops, bolt_op):
    covers_sets = {"base_hub_covers", "top_hub_covers"}
    for o in ops:
        if o.get("child_set") in covers_sets and o["stream"] == bolt_op["stream"]:
            return o["id"]
    return None


def desired_order(steps_by_key, dag, variant) -> List[str]:
    ops = dag["operations"]
    want: List[str] = []
    if variant == "interleaved_covers":
        bolt_op_ids = {o["id"] for o in ops if o.get("bind_each_to_cover")}
        covers_for = {bid: _covers_op_for_bolts(ops, next(o for o in ops if o["id"] == bid))
                      for bid in bolt_op_ids}
        handled_bolt = set()
        for op in ops:
            oid = op["id"]
            if oid in handled_bolt:
                continue
            checks, flips, members = _op_groups(steps_by_key, oid)
            want += checks + flips
            paired_bolt = next((b for b, c in covers_for.items() if c == oid), None)
            if paired_bolt:
                _, _, bolt_members = _op_groups(steps_by_key, paired_bolt)
                bolts_by_cover = collections.defaultdict(list)
                for bk in bolt_members:
                    bolts_by_cover[steps_by_key[bk]["cover_key"]].append(bk)
                for ck in members:  # cover member keys, already name-sorted
                    want.append(ck)
                    want += sorted(bolts_by_cover.get(ck, []),
                                   key=lambda k: steps_by_key[k]["child"])
                handled_bolt.add(paired_bolt)
            else:
                want += members
        return want
    # canonical_grouped: ops in declaration order; checks, flips, members
    for op in ops:
        checks, flips, members = _op_groups(steps_by_key, op["id"])
        want += checks + flips + members
    return want


def linearize(steps_by_key, dag, variant, seed) -> List[str]:
    if variant == "random_topo":
        rng = random.Random(seed)
        return kahn(steps_by_key, lambda ready: rng.choice(sorted(ready)))
    want = desired_order(steps_by_key, dag, variant)
    rank = {k: i for i, k in enumerate(want)}
    return kahn(steps_by_key, lambda ready: min(ready, key=lambda k: rank[k]))


def auto_label(step) -> str:
    a, c, p = step["action"], step.get("child"), step.get("parent")
    return {
        "combine": f"Install {c} onto {p}" if p else f"Install {c}",
        "stage": f"Stage {c} above its socket on {p}",
        "focus": f"Move {c} to the working area",
        "unfocus": f"Move {c} away from the working area",
        "upright": f"Stand {c} upright",
        "flip": f"Flip {c} if pose check fails",
        "check_pose": f"Check {c} is correctly posed",
        "inspect": "Validate scene",
    }.get(a, f"{a} {c}")


def _emit_step(step, idx, key_to_id) -> dict:
    out = {
        "id": key_to_id[step["key"]],
        "phase": step["phase"],
        "stream": step["stream"],
        "op": step["op"],
        "action": step["action"],
        "child": step["child"],
        "parent": step["parent"],
        "plug": step["plug"],
        "socket": step["socket"],
    }
    if step["axis"] is not None:
        out["axis"] = step["axis"]
    if step["hover_m"] is not None:
        out["hover_m"] = step["hover_m"]
    if step["condition"] is not None:
        out["condition"] = {"check": key_to_id[step["condition"]["check_key"]],
                            "equals": step["condition"]["equals"]}
    out["simultaneous"] = step["simultaneous"]
    out["preconditions"] = [key_to_id[p] for p in step["pre_keys"]]
    out["human_label"] = auto_label(step)
    out["executor_mode"] = "magic"
    return out


def build_instance(dag, parts, variant, seed: int = 0) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    steps_by_key = expand(dag, parts)
    order = linearize(steps_by_key, dag, variant, seed)
    key_to_id = {k: f"inst_{i + 1:03d}_{steps_by_key[k]['action']}_{slug(steps_by_key[k]['child'])}"
                 for i, k in enumerate(order)}
    steps = [_emit_step(steps_by_key[k], i, key_to_id) for i, k in enumerate(order)]
    return {
        "meta": {
            "variant": variant,
            "source_dag": "assembly/gearbox_dag.yaml",
            "generated_by": "assembly/instantiate.py",
            "seed": seed if variant == "random_topo" else None,
            "total_steps": len(steps),
        },
        "steps": steps,
    }


def validate_instance(instance: dict, parts: Dict[str, dict]) -> None:
    steps = instance["steps"]
    ids = [s["id"] for s in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate step ids")
    seen = set()
    id_set = set(ids)
    for s in steps:
        # registry resolution
        for fld in ("child", "parent"):
            name = s.get(fld)
            if name is not None and name not in parts:
                raise ValueError(f"{s['id']}: {fld}={name!r} not in registry")
        if s["action"] in ("combine", "stage"):
            if s["plug"] and s["plug"] not in (parts[s["child"]].get("plugs") or []):
                raise ValueError(f"{s['id']}: plug {s['plug']!r} not on {s['child']}")
            if s["socket"] and s["parent"] and \
                    s["socket"] not in (parts[s["parent"]].get("sockets") or []):
                raise ValueError(f"{s['id']}: socket {s['socket']!r} not on {s['parent']}")
        # preconditions exist and precede (topo order valid)
        for p in s["preconditions"]:
            if p not in id_set:
                raise ValueError(f"{s['id']}: unknown precondition {p}")
            if p not in seen:
                raise ValueError(f"{s['id']}: precondition {p} appears later (not topo-ordered)")
        # conditional flip references an existing earlier check_pose
        cond = s.get("condition")
        if cond is not None:
            if cond["check"] not in id_set:
                raise ValueError(f"{s['id']}: condition.check {cond['check']} missing")
            if cond["check"] not in seen:
                raise ValueError(f"{s['id']}: condition.check {cond['check']} not earlier")
        seen.add(s["id"])


def list_variants() -> List[str]:
    return list(VARIANTS)


def load_instance(name: str, out_dir=DEFAULT_OUT_DIR) -> dict:
    return load_yaml(Path(out_dir) / f"{name}.yaml")


def write_instance(instance: dict, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{instance['meta']['variant']}.yaml"
    with open(path, "w") as fh:
        fh.write("# GENERATED by assembly/instantiate.py — do not edit by hand.\n")
        fh.write("# Regenerate: python3 -m assembly.instantiate --variant all\n")
        yaml.safe_dump(instance, fh, sort_keys=False, default_flow_style=False, width=100)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Instantiate gearbox assembly variants from the DAG.")
    ap.add_argument("--variant", default="all", help="'all' or one of: " + ", ".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0, help="seed for random_topo")
    ap.add_argument("--dag", default=str(DEFAULT_DAG))
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)

    dag = load_yaml(args.dag)
    parts = build_registry_index(load_yaml(args.registry))
    variants = VARIANTS if args.variant == "all" else [args.variant]
    for v in variants:
        inst = build_instance(dag, parts, v, seed=args.seed)
        validate_instance(inst, parts)
        path = write_instance(inst, args.out_dir)
        print(f"[OK] {v}: {inst['meta']['total_steps']} steps -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
