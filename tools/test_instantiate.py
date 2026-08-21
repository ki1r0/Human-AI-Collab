#!/usr/bin/env python3
"""Standalone test suite for the DAG + instantiator.
Run: python3 tools/test_instantiate.py
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAG_PATH = os.path.join(REPO, "assembly", "gearbox_dag.yaml")
REG_PATH = os.path.join(REPO, "assembly", "asset_registry.yaml")


def _load(p):
    with open(p) as fh:
        return yaml.safe_load(fh)


def test_dag_has_required_blocks():
    dag = _load(DAG_PATH)
    for block in ("meta", "phases", "streams", "sets", "cover_bolt_binding", "operations"):
        assert block in dag, f"missing block: {block}"


def test_dag_operations_acyclic_and_no_dangling_preconditions():
    dag = _load(DAG_PATH)
    ops = dag["operations"]
    ids = [o["id"] for o in ops]
    assert len(ids) == len(set(ids)), "duplicate op ids"
    idset = set(ids)
    for o in ops:
        for p in (o.get("preconditions") or []):
            assert p in idset, f"{o['id']} has dangling precondition {p}"
    # Kahn acyclicity
    indeg = {i: 0 for i in ids}
    adj = collections.defaultdict(list)
    for o in ops:
        for p in (o.get("preconditions") or []):
            adj[p].append(o["id"]); indeg[o["id"]] += 1
    q = [i for i in ids if indeg[i] == 0]; seen = 0
    while q:
        n = q.pop(); seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0: q.append(m)
    assert seen == len(ids), "operations graph has a cycle"
    assert "op_flip_base" not in idset, "op_flip_base should not exist (pose_check handles flips)"


from assembly import instantiate as I  # noqa: E402


def test_resolve_set_base_m6_bolts_has_12_sorted():
    parts = I.build_registry_index(_load(REG_PATH))
    bolts = I.resolve_set({"role": "fastener", "parent": "Casing_Base",
                           "name_prefix": "M6_Hub_Bolt"}, parts)
    assert len(bolts) == 12, bolts
    assert bolts == sorted(bolts)
    assert bolts[0] == "M6_Hub_Bolt_01_base"
    assert "M6_Hub_Bolt_13_base" not in bolts  # stray, not in registry


def test_resolve_set_shafts_and_m10_and_accessories():
    parts = I.build_registry_index(_load(REG_PATH))
    assert I.resolve_set({"role": "shaft", "parent": "Casing_Base"}, parts) == \
        ["Input_Shaft", "Output_Shaft", "Transfer_Shaft"]
    assert len(I.resolve_set({"role": "fastener", "parent": "Casing_Top",
                              "name_prefix": "M10_Casing_Bolt"}, parts)) == 6
    assert set(I.resolve_set({"role": "accessory", "parent": "Casing_Base"}, parts)) == \
        {"Oil_Level_Indicator_01", "Oil_Level_Indicator_02", "Breather_Plug"}


def test_slug():
    assert I.slug("Hub_Cover_Output_Base") == "hub_cover_output_base"
    assert I.slug(None) == "scene"


def _members():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    sets = {n: I.resolve_set(q, parts) for n, q in dag["sets"].items()}
    steps = {}
    omk = collections.defaultdict(list)
    for op in dag["operations"]:
        for st in I.make_member_steps(op, parts, sets):
            steps[st["key"]] = st
            omk[op["id"]].append(st["key"])
    return steps, omk


def test_make_member_steps_counts():
    steps, omk = _members()
    assert len(omk["op_focus_base"]) == 1
    assert len(omk["op_install_base_covers"]) == 3
    assert len(omk["op_install_base_m6_bolts"]) == 12
    assert len(omk["op_stage_shafts"]) == 3
    # 6 bolts + 6 paired nuts
    assert len(omk["op_install_m10_pairs"]) == 12


def test_make_member_steps_binds_plug_socket_from_registry():
    steps, omk = _members()
    cover = next(steps[k] for k in omk["op_install_base_covers"]
                if steps[k]["child"] == "Hub_Cover_Output_Base")
    assert cover["action"] == "combine"
    assert cover["parent"] == "Casing_Base"
    assert cover["plug"] == "plug_main"
    assert cover["socket"] == "socket_hub_output"
    assert cover["injected"] is False


def test_make_member_steps_stage_has_hover_and_paired_nut_marked():
    steps, omk = _members()
    stage = next(steps[k] for k in omk["op_stage_shafts"])
    assert stage["action"] == "stage" and stage["hover_m"] == 0.15
    nut = next(steps[k] for k in omk["op_install_m10_pairs"]
               if steps[k]["child"].startswith("M10_Casing_Nut"))
    assert nut["is_paired_nut"] is True
    assert nut["parent"].startswith("M10_Casing_Bolt")


def _wired_no_pose():
    """Members + wiring, but skip pose-check injection (tested separately)."""
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    sets = {n: I.resolve_set(q, parts) for n, q in dag["sets"].items()}
    steps, omk = {}, collections.defaultdict(list)
    for op in dag["operations"]:
        for st in I.make_member_steps(op, parts, sets):
            steps[st["key"]] = st
            omk[op["id"]].append(st["key"])
    I.wire_preconditions(steps, dag["operations"], omk, {}, {}, parts,
                         dag["cover_bolt_binding"])
    return dag, parts, steps, omk


def test_bind_each_to_cover_bolt_depends_on_its_cover():
    dag, parts, steps, omk = _wired_no_pose()
    bolt1 = steps["op_install_base_m6_bolts:combine:M6_Hub_Bolt_01_base"]
    cover_out = "op_install_base_covers:combine:Hub_Cover_Output_Base"
    assert bolt1["pre_keys"] == [cover_out]   # socket 1 -> Output_Base [1,2,3,4]
    assert bolt1["cover_key"] == cover_out
    bolt5 = steps["op_install_base_m6_bolts:combine:M6_Hub_Bolt_05_base"]
    assert bolt5["cover_key"] == "op_install_base_covers:combine:Hub_Cover_Small_Base_01"


def test_paired_nut_depends_on_its_bolt():
    dag, parts, steps, omk = _wired_no_pose()
    nut = steps["op_install_m10_pairs:combine:M10_Casing_Nut_01"]
    assert nut["pre_keys"] == ["op_install_m10_pairs:combine:M10_Casing_Bolt_01"]


def test_default_precondition_unions_all_member_keys():
    dag, parts, steps, omk = _wired_no_pose()
    unfocus = steps["op_unfocus_base:unfocus:Casing_Base"]
    assert set(unfocus["pre_keys"]) == set(omk["op_install_base_m6_bolts"])  # all 12 bolts
    mate = steps["op_mate_top:combine:Casing_Top"]
    assert set(mate["pre_keys"]) == set(omk["op_insert_shafts"]) | set(omk["op_unfocus_top"])


def test_expand_injects_check_and_conditional_flip():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    steps = I.expand(dag, parts)
    ck = "op_install_base_covers:check_pose:Casing_Base"
    fk = "op_install_base_covers:flip:Casing_Base"
    assert steps[ck]["action"] == "check_pose" and steps[ck]["injected"] is True
    assert steps[fk]["action"] == "flip" and steps[fk]["axis"] == "x"
    assert steps[fk]["condition"] == {"check_key": ck, "equals": 0}
    assert steps[fk]["pre_keys"] == [ck]
    # check depends on focus_base member; cover members depend on the flip
    assert steps[ck]["pre_keys"] == ["op_focus_base:focus:Casing_Base"]
    cover = steps["op_install_base_covers:combine:Hub_Cover_Output_Base"]
    assert fk in cover["pre_keys"]
    # stage_shafts injects 4 checks + 4 flips
    flips = [k for k in steps if k.startswith("op_stage_shafts:flip:")]
    assert len(flips) == 4


def test_expand_tags_simultaneous_groups():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    steps = I.expand(dag, parts)
    stage = steps["op_stage_shafts:stage:Input_Shaft"]
    assert stage["simultaneous"] == "stage_shafts"
    insert = steps["op_insert_shafts:combine:Input_Shaft"]
    assert insert["simultaneous"] == "insert_shafts"
    bolt = steps["op_install_m10_pairs:combine:M10_Casing_Bolt_01"]
    nut = steps["op_install_m10_pairs:combine:M10_Casing_Nut_01"]
    assert bolt["simultaneous"] == "m10_pair_01" == nut["simultaneous"]


def _is_topo_valid(steps_list):
    seen = set()
    by_id = {s["id"]: s for s in steps_list}
    for s in steps_list:
        for p in s["preconditions"]:
            assert p in by_id, f"unknown precond {p}"
            assert p in seen, f"{s['id']} precedes its precondition {p}"
        seen.add(s["id"])
    return True


def test_linearize_canonical_is_valid_and_covers_before_bolts():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    inst = I.build_instance(dag, parts, "canonical_grouped")
    steps = inst["steps"]
    _is_topo_valid(steps)
    order = [s["id"] for s in steps]
    last_cover = max(i for i, s in enumerate(steps)
                     if s["op"] == "op_install_base_covers" and not s.get("condition")
                     and s["action"] == "combine")
    first_bolt = min(i for i, s in enumerate(steps)
                     if s["op"] == "op_install_base_m6_bolts")
    assert last_cover < first_bolt, "canonical must place all base covers before base bolts"
    # symmetric check for the top side
    last_top_cover = max(i for i, s in enumerate(steps)
                         if s["op"] == "op_install_top_covers" and not s.get("condition")
                         and s["action"] == "combine")
    first_top_bolt = min(i for i, s in enumerate(steps)
                         if s["op"] == "op_install_top_m6_bolts")
    assert last_top_cover < first_top_bolt, "canonical must place all top covers before top bolts"


def test_linearize_interleaved_places_bolts_right_after_their_cover():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    inst = I.build_instance(dag, parts, "interleaved_covers")
    steps = inst["steps"]
    _is_topo_valid(steps)
    ids = [s["id"] for s in steps]
    # the 4 bolts bound to Output_Base (sockets 1-4) come before Small_Base_01's cover
    def idx(sub): return next(i for i, x in enumerate(ids) if sub in x)
    out_cover = idx("combine_hub_cover_output_base")
    small1_cover = idx("combine_hub_cover_small_base_01")
    b1 = idx("combine_m6_hub_bolt_01_base")
    assert out_cover < b1 < small1_cover


def test_random_topo_is_valid_and_seed_reproducible():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    a = I.build_instance(dag, parts, "random_topo", seed=7)
    b = I.build_instance(dag, parts, "random_topo", seed=7)
    _is_topo_valid(a["steps"])
    assert [s["id"] for s in a["steps"]] == [s["id"] for s in b["steps"]]


def test_validate_instance_passes_for_all_variants():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    for v in I.VARIANTS:
        inst = I.build_instance(dag, parts, v, seed=0)
        I.validate_instance(inst, parts)   # raises on any problem


def test_validate_instance_catches_bad_socket():
    dag = _load(DAG_PATH)
    parts = I.build_registry_index(_load(REG_PATH))
    inst = I.build_instance(dag, parts, "canonical_grouped")
    # corrupt one combine step's socket to a non-existent name
    for s in inst["steps"]:
        if s["action"] == "combine" and s["socket"]:
            s["socket"] = "socket_does_not_exist"
            break
    try:
        I.validate_instance(inst, parts)
        raised = False
    except Exception:
        raised = True
    assert raised, "validate_instance should reject an unknown socket"


INST_DIR = os.path.join(REPO, "assembly", "instances")


def _load_instance(v):
    return _load(os.path.join(INST_DIR, f"{v}.yaml"))


def test_generated_files_exist_validate_and_same_step_multiset():
    parts = I.build_registry_index(_load(REG_PATH))
    multisets = {}
    for v in I.VARIANTS:
        inst = _load_instance(v)
        I.validate_instance(inst, parts)
        assert "shortcut_aliases" not in inst
        # identity of a step = (action, child, parent) — order/id independent
        multisets[v] = sorted((s["action"], s["child"], s["parent"]) for s in inst["steps"])
    assert multisets["canonical_grouped"] == multisets["interleaved_covers"]
    assert multisets["canonical_grouped"] == multisets["random_topo"]


def test_generated_counts_match_reqs():
    inst = _load_instance("canonical_grouped")
    steps = inst["steps"]
    n = lambda pred: sum(1 for s in steps if pred(s))
    # req: 12 M6 bolts per side
    assert n(lambda s: s["action"] == "combine" and s["child"].startswith("M6_Hub_Bolt") and s["child"].endswith("_base")) == 12
    assert n(lambda s: s["action"] == "combine" and s["child"].startswith("M6_Hub_Bolt") and s["child"].endswith("_top")) == 12
    # req 2: 3 stage + 3 insert shafts
    assert n(lambda s: s["action"] == "stage") == 3
    # req 3: 6 M10 bolts + 6 nuts onto Casing_Top / their bolt
    assert n(lambda s: s["action"] == "combine" and s["child"].startswith("M10_Casing_Bolt")) == 6
    assert n(lambda s: s["action"] == "combine" and s["child"].startswith("M10_Casing_Nut")) == 6


def test_req3_pairs_share_simultaneous_group():
    steps = _load_instance("canonical_grouped")["steps"]
    by_child = {s["child"]: s for s in steps}
    assert by_child["M10_Casing_Bolt_01"]["simultaneous"] == \
        by_child["M10_Casing_Nut_01"]["simultaneous"] == "m10_pair_01"


def test_pose_check_pairs_present_and_conditional():
    steps = _load_instance("canonical_grouped")["steps"]
    checks = [s for s in steps if s["action"] == "check_pose"]
    flips = [s for s in steps if s["action"] == "flip"]
    # 1 (base covers) + 1 (top covers) + 4 (stage shafts) + 1 (accessories) = 7
    assert len(checks) == 7 and len(flips) == 7
    for f in flips:
        assert f["condition"]["equals"] == 0
        assert any(c["id"] == f["condition"]["check"] for c in checks)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"[PASS] {name}")
            except Exception as e:
                fails += 1; print(f"[FAIL] {name}: {e}")
    sys.exit(1 if fails else 0)
