"""Fit count estimates from independent physical branch records, never from test runs."""

import argparse
import collections
import json
import math
from pathlib import Path

from .decision import Estimator, state_key


COSTS = {"wall_s", "contact_exposure_Ns", "helper_effort", "model_money"}
OUTCOMES = {"SUCCESS", "PROGRESS", "STILL_STALLED", "GRASP_LOST", "PROTECTION", "UNKNOWN"}


def record_branch(output, plan):
    """Extract AFTER-terminal labels from a reset-and-prefix replay, not a pose snapshot restore."""
    output = Path(output)
    decisions = [json.loads(line) for line in (output / "decision_trace.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (output / "public_trace.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    index = len(plan["prefix"])
    if index >= len(decisions):
        raise ValueError("calibration prefix did not reach the branch")
    results = {event["payload"]["call_id"]: event["payload"] for event in events if event["kind"] == "tool_result"}
    result = results[f"a_{index}"]
    if result["status"] == "REJECTED":
        raise ValueError("rejected tool is not an executed calibration branch")
    terminal = summary["primary_gt"]
    if terminal == "UNKNOWN":
        raise ValueError("uncertified/missing terminal GT cannot fit a response estimator")
    before = decisions[index]["public_state"]
    after = decisions[index+1]["public_state"] if index+1 < len(decisions) else None
    observation = next(event["payload"] for event in events if event["kind"] == "observation" and
                       event["payload"]["observation_id"] == result["observation_id_after"])
    tool = next(event["payload"]["tool"] for event in events if event["kind"] == "tool_call" and
                event["payload"]["call_id"] == f"a_{index}")
    if tool == "ask_act" and summary.get("help_delivery") == "local_outbox":
        raise ValueError("a help request has no helper-completion label; do not fit it as task completion")
    channels = observation["channels"]
    reported = all(channels.get(name, {}).get("value") is True for name in
                   ("visual_seated", "released", "stable_observed"))
    label = ("SUCCESS" if reported else "PROTECTION" if result["status"] == "ABORTED" else
             "GRASP_LOST" if channels.get("holding", {}).get("value") == "no" else
             "STILL_STALLED" if result["status"] == "STALLED" else
             "PROGRESS" if channels.get("tcp_axial_delta_mm", {}).get("value") else "UNKNOWN")
    # Only the candidate and its continuation are charged, excluding the shared prefix.
    begin = observation["wall_timestamp"] - result["measured_cost"]["motion_s"] - result["measured_cost"]["helper_s"]
    prefix_call = f"a_{index-1}" if index else None
    if prefix_call:
        prefix_obs = results[prefix_call]["observation_id_after"]
        initial_obs = next(event["payload"] for event in events if event["kind"] == "observation" and
                           event["payload"]["observation_id"] == prefix_obs)
        begin = initial_obs["wall_timestamp"]
    else:
        initial_obs = next(event["payload"] for event in events if event["kind"] == "observation")
    is_help = tool == "ask_act"
    final_obs = [event["payload"] for event in events if event["kind"] == "observation"][-1] if is_help else observation
    executed = [value for name, value in results.items() if (int(name.split("_")[-1]) >= index if is_help
                else name == f"a_{index}")]
    costs = {"wall_s": max(0, final_obs["wall_timestamp"]-begin), **{name: sum(value["measured_cost"][name] for value in executed)
             for name in ("contact_exposure_Ns", "helper_effort", "model_money")}}
    initial_load = initial_obs["channels"].get("contact_exposure_Ns", {}).get("value")
    final_load = final_obs["channels"].get("contact_exposure_Ns", {}).get("value")
    if initial_load is None or final_load is None:
        raise ValueError("missing measured contact exposure cannot be fitted as zero cost")
    costs["contact_exposure_Ns"] = max(0, final_load-initial_load)
    row = {"snapshot_id": plan["snapshot_id"], "split": plan["split"], "backend": summary["backend"],
           "candidate_id": plan["candidate_id"], "tool": tool, "public_state": before, "next_state": after,
           "outcome": label, "terminal_success": terminal == "PASS", "cost": costs,
           "completion_scope": "help_verify_robot_resume" if is_help else "action",
           "restore_method": "independent_reset_and_prefix_replay", "prefix": plan["prefix"]}
    if "initial_category" in plan:
        row["initial_category"] = plan["initial_category"]
    (output / "calibration_record.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def entropy(counts):
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def fit(records, *, allow_contract_data=False):
    groups, splits, backends = collections.defaultdict(list), {}, set()
    for row in records:
        snapshot = row["snapshot_id"]
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown calibration split")
        if snapshot in splits and splits[snapshot] != split:
            raise ValueError("branches of one initial snapshot cross data splits")
        splits[snapshot] = split
        backend = row["backend"]
        backends.add(backend)
        if backend not in {"isaac", "contract_only"}:
            raise ValueError("uncertified calibration source")
        if backend == "contract_only" and not allow_contract_data:
            raise ValueError("contract data is not physical calibration")
        key = state_key(row["public_state"])
        if row["outcome"] not in OUTCOMES or type(row["terminal_success"]) is not bool:
            raise ValueError("invalid branch outcome")
        if set(row["cost"]) != COSTS or any(not math.isfinite(v) or v < 0 for v in row["cost"].values()):
            raise ValueError("invalid measured branch costs")
        if row.get("next_state") is not None:
            state_key(row["next_state"])
        if (row.get("tool") == "ask_act" or row["candidate_id"] == "help") and row.get("completion_scope") != "help_verify_robot_resume":
            raise ValueError("help labels must include verification and robot completion")
        if split == "train":
            groups[key, row["candidate_id"]].append(row)
    if len(backends) != 1 or not groups:
        raise ValueError("need nonempty, unmixed training data")
    table = {"version": 1, "provenance": {"backend": next(iter(backends)),
             "train_snapshots": sum(split == "train" for split in splits.values()),
             "estimator": "public_bins_empirical_counts"}, "states": {}}
    for (key, candidate), rows in groups.items():
        outcomes = {}
        for label in OUTCOMES:
            matches = [row for row in rows if row["outcome"] == label]
            if not matches:
                continue
            states = {state_key(row["next_state"]) for row in matches if row.get("next_state") is not None}
            # ponytail: ambiguous next-state bins abstain; expand to distributions when data supports them.
            next_state = json.loads(next(iter(states))) if len(states) == 1 and all(
                row.get("next_state") is not None for row in matches) else None
            outcomes[label] = {"probability": len(matches) / len(rows), "next_state": next_state,
                               "terminal_success_probability": sum(row["terminal_success"] for row in matches) / len(matches)
                               if label == "SUCCESS" else None}
        estimate = {"success_probability": sum(row["terminal_success"] for row in rows) / len(rows),
                    "support_count": len(rows),
                    "cost": {name: sum(row["cost"][name] for row in rows) / len(rows) for name in COSTS},
                    "outcomes": outcomes}
        if all(row.get("initial_category") is not None for row in rows):
            categories = collections.Counter(row["initial_category"] for row in rows)
            posterior = 0.0
            for label in OUTCOMES:
                matches = [row for row in rows if row["outcome"] == label]
                posterior += len(matches) / len(rows) * entropy(collections.Counter(
                    row["initial_category"] for row in matches))
            estimate["information_gain_bits"] = max(0.0, entropy(categories) - posterior)
        table["states"].setdefault(key, {})[candidate] = estimate
    estimator = Estimator(table, allow_contract_data=allow_contract_data)
    validation = [row for row in records if row["split"] == "validation"]
    supported = [(row, estimator.predict(row["public_state"], row["candidate_id"])) for row in validation]
    supported = [(row, estimate) for row, estimate in supported if estimate is not None]
    report = {"validation_count": len(validation), "supported_count": len(supported),
              "coverage": len(supported) / len(validation) if validation else None,
              "brier": sum((estimate.success_probability - row["terminal_success"]) ** 2
                           for row, estimate in supported) / len(supported) if supported else None,
              "test_records_not_fitted": sum(row["split"] == "test" for row in records)}
    return table, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    table, report = fit(records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "estimator.json").write_text(json.dumps(table, indent=2) + "\n")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
