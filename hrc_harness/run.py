"""Shared episode loop. Contract smoke is explicitly not a physical M0/M1 result."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml
from hrc_m1.logger import _public

from .contracts import ToolCall
from .contract_world import ContractWorld
from .decision import Estimator, Selector
from .evidence import features, monitor
from .proposer import HttpProposer, rule_proposal, validate_proposal
from .runtime import Interrupted, Runtime


METHODS = ("proposed", "generic", "repair_adapted", "random_safe", "information", "auto", "fixed_retry")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("backend") not in {"contract_only", "isaac"}:
        raise ValueError("unknown backend")
    if config.get("primary_endpoint") not in {"seat_gt", "seat_and_registration_gt"}:
        raise ValueError("unknown primary endpoint")
    return config


def run_episode(adapter, config, output, *, method="proposed", seed=0, branch_plan=None):
    if method not in METHODS:
        raise ValueError("unknown comparison method")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    manifest = {"backend": config["backend"], "method": method, "seed": seed,
                "git_revision": revision, "worktree_dirty": bool(dirty), "config": config,
                "model_mode": "scripted_actions_no_model" if branch_plan else (
                    "http" if config.get("model") else "interface_test_rule"),
                "physical_validation": False, "primary_endpoint": config["primary_endpoint"]}
    scripted_smoke = branch_plan and not branch_plan.get("record_calibration", True)
    manifest["run_role"] = "scripted_smoke" if scripted_smoke else ("branch_calibration" if branch_plan else "episode")
    if branch_plan:
        manifest["action_script"] = branch_plan
    (output / "manifest.json").write_text(json.dumps(_public(manifest), indent=2) + "\n")
    def append(name, row):
        with (output / (name + ".jsonl")).open("a") as stream:
            stream.write(json.dumps(_public(row), allow_nan=False) + "\n")
    runtime = Runtime(adapter, config, sink=lambda row: append("public_trace", row))
    estimator = Estimator.load(Path(config["estimator_path"]) if config.get("estimator_path") else None,
                               allow_contract_data=config["backend"] == "contract_only")
    selector = Selector(estimator, cost_weights=config["cost_weights"], seed=seed, help_mode=runtime.help_mode)
    proposer = HttpProposer(config["model"], trace=lambda row: append("model_trace", row)) if config.get("model") else rule_proposal
    decisions, failures, fallback_count = [], [], 0
    try:
        for _ in range(config.get("startup_observe_ticks", 0)):
            runtime._guard()
            adapter.idle_step()
            runtime._guard()
        for index in range(config["max_decisions"]):
            if runtime.terminal:
                break
            runtime._guard()
            runtime.observe()
            view = runtime.decision_observation()
            state = features(view, runtime.clock(), config["freshness_s"], runtime.ledger)
            available = runtime.candidates()
            # Same capped candidate universe and estimator information for all comparison methods.
            probes = [spec for spec in available if spec.resources.probes][:config["candidate_probe_limit"]]
            candidates = [spec for spec in available if not spec.resources.probes] + probes
            resume = runtime.resume_resources if runtime.help_mode == "intervene_resume" and any(
                s.tool == "ask_act" for s in candidates) else 0
            rows = selector.rank(state, candidates, runtime.feasible, resume)
            context = {"goal": "Seat Hub Cover Output Top on Casing Top/socket_hub_output",
                       "observation": view.public_dict(runtime.clock(), config["freshness_s"]),
                       "outcome_window": [obs.public_dict(runtime.clock(), config["freshness_s"]) for obs in runtime.image_history],
                       "ledger": runtime.ledger.public_context(compact_observations=True), "features": state,
                       "monitor": monitor(view, runtime.clock(), config["freshness_s"]),
                       "used": asdict(runtime.used), "limits": config["budget"],
                       "verification": runtime.verification, "estimates": rows,
                       "help_mode": runtime.help_mode,
                       "uncertified_candidates": [s.candidate_id for s in runtime.registry.values() if not s.certified],
                       "candidates": [{"candidate_id": s.candidate_id, "tool": s.tool,
                                       "profile": s.profile, "is_probe": bool(s.resources.probes)} for s in candidates]}
            proposal = None
            if isinstance(proposer, HttpProposer):
                proposer.image_history = list(runtime.image_history)
            forced = branch_plan["prefix"] + [branch_plan["candidate_id"]] + branch_plan["continuation"] if branch_plan else None
            for repair in range(0 if forced else 2):
                try:
                    raw = runtime.reason(proposer, context)
                    proposal = validate_proposal(raw, context, runtime.ledger, config["candidate_probe_limit"])
                    runtime.apply_visual_facts(proposal["observed_facts"])
                    break
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    failures.append({"kind": "proposal_format", "attempt": repair + 1, "error": str(error)})
                    context["format_error"] = str(error)
            if forced:
                selected = "finish" if monitor(view, runtime.clock(), config["freshness_s"])["reported_success"] else (
                    forced[index] if index < len(forced) else "stop")
                why = "offline_branch_script_no_GT_feedback"
            elif proposal is None:
                selected, why = "stop", "invalid_proposal_after_one_repair"
            else:
                proposed_ids = {item["candidate_id"] for item in proposal["candidates"]}
                candidates = [spec for spec in candidates if spec.candidate_id in proposed_ids]
                rows = [row for row in rows if row["candidate_id"] in proposed_ids]
                fallback = proposal["suggested_action"]
                if method in {"auto", "fixed_retry"}:
                    ids = {s.candidate_id for s in candidates}
                    if "finish" in ids:
                        selected = "finish"
                    elif "pick" in ids:
                        selected = "pick"
                    elif state["consecutive_stalls"] and method == "auto":
                        selected = "stop"
                    elif state["consecutive_stalls"] >= 2 and method == "fixed_retry":
                        selected = "help" if "help" in ids else "stop"
                    else:
                        selected = "seat" if "seat" in ids else "stop"
                    why = "m0_fixed_policy"
                else:
                    selected, why = selector.choose(rows, candidates, method=method,
                                                   suggested=fallback, fallback=fallback)
            fallback_count += "fallback" in why
            decisions.append({"decision_index": index, "public_state": state, "proposal": proposal,
                              "all_registered_candidates": [s.candidate_id for s in available],
                              "ranked": rows, "choice": selected, "reason": why,
                              "remaining_wall_s": config["budget"]["wall_s"] - runtime.wall_s})
            append("decision_trace", decisions[-1])
            # Model waiting advances physics. Dispatch from a new sensor packet, never its old trajectory.
            runtime.observe()
            if proposal:
                candidate = next((item for item in proposal["candidates"] if item["candidate_id"] == selected), {})
                refs = tuple(dict.fromkeys(proposal["evidence_refs"] + candidate.get("evidence_refs", [])))
            else:
                refs = (next(event["event_id"] for event in reversed(runtime.ledger.public_context()["events"])
                             if event["kind"] == "observation"),)
            result = runtime.execute(ToolCall(f"a_{index}", selected, runtime.epoch,
                                             runtime.observation.observation_id, evidence_refs=refs))
            if result.status == "REJECTED":
                runtime.adapter.safe_hold()
                runtime.terminal = "stop"
        if runtime.terminal is None:
            adapter.safe_hold()
            runtime.terminal = "stop"
    except (Interrupted, OSError, TimeoutError) as error:
        adapter.safe_hold()
        runtime.terminal = "stop"
        failures.append({"kind": "runtime", "error": str(error)})
    finally:
        # Evaluation is performed exactly once, AFTER the online loop is terminal.
        adapter.safe_hold()
        if runtime.terminal is None:
            runtime.terminal = "stop"
        private = adapter.evaluate()
        reported = runtime.terminal == "finish"
        primary = private["seat_gt"]
        if config["primary_endpoint"] == "seat_and_registration_gt":
            primary = "FAIL" if "FAIL" in (primary, private["registration_gt"]) else (
                "PASS" if primary == private["registration_gt"] == "PASS" else "UNKNOWN")
        cost = {"wall_s": runtime.wall_s, **asdict(runtime.cost)}
        if hasattr(adapter, "force_exposure"):
            cost["contact_exposure_Ns"] = adapter.force_exposure if adapter.contact_exposure_valid else None
        cost["sim_s"] = adapter.tick * config.get("task_snapshot", {}).get("scene", {}).get("physics_dt_s", 0)
        summary = {"backend": config["backend"], "terminal": runtime.terminal,
                   "primary_gt": primary, "seat_gt": private["seat_gt"],
                   "registration_gt": private["registration_gt"], "reported_success": reported,
                   "false_finish": None if reported and primary == "UNKNOWN" else reported and primary == "FAIL",
                   "resources": asdict(runtime.used),
                   "cost": cost, "fallback_count": fallback_count, "errors": failures,
                   "physical_validation": private.get("physical_validation", False)}
        summary["help_request_sent"] = runtime.terminal == "help_requested"
        summary["help_delivery"] = "local_outbox" if runtime.help_request else None
        if runtime.help_request:
            (output / "help_request.json").write_text(json.dumps(asdict(runtime.help_request), indent=2) + "\n")
        with (output / "private_gt.jsonl").open("w") as stream:
            import os
            os.chmod(stream.fileno(), 0o600)
            stream.write(json.dumps(private, allow_nan=False) + "\n")
        (output / "decision_trace.jsonl").touch(exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if branch_plan and not scripted_smoke:
            from .calibration import record_branch
            record_branch(output, branch_plan)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harness_contract.yaml")
    parser.add_argument("--condition", choices=("nominal", "blocked", "offset", "grasp"), default="nominal")
    parser.add_argument("--method", choices=METHODS, default="proposed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-plan", type=Path, help="Offline script; record_calibration=false selects infrastructure smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["backend"] != "contract_only":
        parser.error("Use hrc_harness.isaac for a physical scene; contract config cannot authorize robot motion")
    plan = json.loads(args.branch_plan.read_text()) if args.branch_plan else None
    summary = run_episode(ContractWorld(args.condition), config, args.output, method=args.method, seed=args.seed, branch_plan=plan)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
