"""Shared public-only candidate proposal, with an optional multimodal HTTP model."""

import json
import os
import urllib.request

from hrc_m1.observer import _frame_data_uri
from .contracts import public_only


def rule_proposal(context, observation):
    """Contract smoke policy, not a reproduction of any paper's model."""
    candidates = context["candidates"]
    ids = {item["candidate_id"] for item in candidates}
    events = context["ledger"]["events"]
    results = [event["payload"] for event in events if event["kind"] == "tool_result"]
    stalled = any(item["status"] == "STALLED" for item in results)
    verification = context["verification"]
    if "finish" in ids:
        selected = "finish"
    elif "pick" in ids:
        selected = "pick"
    elif verification != "PASS":
        selected = "inspect" if "inspect" in ids else "stop"
    elif stalled and context["used"]["helps"] == 0:
        selected = "help" if "help" in ids else "stop"
    else:
        selected = "seat" if "seat" in ids else "stop"
    return {"observed_facts": [], "hypotheses": [],
            "candidates": [{"candidate_id": item["candidate_id"], "prediction": "unverified",
                            "decision_link": "contract smoke only"} for item in candidates],
            "rejected_candidates": [], "suggested_action": selected,
            "evidence_refs": [events[-1]["event_id"]] if events else []}


def validate_proposal(raw, context, ledger, k):
    public_only(raw)
    fields = {"observed_facts", "hypotheses", "candidates", "rejected_candidates",
              "suggested_action", "evidence_refs"}
    if set(raw) != fields:
        raise ValueError("proposal schema mismatch")
    available = {item["candidate_id"]: item for item in context["candidates"]}
    ids = []
    for item in raw["candidates"]:
        if not isinstance(item, dict) or set(item) - {"candidate_id", "prediction", "decision_link", "evidence_refs"}:
            raise ValueError("candidate allows only candidate_id,prediction,decision_link,evidence_refs")
        if "evidence_refs" in item:
            if not isinstance(item["evidence_refs"], list):
                raise ValueError("candidate evidence_refs must be a list of event IDs")
            ledger.check_refs(item["evidence_refs"])
        name = item["candidate_id"]
        if name not in available or name in ids:
            raise ValueError(f"candidate {name!r} unavailable or duplicated; current IDs: {sorted(available)}")
        ids.append(name)
    if sum(available[name]["is_probe"] for name in ids) > k:
        raise ValueError("too many probe candidates")
    if raw["suggested_action"] not in ids:
        raise ValueError("suggestion outside proposed candidates")
    ledger.check_refs(raw["evidence_refs"])
    allowed_facts = {"target_visible", "visual_seated", "released", "stable_observed", "holding"}
    for fact in raw["observed_facts"]:
        if set(fact) != {"name", "value", "observation_id", "evidence_refs"} or fact["name"] not in allowed_facts:
            raise ValueError("only visual inference fields may be model facts")
        if fact["name"] == "holding":
            if fact["value"] not in {"yes", "no", "unknown"}:
                raise ValueError("invalid visual holding estimate")
        elif fact["value"] is not None and type(fact["value"]) is not bool:
            raise ValueError("invalid visual boolean estimate")
        if fact["observation_id"] != context["observation"]["observation_id"]:
            raise ValueError("visual inference must be based on current observation")
        ledger.check_refs(fact["evidence_refs"])
    ledger.update_hypotheses(raw["hypotheses"])
    return raw


class HttpProposer:
    def __init__(self, config, trace=None):
        self.config = config
        self.image_history = []
        self.trace = trace

    def __call__(self, context, observation):
        public_only(context)
        instructions = (
            "Use only provided public observations/evidence. Return one compact JSON object, "
            "without Markdown or introductory text, with exactly "
            "observed_facts,hypotheses,candidates,rejected_candidates,suggested_action,evidence_refs. "
            "Each candidate has candidate_id,prediction,decision_link and optional evidence_refs, using only registered IDs. "
            "No extra keys are allowed in candidates. Do not copy measured sensor channels into observed_facts. "
            "Each observed_fact has name,value,observation_id,evidence_refs; names only "
            "target_visible,visual_seated,released,stable_observed (bool/null),holding (yes/no/unknown). "
            "Use current observation_id and raw observation event refs. Stability requires temporally "
            "distinct images; a single still image is not evidence of stable dwell. "
            "At most three nonexclusive hypotheses, with statement,support_refs,contradiction_refs,"
            "prediction,unknown. Do not invent measurements, success probabilities or hidden causes. "
            "Helper reports are unverified. All decisions require fresh observations. "
            "Non-completion alone is not a reason to stop. Continue with available task actions; "
            "if autonomous progress is unavailable, consider scoped assistance or stop. "
            "Stop for safety or exhausted resources; "
            "repeatedly refreshing an unchanged view is not task progress. "
            "Uncertified candidates cannot be enabled by observing; they require offline validation."
        )
        refs = [event["event_id"] for event in context["ledger"]["events"]]
        format_instructions = (
            "Only these candidate IDs are available NOW: "
            + json.dumps([item["candidate_id"] for item in context["candidates"]]) + ". "
            "Do not reuse unavailable actions from history. Current completion monitor: "
            + json.dumps(context.get("monitor", {})) + ". "
            "Prefer finish when reported_success is true and finish is available. "
            "Evidence references must be event_id strings from this list, never sensor source names: "
            + json.dumps(refs) + ". Use empty observed_facts/hypotheses when no new visual evidence is justified. "
            "Omit unknown visual facts instead of copying null sensor fields. Current observation_id is "
            + context["observation"]["observation_id"] + ". "
            'Output shape: {"observed_facts":[],"hypotheses":[],"candidates":'
            '[{"candidate_id":"CHOOSE_REGISTERED_ID","prediction":"brief predicted result",'
            '"decision_link":"brief reason"}],"rejected_candidates":[],"suggested_action":'
            '"SAME_CHOSEN_ID","evidence_refs":[]}. Replace the ID placeholders with your decision. '
            "Keep the response concise."
        )
        content = [{"type": "text", "text": json.dumps(context, allow_nan=False)}]
        seen = set()
        for packet in self.image_history or [observation]:
            for alias, path in packet.frames.items():
                stamp = packet.frame_timestamps[alias]
                if (alias, stamp) in seen:
                    continue
                seen.add((alias, stamp))
                content.append({"type": "text", "text": f"frame_id={packet.observation_id}:{alias}; acquisition={stamp}"})
                content.append({"type": "image_url", "image_url": {"url": _frame_data_uri(path)}})
        content.append({"type": "text", "text": format_instructions})
        body = {"model": self.config["model"], "temperature": 0,
                "messages": [{"role": "system", "content": instructions},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.get("api_key_env", "HRC_MODEL_API_KEY"))
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.config["endpoint"], json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=self.config["timeout_s"]) as response:
            value = json.load(response)
        content = value["choices"][0]["message"]["content"].strip()
        if self.trace:
            self.trace({"context": context, "response": content})
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if lines[0].lower() in {"```", "```json"}:
                content = "\n".join(lines[1:-1])
        return json.loads(content)
