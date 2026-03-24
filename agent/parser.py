import json
import re

_ROOT_KEYS = {
    "reply",
    "stm_observation",
    "action",
    "belief_update",
    "belief_state_update",
    "static_context",
    "meta",
}


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    t = t.strip("`\n ")
    if t.lower().startswith("json"):
        t = t[4:].strip()
    return t


def _repair_json_fragment(fragment: str) -> str:
    """Best-effort repair for common model formatting issues."""
    s = fragment.strip()
    # Remove trailing commas before object/array end.
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Remove a dangling key prefix like: ..., "inferre
    s = re.sub(r',\s*"[^"]*$', "", s)
    # If model output was cut inside a quoted string, close that string.
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
    if in_str:
        if s.endswith("\\"):
            s = s[:-1]
        s += '"'
    # Close missing braces/brackets if model got cut off.
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if open_brackets > 0:
        s += "]" * open_brackets
    if open_braces > 0:
        s += "}" * open_braces
    return s


def _extract_json_candidates(text: str) -> list[str]:
    """Extract balanced top-level JSON object candidates from arbitrary text."""
    candidates: list[str] = []
    t = text
    for start in range(len(t)):
        if t[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(t[start : i + 1])
                    break
    return candidates


def _is_root_payload(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    return bool(keys & _ROOT_KEYS)


def parse_json_response(text: str):
    """Parse a JSON object from model output; raises ValueError on failure."""
    text = _strip_fences(text)

    # Fast path.
    try:
        parsed = json.loads(text)
        if _is_root_payload(parsed):
            return parsed
        raise ValueError(
            f"Parsed non-root JSON fragment: keys={list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}"
        )
    except Exception:
        pass

    start = text.find("{")
    candidates: list[str] = []
    if start != -1:
        # Always try the full top-level fragment first (even if unbalanced).
        candidates.append(text[start:])

    balanced = _extract_json_candidates(text)
    balanced.sort(key=len, reverse=True)
    for candidate in balanced:
        if candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        raise ValueError("No JSON object found in model output.")

    last_err = None
    parsed_non_root = None
    for candidate in candidates:
        repaired = _repair_json_fragment(candidate)
        try:
            parsed = json.loads(repaired)
            if _is_root_payload(parsed):
                return parsed
            if parsed_non_root is None:
                parsed_non_root = parsed
        except Exception as exc:
            last_err = exc
            # If still malformed, progressively trim tail and retry.
            tail = repaired
            for _ in range(256):
                if len(tail) < 2:
                    break
                tail = tail[:-1].rstrip()
                if not tail:
                    break
                retried = _repair_json_fragment(tail)
                try:
                    parsed = json.loads(retried)
                    if _is_root_payload(parsed):
                        return parsed
                    if parsed_non_root is None:
                        parsed_non_root = parsed
                    break
                except Exception as exc2:
                    last_err = exc2

    if parsed_non_root is not None:
        raise ValueError(f"Parsed non-root JSON fragment: keys={list(parsed_non_root) if isinstance(parsed_non_root, dict) else type(parsed_non_root).__name__}")

    preview = text[:220].replace("\n", " ")
    raise ValueError(f"Unterminated JSON object in model output: {last_err}; preview={preview}")
