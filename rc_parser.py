import json
import re


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


def parse_json_response(text: str):
    """Parse a JSON object from model output; raises ValueError on failure."""
    text = _strip_fences(text)

    # Fast path.
    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = _extract_json_candidates(text)
    if not candidates:
        start = text.find("{")
        if start != -1:
            candidates = [text[start:]]

    if not candidates:
        raise ValueError("No JSON object found in model output.")

    # Prefer longer candidates first; they usually contain the full payload.
    candidates.sort(key=len, reverse=True)
    last_err = None
    for candidate in candidates:
        repaired = _repair_json_fragment(candidate)
        try:
            return json.loads(repaired)
        except Exception as exc:
            last_err = exc

    preview = text[:220].replace("\n", " ")
    raise ValueError(f"Unterminated JSON object in model output: {last_err}; preview={preview}")
