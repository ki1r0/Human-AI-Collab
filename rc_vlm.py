import json
import urllib.error
import urllib.request
import time

from rc_config import COSMOS_URL, COSMOS_HOST, COSMOS_PORT, COSMOS_MODEL, COSMOS_API_KEY, TIMEOUT_SEC, SYSTEM_MSG
from rc_state import STATE
from rc_log import log_info, log_line


class VlmHttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _count_images(messages) -> int:
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    count += 1
    return count


def cosmos_endpoint() -> str:
    if COSMOS_URL:
        return COSMOS_URL
    return f"http://{COSMOS_HOST}:{COSMOS_PORT}/v1/chat/completions"


def mark_cosmos_state(ok: bool, detail: str):
    if ok:
        if STATE.cosmos_state is not True:
            log_line("INFO ", f"Cosmos connected: {cosmos_endpoint()} (model: {COSMOS_MODEL})")
    else:
        if detail != STATE.cosmos_last_detail:
            log_line("WARN ", f"Cosmos connection failed: {detail}")
    STATE.cosmos_state = ok
    STATE.cosmos_last_detail = detail


def build_messages(allow_last_image: bool = True):
    msgs = []
    if SYSTEM_MSG:
        msgs.append({"role": "system", "content": SYSTEM_MSG})

    history = list(STATE.chat_history)
    last_idx = len(history) - 1
    for idx, msg in enumerate(history):
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            text_only = "\n".join([t for t in text_parts if t]).strip()
            if allow_last_image and idx == last_idx:
                msgs.append(msg)
            else:
                msgs.append({"role": msg.get("role", "user"), "content": text_only})
        else:
            msgs.append(msg)
    return msgs


def call_vlm(messages):
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        raise RuntimeError("COSMOS_MODEL is not set. Set COSMOS_MODEL env var or edit this file.")

    payload = {
        "model": COSMOS_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.4,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if COSMOS_API_KEY:
        headers["Authorization"] = f"Bearer {COSMOS_API_KEY}"

    req = urllib.request.Request(cosmos_endpoint(), data=data, headers=headers, method="POST")

    t0 = time.time()
    endpoint = cosmos_endpoint()
    num_images = _count_images(messages)

    print(f"[VLM-SIMPLE] POST {endpoint} | images={num_images} | max_tokens=256", flush=True)
    log_info(f"Sending request to Cosmos (images={num_images})")

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            status_code = resp.getcode()
            raw = resp.read().decode("utf-8")
            print(f"[VLM-SIMPLE] HTTP {status_code} | response_bytes={len(raw)}", flush=True)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        print(f"[VLM-SIMPLE] HTTP ERROR {exc.code}: {body[:500]}", flush=True)
        mark_cosmos_state(False, f"HTTP {exc.code}")
        raise VlmHttpError(exc.code, body)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[VLM-SIMPLE] CONNECTION ERROR: {exc}", flush=True)
        mark_cosmos_state(False, str(exc))
        raise

    dt = time.time() - t0
    try:
        res = json.loads(raw)
        text = str(res["choices"][0]["message"]["content"]).strip()
        usage = res.get("usage", {})
        print(f"[VLM-SIMPLE] Response in {dt:.2f}s | tokens={usage} | content_len={len(text)}", flush=True)
        print(f"[VLM-SIMPLE] Raw content: {text[:500]}", flush=True)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"[VLM-SIMPLE] PARSE ERROR: {exc} | raw={raw[:500]}", flush=True)
        mark_cosmos_state(False, f"bad response: {exc}")
        raise RuntimeError(f"Bad Cosmos response: {exc}")

    mark_cosmos_state(True, f"{dt:.2f}s")
    log_info(f"VLM replied in {dt:.2f}s")
    log_line("INFO ", f"VLM response len={len(text)}")
    if text:
        log_line("INFO ", f"VLM response head: {text[:200]}")
    return text


def test_cosmos_connection():
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        log_line("WARN ", "COSMOS_MODEL is not set. Update COSMOS_MODEL or set COSMOS_MODEL env var.")
        return False
    log_line("INFO ", f"Testing Cosmos endpoint: {cosmos_endpoint()}")
    try:
        reply = call_vlm([{"role": "user", "content": "Reply with just: OK"}])
    except Exception as exc:
        log_line("WARN ", f"Cosmos test failed: {exc}")
        return False
    log_line("INFO ", f"Cosmos OK: {reply[:120]}")
    return True
