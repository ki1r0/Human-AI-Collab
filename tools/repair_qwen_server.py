#!/usr/bin/env python3
"""Tiny local OpenAI-compatible Qwen planner and visual-verifier endpoint.

It accepts public planner text and image blocks from the visual verifier.
It is intentionally a local experiment launcher, not a general model service;
clients must send only public observations.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI image_url blocks to Transformers image blocks."""
    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url")
                    if not isinstance(url, str) or not url:
                        raise ValueError("image_url block has no URL")
                    blocks.append({"type": "image", "url": url})
                else:
                    blocks.append(block)
            normalized.append({**message, "content": blocks})
        elif isinstance(content, str):
            normalized.append({**message, "content": [{"type": "text", "text": content}]})
        else:
            normalized.append(message)
    return normalized


class PlannerServer:
    def __init__(self, model_path: str, *, max_new_tokens: int = 128) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda:0")
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        logging.info("loaded planner model from %s", model_path)

    def complete(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            template_kwargs["enable_thinking"] = False
            inputs = self.processor.apply_chat_template(normalize_messages(messages), **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            inputs = self.processor.apply_chat_template(normalize_messages(messages), **template_kwargs)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        prompt_len = int(inputs["input_ids"].shape[1])
        text = self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip()
        return text


def serve(model_path: str, host: str, port: int, max_new_tokens: int) -> None:
    planner = PlannerServer(model_path, max_new_tokens=max_new_tokens)

    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, value: dict[str, Any]) -> None:
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write(200, {"status": "ok", "service": "repair-qwen-planner"})
            else:
                self._write(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                content = planner.complete(payload)
                self._write(200, {"id": "repair-local", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})
            except Exception as exc:  # pragma: no cover - exercised by the runtime container
                logging.exception("planner request failed")
                self._write(500, {"error": {"type": type(exc).__name__, "message": str(exc)}})

        def log_message(self, format: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    logging.info("planner server listening on %s:%s", host, port)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(args.model, args.host, args.port, args.max_new_tokens)


if __name__ == "__main__":
    main()
