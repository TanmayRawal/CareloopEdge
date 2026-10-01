"""Client for a local llama.cpp server (OpenAI-compatible /v1/chat/completions). Fully offline.

Start on the Pi (systemd) or PC:
  llama-server -m Qwen3-1.7B-Q4_K_M.gguf -c 4096 -t 4 --host 127.0.0.1 --port 8081
Qwen3's "thinking" mode is disabled per request (chat_template_kwargs) for short, fast answers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LocalLLM:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base = (base_url or os.environ.get("NEUROEDGE_LLM_URL", "http://127.0.0.1:8081")).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat(self, messages: list[dict], max_tokens: int = 350, temperature: float = 0.2,
             json_schema: dict | None = None) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if json_schema is not None:   # grammar-constrained decoding: output is guaranteed to parse
            payload["response_format"] = {"type": "json_object", "schema": json_schema}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        text = out["choices"][0]["message"]["content"] or ""
        # strip any residual <think> block a model might emit
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()
