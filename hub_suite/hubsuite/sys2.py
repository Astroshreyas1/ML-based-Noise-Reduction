"""System 2: a local LLM, off the hot path, behind an OpenAI-compatible endpoint (llama.cpp `llama-server`
or Ollama). Two jobs:

  enrich(tx_id)   schema-constrained JSON extraction of fields the rules may have missed (size, activity,
                  direction, casualties, requested support...). Every value is checked by
                  radiotext.grounded() against the transcript; ungrounded values are dropped and logged.
  ask(question)   answers the officer's questions from the situation picture only (markers + transcripts
                  passed as context), must cite transmission ids; answers without a valid citation are
                  flagged. (MCP clients get the same data through hubsuite.mcp_server.)

Models (user, 2026-09-27: target box has 48 GB VRAM, "least hallucinating"): Qwen3-32B (non-thinking) Q5_K_M
on the workstation; Qwen3-4B Q4_K_M on the laptop. Hallucination is contained structurally (grounding,
citations, no marker creation), not by trusting any model.
"""
from __future__ import annotations

import json
import re
import urllib.request

from .situation import Situation

SCHEMA = {
    "type": "object",
    "properties": {
        "sender": {"type": ["string", "null"]},
        "receiver": {"type": ["string", "null"]},
        "enemy_size": {"type": ["string", "null"]},
        "enemy_activity": {"type": ["string", "null"]},
        "direction": {"type": ["string", "null"]},
        "grid_digits": {"type": ["string", "null"]},
        "casualties": {"type": ["integer", "null"]},
        "request": {"type": ["string", "null"]},
    },
    "required": ["sender", "receiver", "enemy_size", "enemy_activity", "direction", "grid_digits", "casualties", "request"],
}
SYS = ("You extract fields from ONE military radio transmission transcript. Copy values verbatim from the "
       "transcript (numbers as digits). If a field is not stated, use null. Never guess, never infer, never add.")


class System2:
    def __init__(self, endpoint: str, sit: Situation, model: str = "local", timeout: float = 20.0):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.sit = sit
        self.model = model
        self.timeout = timeout

    def _chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 256) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "fields", "schema": schema, "strict": True}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def enrich(self, tx_id: str) -> dict:
        tx = self.sit.tx.get(tx_id)
        if tx is None or not tx.text.strip():
            return {}
        try:
            out = json.loads(self._chat([{"role": "system", "content": SYS},
                                         {"role": "user", "content": f"Transcript: {tx.text}\n/no_think"}], SCHEMA))
        except Exception as e:                                   # System 2 failing never affects System 1
            return {"error": str(e)}
        return self.sit.enrich(tx_id, out, "llm")

    def ask(self, question: str) -> dict:
        snap = self.sit.snapshot()
        ctx = {"markers": [{k: m[k] for k in ("id", "kind", "label", "grid", "tx", "source")} for m in snap["markers"]],
               "transmissions": [{"id": t["id"], "text": t["text"]} for t in snap["transmissions"][-60:]]}
        sys = ("You answer a commander's question using ONLY the situation data given. Cite the transmission ids "
               "you used in square brackets, e.g. [tx003]. If the data does not answer the question, say "
               "'Not reported.' Never guess.")
        ans = self._chat([{"role": "system", "content": sys},
                          {"role": "user", "content": f"Situation: {json.dumps(ctx)}\n\nQuestion: {question}\n/no_think"}],
                         max_tokens=300)
        ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
        cited = re.findall(r"\[(tx\d+|op\d+)\]", ans)
        valid = [c for c in cited if c in self.sit.tx]
        return {"answer": ans, "citations": valid, "unverified": not valid and "not reported" not in ans.lower()}
