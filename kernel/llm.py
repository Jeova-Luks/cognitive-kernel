"""Minimal client for an OpenAI-compatible local server (LM Studio, Ollama, llama.cpp).

Stdlib only. Defaults target LM Studio on localhost:1234 with Llama 3.1 8B
Instruct, which stands in for the (not yet trained) 100M base model so the
Phase 2 pipeline can run end to end. Override with environment variables
CK_LLM_BASE_URL, CK_LLM_MODEL and CK_EMBED_MODEL, or constructor arguments.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"


class LLMError(RuntimeError):
    """The local LLM server could not be reached or returned an error."""


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LocalLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 embed_model: str | None = None, timeout: float = 600.0):
        self.base_url = (base_url or os.environ.get("CK_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("CK_LLM_MODEL") or DEFAULT_MODEL
        self.embed_model = embed_model or os.environ.get("CK_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.timeout = timeout
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        last = None
        for wait in (0, 2, 10, 30):          # retry transient failures on long runs
            time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                last = LLMError(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
                if e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = LLMError(f"cannot reach {self.base_url} ({e}). Is the LM Studio server "
                                f"running? Try: lms server start && lms load {self.model}")
        raise last

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=10) as resp:
                return [m["id"] for m in json.load(resp)["data"]]
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LLMError(f"cannot reach {self.base_url} ({e}). Is the LM Studio server running?")

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 512, seed: int | None = 0) -> ChatResult:
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        t0 = time.perf_counter()
        data = self._post("/chat/completions", body)
        dt = time.perf_counter() - t0
        usage = data.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.calls += 1
        return ChatResult(data["choices"][0]["message"]["content"] or "", pt, ct, dt)

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.embed_model, "input": texts})
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]
