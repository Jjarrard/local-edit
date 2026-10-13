"""Minimal OpenAI-compatible chat client for a local Jan server."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import requests


class JanConnectionError(RuntimeError):
    pass


class ContextTooLargeError(JanConnectionError):
    """The prompt didn't fit the model's context window.

    Carries the server's own numbers so the caller can shrink to the real
    limit and retry, instead of guessing (or dead-ending on the user).
    """

    def __init__(self, message: str, n_ctx: int | None = None, n_prompt_tokens: int | None = None):
        super().__init__(message)
        self.n_ctx = n_ctx
        self.n_prompt_tokens = n_prompt_tokens


def _parse_context_error(body: str) -> tuple[int | None, int | None]:
    """Pull (n_ctx, n_prompt_tokens) out of a llama.cpp context-overflow error.

    Falls back to a regex over the message text, because the structured
    fields aren't present in every llama.cpp build.
    """
    try:
        payload = json.loads(body)
        err = payload.get("error", payload)
        if isinstance(err, dict):
            n_ctx = err.get("n_ctx")
            n_prompt = err.get("n_prompt_tokens")
            if isinstance(n_ctx, int):
                return n_ctx, n_prompt if isinstance(n_prompt, int) else None
            if err.get("type") == "exceed_context_size_error" or "context size" in str(err.get("message", "")):
                m = re.search(r"\((\d+)\s*tokens\).*?\((\d+)\s*tokens\)", str(err.get("message", "")))
                if m:
                    return int(m.group(2)), int(m.group(1))
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    if "context size" in body or "exceed_context" in body:
        m = re.search(r"\((\d+)\s*tokens\).*?\((\d+)\s*tokens\)", body)
        if m:
            return int(m.group(2)), int(m.group(1))
    return None, None


@dataclass
class JanClient:
    base_url: str = "http://127.0.0.1:1338/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: float = 120.0

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def list_models(self) -> list[str]:
        url = f"{self.base_url.rstrip('/')}/models"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise JanConnectionError(f"could not reach Jan at {url}: {exc}") from exc
        data = resp.json()
        return [m.get("id", "?") for m in data.get("data", [])]

    def stream_chat(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from a chat completion, streaming.

        `max_tokens` and `model` override the client's defaults for this call
        only - used to route cheap phases (planning, review, command work) to
        a small fast model while code edits go to the capable one.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload, stream=True, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise JanConnectionError(
                f"could not reach Jan at {url} ({exc}). Is Jan's local API server running "
                f"and a model loaded?"
            ) from exc
        if not resp.ok:
            body = resp.text.strip()
            n_ctx, n_prompt = _parse_context_error(body)
            if n_ctx:
                raise ContextTooLargeError(
                    f"prompt was {n_prompt} tokens but the model's context is {n_ctx}",
                    n_ctx=n_ctx,
                    n_prompt_tokens=n_prompt,
                )
            raise JanConnectionError(
                f"Jan rejected the request ({resp.status_code}) at {url}: {body[:500] or '(no body)'}"
            )

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                payload_str = raw_line[len("data:") :].strip()
                if payload_str == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
        finally:
            # If the caller stops consuming early (e.g. it detected the model
            # is stuck in a repetition loop), closing the socket here signals
            # the server to stop generating instead of burning tokens/time
            # on a response nobody wants.
            resp.close()

    def chat(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Non-streaming convenience wrapper: collects the full response."""
        return "".join(self.stream_chat(messages, stop=stop, max_tokens=max_tokens, model=model))
