"""Anthropic Messages API provider."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .. import http
from ..errors import ProviderError
from ..types import Message
from .base import Provider

API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    """Claude models through POST {base}/messages."""

    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key(), "anthropic-version": API_VERSION}

    def _payload(
        self,
        model: str,
        messages: list[Message],
        system: str,
        temperature: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Return the concatenated text blocks of the reply."""
        url = f"{self.base_url}/messages"
        payload = self._payload(model, messages, system, temperature, max_tokens)
        data = http.post_json(url, self._headers(), payload, self.timeout)
        try:
            blocks = data["content"]
            return "".join(
                str(b.get("text", "")) for b in blocks if b.get("type") == "text"
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError(f"Unexpected response from Anthropic at {url}") from None

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield text deltas from the server-sent event stream."""
        url = f"{self.base_url}/messages"
        payload = self._payload(model, messages, system, temperature, max_tokens)
        payload["stream"] = True
        lines = http.post_lines(url, self._headers(), payload, self.timeout)
        for data in http.iter_sse_data(lines):
            try:
                event = json.loads(data)
                kind = event.get("type")
                if kind == "error":
                    err = event.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise ProviderError(f"Anthropic stream error: {msg or 'unknown error'}")
                if kind != "content_block_delta":
                    continue
                delta = event["delta"]
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield str(delta["text"])
            except json.JSONDecodeError:
                raise ProviderError("Invalid JSON in Anthropic stream") from None
            except (KeyError, IndexError, TypeError, AttributeError):
                raise ProviderError("Unexpected event in Anthropic stream") from None
