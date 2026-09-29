"""Google Gemini provider (Generative Language API, generateContent)."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .. import http
from ..errors import ConfigError, ProviderError
from ..types import Message
from .base import Provider

FALLBACK_KEY_ENV = "GOOGLE_API_KEY"


class GeminiProvider(Provider):
    name = "gemini"
    api_key_env = "GEMINI_API_KEY"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def api_key(self) -> str:
        """options.api_key > env[options.api_key_env] > $GEMINI_API_KEY > $GOOGLE_API_KEY."""
        if self.options.get("api_key") or self.options.get("api_key_env"):
            return super().api_key()
        for env in (self.api_key_env, FALLBACK_KEY_ENV):
            if env and os.environ.get(env):
                return os.environ[env]
        raise ConfigError(
            f"Provider '{self.name}' needs an API key: set ${self.api_key_env} "
            f"(or ${FALLBACK_KEY_ENV}) or 'api_key_env' in your config."
        )

    # -- request building -------------------------------------------------

    def _url(self, model: str, method: str) -> str:
        model = model[len("models/") :] if model.startswith("models/") else model
        return f"{self.base_url}/models/{model}:{method}"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key()}

    @staticmethod
    def _payload(
        messages: list[Message], system: str, temperature: float | None, max_tokens: int
    ) -> dict[str, Any]:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            config["temperature"] = temperature
        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    # -- response parsing -------------------------------------------------

    @staticmethod
    def _extract(data: Any) -> tuple[str, str]:
        """Return (text, finishReason) from one response / stream chunk."""
        try:
            if not isinstance(data, dict):
                raise TypeError("not an object")
            if data.get("error"):
                err = data["error"]
                msg = err.get("message") if isinstance(err, dict) else err
                raise ProviderError(f"Gemini error: {msg}")
            candidates = data.get("candidates")
            if not candidates:
                reason = (data.get("promptFeedback") or {}).get("blockReason")
                if reason:
                    raise ProviderError(f"Gemini blocked the prompt (blockReason: {reason})")
                if "candidates" in data or "promptFeedback" in data or "usageMetadata" in data:
                    return "", ""  # e.g. a trailing metadata-only stream chunk
                raise KeyError("candidates")
            candidate = candidates[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)
            return text, str(candidate.get("finishReason") or "")
        except (KeyError, IndexError, TypeError, AttributeError) as err:
            raise ProviderError(
                f"Unexpected response from Gemini ({type(err).__name__}: {err})"
            ) from err

    # -- Provider API -----------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        data = http.post_json(
            self._url(model, "generateContent"),
            self._headers(),
            self._payload(messages, system, temperature, max_tokens),
            timeout=self.timeout,
        )
        text, finish = self._extract(data)
        if not text and finish and finish != "STOP":
            raise ProviderError(f"Gemini returned no text (finishReason: {finish})")
        return text

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        lines = http.post_lines(
            self._url(model, "streamGenerateContent") + "?alt=sse",
            self._headers(),
            self._payload(messages, system, temperature, max_tokens),
            timeout=self.timeout,
        )
        for raw in http.iter_sse_data(lines):
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError as err:
                raise ProviderError(f"Invalid JSON in Gemini stream: {err}") from err
            text, _finish = self._extract(chunk)
            if text:
                yield text
