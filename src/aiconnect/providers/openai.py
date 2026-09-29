"""OpenAI Chat Completions provider and OpenAI-compatible presets."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Iterator

from .. import http
from ..errors import ProviderError
from ..types import Message
from .base import Provider


class OpenAIProvider(Provider):
    """OpenAI (or any OpenAI-compatible server via options.base_url) through /chat/completions."""

    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"
    # Newer OpenAI models reject `max_tokens`; compatible servers generally expect it.
    max_tokens_field: ClassVar[str] = "max_completion_tokens"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key()}"}

    def _payload(
        self,
        model: str,
        messages: list[Message],
        system: str,
        temperature: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat,
            self.max_tokens_field: max_tokens,
        }
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
        """Return choices[0].message.content (None becomes an empty string)."""
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(model, messages, system, temperature, max_tokens)
        data = http.post_json(url, self._headers(), payload, self.timeout)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"Unexpected response from {self.name} at {url}") from None
        return "" if content is None else str(content)

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield content deltas from the server-sent event stream."""
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(model, messages, system, temperature, max_tokens)
        payload["stream"] = True
        lines = http.post_lines(url, self._headers(), payload, self.timeout)
        for data in http.iter_sse_data(lines):
            try:
                event = json.loads(data)
                if isinstance(event, dict) and event.get("error"):
                    err = event["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise ProviderError(f"{self.name} stream error: {msg or 'unknown error'}")
                choices = event["choices"]
                if not choices:  # e.g. a trailing usage-only chunk
                    continue
                text = choices[0].get("delta", {}).get("content")
            except json.JSONDecodeError:
                raise ProviderError(f"Invalid JSON in {self.name} stream") from None
            except (KeyError, IndexError, TypeError, AttributeError):
                raise ProviderError(f"Unexpected event in {self.name} stream") from None
            if text:
                yield str(text)


class _CompatibleProvider(OpenAIProvider):
    """Base for OpenAI-compatible services that use the classic `max_tokens` field."""

    max_tokens_field = "max_tokens"


class OpenRouterProvider(_CompatibleProvider):
    name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"


class GroqProvider(_CompatibleProvider):
    name = "groq"
    api_key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"


class MistralProvider(_CompatibleProvider):
    name = "mistral"
    api_key_env = "MISTRAL_API_KEY"
    default_base_url = "https://api.mistral.ai/v1"


class DeepSeekProvider(_CompatibleProvider):
    name = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com/v1"


class XAIProvider(_CompatibleProvider):
    name = "xai"
    api_key_env = "XAI_API_KEY"
    default_base_url = "https://api.x.ai/v1"
