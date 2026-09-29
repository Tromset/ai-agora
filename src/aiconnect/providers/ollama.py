"""Ollama provider (local models via the /api/chat endpoint)."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .. import http
from ..errors import ProviderError
from ..types import Message
from .base import Provider

LOCAL_URL = "http://localhost:11434"


def _host_from_env() -> str:
    """Base URL from $OLLAMA_HOST (which may lack a scheme, e.g. '127.0.0.1:11434')."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return LOCAL_URL
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


class OllamaProvider(Provider):
    name = "ollama"
    api_key_env = None
    default_base_url = LOCAL_URL

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        # Computed per instance (not at import) so $OLLAMA_HOST changes are honoured.
        if not self.options.get("base_url"):
            self.base_url = _host_from_env()

    def api_key(self) -> str:
        return ""  # Ollama needs no key

    @staticmethod
    def _payload(
        model: str,
        messages: list[Message],
        system: str,
        temperature: float | None,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages if m.role != "system")
        options: dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        return {"model": model, "messages": chat, "stream": stream, "options": options}

    @staticmethod
    def _reraise(err: ProviderError, model: str) -> None:
        """Re-raise `err`, adding an actionable hint for the common local failures."""
        text = str(err)
        if "Cannot reach" in text:
            raise ProviderError(f"{text}. Is Ollama running? (ollama serve)") from err
        if "404" in text and "not found" in text.lower():
            raise ProviderError(f"{text}. Try: ollama pull {model}") from err
        raise err

    @staticmethod
    def _content(data: Any) -> str:
        """Extract message.content from one response object / stream line."""
        try:
            if not isinstance(data, dict):
                raise TypeError("not an object")
            if data.get("error"):
                raise ProviderError(f"Ollama error: {data['error']}")
            return str(data["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as err:
            raise ProviderError(
                f"Unexpected response from Ollama ({type(err).__name__}: {err})"
            ) from err

    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        try:
            data = http.post_json(
                f"{self.base_url}/api/chat",
                {},
                self._payload(model, messages, system, temperature, max_tokens, False),
                timeout=self.timeout,
            )
        except ProviderError as err:
            self._reraise(err, model)
        return self._content(data)

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
            f"{self.base_url}/api/chat",
            {},
            self._payload(model, messages, system, temperature, max_tokens, True),
            timeout=self.timeout,
        )
        try:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    raise ProviderError(f"Invalid JSON in Ollama stream: {err}") from err
                if isinstance(obj, dict) and obj.get("done") and not obj.get("error"):
                    # Final object: usually empty content, but yield any that is present.
                    text = str((obj.get("message") or {}).get("content") or "")
                    if text:
                        yield text
                    return
                text = self._content(obj)
                if text:
                    yield text
        except ProviderError as err:
            self._reraise(err, model)
