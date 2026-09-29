from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator

from ..errors import ConfigError
from ..types import Message


class Provider(ABC):
    """Adapter from the neutral Message list to one vendor API.

    `messages` never contains role "system": the system prompt is passed separately.
    Messages are expected to alternate user/assistant, starting with "user".
    """

    name: ClassVar[str]
    # Env var holding the API key; None for providers that need no key (ollama, mock).
    api_key_env: ClassVar[str | None] = None
    default_base_url: ClassVar[str] = ""

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = dict(options or {})
        self.base_url = str(self.options.get("base_url") or self.default_base_url).rstrip("/")
        self.timeout = float(self.options.get("timeout", 120))

    def api_key(self) -> str:
        """Resolve the API key: options.api_key > env[options.api_key_env] > env[cls.api_key_env]."""
        if self.options.get("api_key"):
            return str(self.options["api_key"])
        env = self.options.get("api_key_env") or self.api_key_env
        if env and os.environ.get(env):
            return os.environ[env]
        raise ConfigError(
            f"Provider '{self.name}' needs an API key: set ${env} or 'api_key_env' in your config."
        )

    @abstractmethod
    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Return the full assistant reply as text."""

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield the reply in chunks. Default: one chunk from complete()."""
        yield self.complete(
            model, messages, system=system, temperature=temperature, max_tokens=max_tokens
        )
