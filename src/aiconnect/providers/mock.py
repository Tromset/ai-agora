"""Offline provider for tests and demos: no network, deterministic output."""

from __future__ import annotations

from typing import Any, Iterator

from ..types import Message
from .base import Provider


class MockProvider(Provider):
    """Replies deterministically.

    options:
      replies: list[str]  -> returned in order, cycling (per provider instance)
      otherwise           -> "[<model>] reply #N to: <first 60 chars of last user message>"
    """

    name = "mock"

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        self.calls: list[list[Message]] = []

    def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> str:
        self.calls.append(list(messages))
        n = len(self.calls)
        replies = self.options.get("replies")
        if replies:
            return str(replies[(n - 1) % len(replies)])
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"[{model}] reply #{n} to: {last[:60]}"

    def stream(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> Iterator[str]:
        text = self.complete(model, messages, system=system, temperature=temperature, max_tokens=max_tokens)
        for word in text.split(" "):
            yield word + " "
