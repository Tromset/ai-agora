from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """One chat message, already expressed from the point of view of the model receiving it."""

    role: Role
    content: str


@dataclass
class AgentSpec:
    """A named participant: which provider/model it runs on and how it behaves."""

    name: str
    provider: str
    model: str
    system: str = ""
    temperature: float | None = None
    max_tokens: int = 1024
    # Provider-specific settings: base_url, api_key, api_key_env, timeout, ...
    options: dict[str, Any] = field(default_factory=dict)
