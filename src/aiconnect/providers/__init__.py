"""Provider registry. Modules are imported lazily so a broken provider never breaks the CLI."""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import ConfigError
from .base import Provider

# provider name -> "module:ClassName"
REGISTRY: dict[str, str] = {
    "anthropic": "aiconnect.providers.anthropic:AnthropicProvider",
    "openai": "aiconnect.providers.openai:OpenAIProvider",
    "openrouter": "aiconnect.providers.openai:OpenRouterProvider",
    "groq": "aiconnect.providers.openai:GroqProvider",
    "mistral": "aiconnect.providers.openai:MistralProvider",
    "deepseek": "aiconnect.providers.openai:DeepSeekProvider",
    "xai": "aiconnect.providers.openai:XAIProvider",
    "gemini": "aiconnect.providers.gemini:GeminiProvider",
    "ollama": "aiconnect.providers.ollama:OllamaProvider",
    "claude-code": "aiconnect.providers.cli:ClaudeCodeProvider",
    "codex": "aiconnect.providers.cli:CodexProvider",
    "mock": "aiconnect.providers.mock:MockProvider",
}


def provider_class(name: str) -> type[Provider]:
    try:
        target = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"Unknown provider '{name}'. Known providers: {known}") from None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def get_provider(name: str, options: dict[str, Any] | None = None) -> Provider:
    return provider_class(name)(options)


__all__ = ["Provider", "REGISTRY", "provider_class", "get_provider"]
