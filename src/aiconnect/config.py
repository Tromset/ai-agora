"""Configuration: TOML agent definitions plus inline `name=provider:model` agent arguments."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .providers import REGISTRY
from .types import AgentSpec

# Keys that map to AgentSpec fields; every other key in an agent table goes to options.
_KNOWN_KEYS = {"provider", "model", "system", "system_file", "temperature", "max_tokens"}

INLINE_EXAMPLE = "claude=anthropic:claude-sonnet-5-5 or ollama:llama3.2"


@dataclass
class Config:
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def default_config_paths() -> list[Path]:
    """Search order: ./aic.toml, $AIC_CONFIG, ~/.config/aic/config.toml."""
    paths = [Path.cwd() / "aic.toml"]
    env = os.environ.get("AIC_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path.home() / ".config" / "aic" / "config.toml")
    return paths


def load_config(path: str | Path | None = None) -> Config:
    """Load a config file.

    An explicit `path` must exist. Without one, the first existing default path is used;
    when none exists an empty Config is returned.
    """
    if path is not None:
        target = Path(path).expanduser()
        if not target.is_file():
            raise ConfigError(f"Config file not found: {target}")
    else:
        target = next((p for p in default_config_paths() if p.is_file()), None)
        if target is None:
            return Config()

    try:
        with target.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {target}: {exc}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {target}: {exc}") from None

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError(f"[defaults] must be a table in {target}")
    agents_table = data.get("agents", {})
    if not isinstance(agents_table, dict):
        raise ConfigError(f"[agents] must be a table in {target}")

    agents: dict[str, AgentSpec] = {}
    for name, table in agents_table.items():
        agents[name] = _parse_agent_table(name, table, target)
    return Config(agents=agents, defaults=dict(defaults), path=target)


def _parse_agent_table(name: str, table: Any, path: Path) -> AgentSpec:
    where = f"agent '{name}' in {path}"
    if not isinstance(table, dict):
        raise ConfigError(f"{where}: expected a table, e.g. [agents.{name}]")

    for key in ("provider", "model"):
        value = table.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where}: missing required key '{key}'")
    provider = table["provider"].strip()
    if provider not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"{where}: unknown provider '{provider}'. Known providers: {known}")

    if "system" in table and "system_file" in table:
        raise ConfigError(f"{where}: use either 'system' or 'system_file', not both")
    system = table.get("system", "")
    if not isinstance(system, str):
        raise ConfigError(f"{where}: 'system' must be a string")
    if "system_file" in table:
        rel = table["system_file"]
        if not isinstance(rel, str):
            raise ConfigError(f"{where}: 'system_file' must be a string path")
        file_path = Path(rel).expanduser()
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        try:
            system = file_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"{where}: cannot read system_file {file_path}: {exc}") from None

    temperature = table.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ConfigError(f"{where}: 'temperature' must be a number, got {temperature!r}")
        temperature = float(temperature)

    max_tokens = table.get("max_tokens", 1024)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigError(f"{where}: 'max_tokens' must be a positive integer, got {max_tokens!r}")

    options = {k: v for k, v in table.items() if k not in _KNOWN_KEYS}
    return AgentSpec(
        name=name,
        provider=provider,
        model=table["model"].strip(),
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        options=options,
    )


def parse_agent_arg(value: str) -> AgentSpec:
    """Parse "name=provider:model" or "provider:model" (the name then defaults to the model)."""
    text = value.strip()
    problem = f"Invalid agent '{value}'. Use name=provider:model or provider:model, e.g. {INLINE_EXAMPLE}"
    name = ""
    rest = text
    if "=" in text:
        name, rest = (part.strip() for part in text.split("=", 1))
        if not name:
            raise ConfigError(problem)
    if ":" not in rest:
        raise ConfigError(problem)
    provider, model = (part.strip() for part in rest.split(":", 1))
    if not provider or not model:
        raise ConfigError(problem)
    if provider not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"Unknown provider '{provider}' in '{value}'. Known providers: {known}")
    return AgentSpec(name=name or model, provider=provider, model=model)


def resolve_agents(config: Config, refs: list[str]) -> list[AgentSpec]:
    """Turn agent references (config names or inline specs) into AgentSpecs with unique names."""
    resolved: list[AgentSpec] = []
    seen: dict[str, int] = {}
    for ref in refs:
        if ref in config.agents:
            spec = config.agents[ref]
            spec = dataclasses.replace(spec, options=dict(spec.options))
        elif ":" in ref or "=" in ref:
            spec = parse_agent_arg(ref)
        else:
            configured = ", ".join(sorted(config.agents)) or "none"
            raise ConfigError(
                f"Unknown agent '{ref}'. Configured agents: {configured}. "
                f"Or give one inline as name=provider:model or provider:model, e.g. {INLINE_EXAMPLE}"
            )
        count = seen.get(spec.name, 0) + 1
        seen[spec.name] = count
        if count > 1:
            spec = dataclasses.replace(spec, name=f"{spec.name}#{count}")
        resolved.append(spec)
    return resolved
