from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiconnect.config import (
    Config,
    default_config_paths,
    load_config,
    parse_agent_arg,
    resolve_agents,
)
from aiconnect.errors import ConfigError
from aiconnect.types import AgentSpec


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, text: str, name: str = "aic.toml") -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path


class LoadConfigTests(TempDirCase):
    def test_full_config(self) -> None:
        path = self.write(
            """
[defaults]
rounds = 4
save_dir = "out"

[agents.claude]
provider = "anthropic"
model = "claude-sonnet-5-5"
system = "You are a philosopher."
temperature = 0.8
max_tokens = 500

[agents.local]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"
timeout = 30
"""
        )
        cfg = load_config(path)
        self.assertEqual(cfg.path, path)
        self.assertEqual(cfg.defaults, {"rounds": 4, "save_dir": "out"})
        claude = cfg.agents["claude"]
        self.assertEqual(claude.name, "claude")
        self.assertEqual(claude.provider, "anthropic")
        self.assertEqual(claude.system, "You are a philosopher.")
        self.assertEqual(claude.temperature, 0.8)
        self.assertEqual(claude.max_tokens, 500)
        self.assertEqual(claude.options, {})
        local = cfg.agents["local"]
        self.assertEqual(local.max_tokens, 1024)
        self.assertIsNone(local.temperature)
        self.assertEqual(local.options, {"base_url": "http://localhost:11434", "timeout": 30})

    def test_integer_temperature_becomes_float(self) -> None:
        path = self.write('[agents.a]\nprovider="mock"\nmodel="m"\ntemperature = 1\n')
        self.assertEqual(load_config(path).agents["a"].temperature, 1.0)

    def test_system_file_relative_to_config(self) -> None:
        sub = self.dir / "conf"
        sub.mkdir()
        (sub / "persona.md").write_text("  Be terse.\n", encoding="utf-8")
        path = sub / "aic.toml"
        path.write_text('[agents.a]\nprovider="mock"\nmodel="m"\nsystem_file="persona.md"\n')
        cfg = load_config(path)
        self.assertEqual(cfg.agents["a"].system, "Be terse.")
        self.assertNotIn("system_file", cfg.agents["a"].options)

    def test_system_file_missing(self) -> None:
        path = self.write('[agents.a]\nprovider="mock"\nmodel="m"\nsystem_file="nope.md"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("nope.md", str(ctx.exception))
        self.assertIn("'a'", str(ctx.exception))

    def test_system_and_system_file_conflict(self) -> None:
        (self.dir / "p.md").write_text("x")
        path = self.write('[agents.a]\nprovider="mock"\nmodel="m"\nsystem="s"\nsystem_file="p.md"\n')
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_missing_provider_or_model(self) -> None:
        for body, key in (('model="m"', "provider"), ('provider="mock"', "model")):
            path = self.write(f"[agents.bad]\n{body}\n")
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            msg = str(ctx.exception)
            self.assertIn("bad", msg)
            self.assertIn(key, msg)
            self.assertIn(str(path), msg)

    def test_unknown_provider(self) -> None:
        path = self.write('[agents.bad]\nprovider="skynet"\nmodel="m"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        msg = str(ctx.exception)
        self.assertIn("skynet", msg)
        self.assertIn("bad", msg)
        self.assertIn("anthropic", msg)

    def test_bad_temperature(self) -> None:
        path = self.write('[agents.hot]\nprovider="mock"\nmodel="m"\ntemperature="warm"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        msg = str(ctx.exception)
        self.assertIn("hot", msg)
        self.assertIn("temperature", msg)
        self.assertIn(str(path), msg)

    def test_bad_max_tokens(self) -> None:
        path = self.write('[agents.a]\nprovider="mock"\nmodel="m"\nmax_tokens="lots"\n')
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_invalid_toml(self) -> None:
        path = self.write("[agents.a\n")
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn(str(path), str(ctx.exception))

    def test_explicit_missing_path_errors(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.dir / "missing.toml")

    def test_missing_default_paths_give_empty_config(self) -> None:
        with mock.patch("aiconnect.config.default_config_paths", return_value=[self.dir / "x.toml"]):
            cfg = load_config()
        self.assertEqual(cfg.agents, {})
        self.assertEqual(cfg.defaults, {})
        self.assertIsNone(cfg.path)

    def test_default_path_is_used_when_present(self) -> None:
        path = self.write('[agents.a]\nprovider="mock"\nmodel="m"\n')
        with mock.patch("aiconnect.config.default_config_paths", return_value=[self.dir / "no.toml", path]):
            cfg = load_config()
        self.assertEqual(cfg.path, path)
        self.assertIn("a", cfg.agents)


class DefaultPathsTests(TempDirCase):
    def test_order_and_env(self) -> None:
        with mock.patch.dict(os.environ, {"AIC_CONFIG": "/tmp/custom.toml"}):
            paths = default_config_paths()
        self.assertEqual(paths[0].name, "aic.toml")
        self.assertEqual(paths[1], Path("/tmp/custom.toml"))
        self.assertEqual(paths[-1].name, "config.toml")

    def test_without_env(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "AIC_CONFIG"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(len(default_config_paths()), 2)


class ParseAgentArgTests(unittest.TestCase):
    def test_named(self) -> None:
        spec = parse_agent_arg("claude=anthropic:claude-sonnet-5-5")
        self.assertEqual((spec.name, spec.provider, spec.model), ("claude", "anthropic", "claude-sonnet-5-5"))

    def test_unnamed_uses_model(self) -> None:
        spec = parse_agent_arg("openai:gpt-5")
        self.assertEqual((spec.name, spec.provider, spec.model), ("gpt-5", "openai", "gpt-5"))

    def test_model_with_colon(self) -> None:
        spec = parse_agent_arg("local=ollama:llama3.2:3b")
        self.assertEqual((spec.provider, spec.model), ("ollama", "llama3.2:3b"))

    def test_errors_include_example(self) -> None:
        for bad in ("justaname", "=openai:x", "openai:", ":model", "a=b"):
            with self.assertRaises(ConfigError) as ctx:
                parse_agent_arg(bad)
            self.assertIn("name=provider:model", str(ctx.exception), bad)

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            parse_agent_arg("x=skynet:v1")
        self.assertIn("skynet", str(ctx.exception))


class ResolveAgentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config(
            agents={
                "a": AgentSpec("a", "mock", "m1", system="sys", options={"replies": ["x"]}),
                "b": AgentSpec("b", "mock", "m2"),
            },
            defaults={},
            path=None,
        )

    def test_config_names_and_inline(self) -> None:
        specs = resolve_agents(self.cfg, ["a", "mock:zzz", "n=mock:q"])
        self.assertEqual([s.name for s in specs], ["a", "zzz", "n"])
        self.assertEqual(specs[0].system, "sys")

    def test_duplicates_suffixed(self) -> None:
        specs = resolve_agents(self.cfg, ["a", "a", "b", "a"])
        self.assertEqual([s.name for s in specs], ["a", "a#2", "b", "a#3"])

    def test_config_not_mutated(self) -> None:
        specs = resolve_agents(self.cfg, ["a", "a"])
        specs[0].options["extra"] = 1
        self.assertEqual(self.cfg.agents["a"].options, {"replies": ["x"]})
        self.assertEqual(self.cfg.agents["a"].name, "a")

    def test_unknown_reference(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            resolve_agents(self.cfg, ["ghost"])
        msg = str(ctx.exception)
        self.assertIn("ghost", msg)
        self.assertIn("a, b", msg)
        self.assertIn("provider:model", msg)


if __name__ == "__main__":
    unittest.main()
