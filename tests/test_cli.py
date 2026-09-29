from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiconnect import __version__
from aiconnect.cli import agent_color, main
from aiconnect.config import load_config
from aiconnect.errors import ProviderError
from aiconnect.orchestrator import Agent

CONFIG = """
[defaults]
rounds = 2

[agents.alice]
provider = "mock"
model = "alice-1"
system = "You are Alice, a cheerful botanist."
replies = ["Ferns are ancient.", "Moss is underrated.", "Seeds are tiny promises."]

[agents.bob]
provider = "mock"
model = "bob-1"
replies = ["Bricks last longer.", "Concrete is honest."]

[agents.judge]
provider = "mock"
model = "judge-1"
replies = ["Alice wins on charm."]
"""


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.config = self.dir / "aic.toml"
        self.config.write_text(CONFIG, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv: str, config: bool = True) -> tuple[int, str, str]:
        args = list(argv)
        if config:
            args = ["--config", str(self.config), *args]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()


class ModeTests(CLITestCase):
    def test_ask(self) -> None:
        code, out, err = self.run_cli("ask", "alice", "What", "grows?")
        self.assertEqual(code, 0, err)
        self.assertIn("alice (mock:alice-1)", out)
        self.assertIn("Ferns are ancient.", out)
        self.assertIn("What grows?", out)

    def test_chat(self) -> None:
        code, out, err = self.run_cli("chat", "alice", "bob", "--topic", "gardens")
        self.assertEqual(code, 0, err)
        self.assertIn("Topic: gardens", out)
        self.assertEqual(out.count("alice (mock:alice-1)"), 2)  # defaults.rounds = 2
        self.assertEqual(out.count("bob (mock:bob-1)"), 2)
        self.assertLess(out.index("Ferns are ancient."), out.index("Bricks last longer."))
        self.assertIn("Moss is underrated.", out)

    def test_chat_rounds_flag_and_stop(self) -> None:
        code, out, _ = self.run_cli("chat", "alice", "bob", "--topic", "t", "--rounds", "1")
        self.assertEqual((code, out.count("bob (mock:bob-1)")), (0, 1))
        code, out, _ = self.run_cli("chat", "alice", "bob", "--topic", "t", "--stop", "ancient")
        self.assertEqual(code, 0)
        self.assertNotIn("Bricks", out)

    def test_relay(self) -> None:
        code, out, err = self.run_cli("relay", "alice", "bob", "--prompt", "seed words")
        self.assertEqual(code, 0, err)
        self.assertIn("Prompt: seed words", out)
        self.assertIn("Ferns are ancient.", out)
        self.assertIn("Bricks last longer.", out)

    def test_debate_with_judge(self) -> None:
        code, out, err = self.run_cli(
            "debate", "alice", "bob", "--motion", "Gardens beat cities", "--judge", "judge"
        )
        self.assertEqual(code, 0, err)
        self.assertIn("Motion: Gardens beat cities", out)
        self.assertIn("JUDGE", out)
        self.assertLess(out.index("Concrete is honest."), out.index("Alice wins on charm."))

    def test_debate_judge_may_also_debate(self) -> None:
        code, out, err = self.run_cli(
            "debate", "alice", "bob", "--motion", "m", "--rounds", "1", "--judge", "alice"
        )
        self.assertEqual(code, 0, err)
        self.assertIn("JUDGE", out)
        self.assertIn("alice#2", out)

    def test_panel_parallel_and_sequential(self) -> None:
        for extra in ([], ["--sequential"]):
            code, out, err = self.run_cli(
                "panel", "alice", "bob", "--question", "best building material?",
                "--synthesizer", "judge", *extra,
            )
            self.assertEqual(code, 0, err)
            self.assertIn("Question: best building material?", out)
            self.assertIn("Ferns are ancient.", out)
            self.assertIn("Bricks last longer.", out)
            self.assertIn("SYNTHESIS", out)
            self.assertIn("Alice wins on charm.", out)

    def test_inline_agents(self) -> None:
        code, out, err = self.run_cli(
            "chat", "x=mock:xm", "mock:ym", "--topic", "t", "--rounds", "1", config=False
        )
        self.assertEqual(code, 0, err)
        self.assertIn("x (mock:xm)", out)
        self.assertIn("ym (mock:ym)", out)

    def test_flags_after_command(self) -> None:
        code, out, _ = self.run_cli("chat", "alice", "bob", "--topic", "t", "--quiet", "--no-color")
        self.assertEqual(code, 0)
        self.assertNotIn("Topic:", out)

    def test_quiet_hides_seed(self) -> None:
        _, out, _ = self.run_cli("--quiet", "ask", "alice", "hidden question")
        self.assertNotIn("hidden question", out)
        self.assertIn("Ferns are ancient.", out)

    def test_no_stream_config_and_flag(self) -> None:
        code, out, _ = self.run_cli("--no-stream", "ask", "alice", "q")
        self.assertEqual(code, 0)
        self.assertIn("Ferns are ancient.", out)
        cfg = self.dir / "nostream.toml"
        cfg.write_text(CONFIG.replace("rounds = 2", "rounds = 2\nstream = false"), encoding="utf-8")
        self.assertFalse(load_config(cfg).defaults["stream"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--config", str(cfg), "ask", "alice", "q"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("Ferns are ancient.", out.getvalue())


class ColorTests(CLITestCase):
    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    def run_tty(self, *argv: str) -> str:
        out = self.TTY()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            main(["--config", str(self.config), *argv])
        return out.getvalue()

    def test_plain_when_not_a_tty(self) -> None:
        _, out, _ = self.run_cli("ask", "alice", "q")
        self.assertNotIn("\x1b[", out)

    def test_color_on_tty(self) -> None:
        self.assertIn("\x1b[", self.run_tty("ask", "alice", "q"))

    def test_no_color_flag_and_env(self) -> None:
        self.assertNotIn("\x1b[", self.run_tty("--no-color", "ask", "alice", "q"))
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertNotIn("\x1b[", self.run_tty("ask", "alice", "q"))

    def test_agent_color_is_stable(self) -> None:
        self.assertEqual(agent_color("alice"), agent_color("alice"))


class SaveTests(CLITestCase):
    def test_save_json_roundtrips_through_replay(self) -> None:
        path = self.dir / "sub" / "run.json"
        code, out, err = self.run_cli("--save", str(path), "chat", "alice", "bob", "--topic", "t")
        self.assertEqual(code, 0, err)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((data["mode"], data["topic"]), ("chat", "t"))
        self.assertEqual(data["turns"][0]["role"], "seed")
        self.assertEqual(len(data["turns"]), 5)

        code, replayed, err = self.run_cli("replay", str(path), config=False)
        self.assertEqual(code, 0, err)
        self.assertIn("Topic: t", replayed)
        self.assertIn("alice (mock:alice-1)", replayed)
        self.assertIn("Concrete is honest.", replayed)

    def test_save_markdown(self) -> None:
        path = self.dir / "run.md"
        code, _, err = self.run_cli("--save", str(path), "ask", "alice", "hello")
        self.assertEqual(code, 0, err)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# aic ask"))
        self.assertIn("Ferns are ancient.", text)

    def test_auto_save_dir(self) -> None:
        save_dir = self.dir / "auto"
        self.config.write_text(CONFIG.replace("rounds = 2", f'rounds = 2\nsave_dir = "{save_dir}"'), encoding="utf-8")
        code, _, err = self.run_cli("relay", "alice", "bob", "--prompt", "p")
        self.assertEqual(code, 0, err)
        files = list(save_dir.glob("relay-*.md"))
        self.assertEqual(len(files), 1)
        self.assertIn("Bricks last longer.", files[0].read_text(encoding="utf-8"))

    def test_explicit_save_overrides_auto_save(self) -> None:
        save_dir = self.dir / "auto"
        self.config.write_text(CONFIG.replace("rounds = 2", f'rounds = 2\nsave_dir = "{save_dir}"'), encoding="utf-8")
        path = self.dir / "explicit.md"
        self.run_cli("--save", str(path), "ask", "alice", "q")
        self.assertTrue(path.exists())
        self.assertFalse(save_dir.exists())

    def test_provider_error_saves_partial_transcript(self) -> None:
        real_respond = Agent.respond
        calls = {"n": 0}

        def flaky(self: Agent, messages, on_token=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 2:
                raise ProviderError("boom: rate limited")
            return real_respond(self, messages, on_token)

        path = self.dir / "partial.json"
        with mock.patch.object(Agent, "respond", flaky):
            code, out, err = self.run_cli("--save", str(path), "chat", "alice", "bob", "--topic", "t")
        self.assertEqual(code, 1)
        self.assertIn("error: boom: rate limited", err)
        self.assertNotIn("Traceback", err)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([t["speaker"] for t in data["turns"]], ["user", "alice"])

    def test_keyboard_interrupt_returns_130(self) -> None:
        def interrupt(self: Agent, messages, on_token=None):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        with mock.patch.object(Agent, "respond", interrupt):
            code, _, err = self.run_cli("ask", "alice", "q")
        self.assertEqual(code, 130)
        self.assertNotIn("Traceback", err)


class ErrorTests(CLITestCase):
    def test_unknown_agent(self) -> None:
        code, out, err = self.run_cli("chat", "alice", "ghost", "--topic", "t")
        self.assertEqual(code, 1)
        self.assertIn("error: Unknown agent 'ghost'", err)
        self.assertIn("alice, bob, judge", err)
        self.assertNotIn("Traceback", err)

    def test_unknown_provider_inline(self) -> None:
        code, _, err = self.run_cli("ask", "x=skynet:v1", "hi")
        self.assertEqual(code, 1)
        self.assertIn("error: Unknown provider 'skynet'", err)

    def test_unknown_provider_in_config(self) -> None:
        self.config.write_text('[agents.bad]\nprovider = "skynet"\nmodel = "m"\n', encoding="utf-8")
        code, _, err = self.run_cli("agents")
        self.assertEqual(code, 1)
        self.assertIn("skynet", err)
        self.assertIn("bad", err)

    def test_missing_explicit_config(self) -> None:
        code, _, err = self.run_cli("--config", str(self.dir / "nope.toml"), "agents", config=False)
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_bad_rounds(self) -> None:
        code, _, err = self.run_cli("chat", "alice", "bob", "--topic", "t", "--rounds", "0")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_missing_required_flag_is_usage_error(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--config", str(self.config), "chat", "alice", "bob"])
        self.assertEqual(code, 2)

    def test_replay_errors(self) -> None:
        code, _, err = self.run_cli("replay", str(self.dir / "missing.json"), config=False)
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        bad = self.dir / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        code, _, err = self.run_cli("replay", str(bad), config=False)
        self.assertEqual(code, 1)
        self.assertIn("not a valid aic JSON transcript", err)


class InfoCommandTests(CLITestCase):
    def test_version(self) -> None:
        code, out, _ = self.run_cli("--version", config=False)
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_no_command_prints_help(self) -> None:
        code, out, _ = self.run_cli(config=False)
        self.assertEqual(code, 0)
        self.assertIn("usage: aic", out)

    def test_agents(self) -> None:
        code, out, _ = self.run_cli("agents")
        self.assertEqual(code, 0)
        self.assertIn("alice", out)
        self.assertIn("mock:alice-1", out)
        self.assertIn("cheerful botanist", out)

    def test_agents_empty(self) -> None:
        self.config.write_text("", encoding="utf-8")
        code, out, _ = self.run_cli("agents")
        self.assertEqual(code, 0)
        self.assertIn("aic init", out)

    def test_providers_status_never_prints_key(self) -> None:
        secret = "sk-super-secret-value"
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}):
            code, out, err = self.run_cli("providers", config=False)
        self.assertEqual(code, 0, err)
        self.assertNotIn(secret, out + err)
        lines = {line.split()[0]: line for line in out.splitlines()[1:]}
        self.assertIn("ANTHROPIC_API_KEY", lines["anthropic"])
        self.assertTrue(lines["anthropic"].rstrip().endswith("key set"))
        self.assertTrue(lines["openai"].rstrip().endswith("missing"))
        self.assertTrue(lines["ollama"].rstrip().endswith("no key needed"))
        self.assertTrue(lines["mock"].rstrip().endswith("no key needed"))
        self.assertIn("gemini", lines)

    def test_init_writes_valid_config_and_refuses_overwrite(self) -> None:
        target = self.dir / "new" / "aic.toml"
        code, out, err = self.run_cli("init", str(target), config=False)
        self.assertEqual(code, 0, err)
        self.assertIn(str(target), out)
        text = target.read_text(encoding="utf-8")
        self.assertIn("#", text)
        cfg = load_config(target)
        self.assertIn("claude", cfg.agents)
        self.assertIn("local", cfg.agents)

        target.write_text("# mine\n", encoding="utf-8")
        code, _, err = self.run_cli("init", str(target), config=False)
        self.assertEqual(code, 1)
        self.assertIn("--force", err)
        self.assertEqual(target.read_text(encoding="utf-8"), "# mine\n")

        code, _, _ = self.run_cli("init", str(target), "--force", config=False)
        self.assertEqual(code, 0)
        self.assertIn("[agents.claude]", target.read_text(encoding="utf-8"))


    def test_init_local_writes_ollama_only_config(self) -> None:
        target = self.dir / "local.toml"
        code, out, err = self.run_cli("init", str(target), "--local", config=False)
        self.assertEqual(code, 0, err)
        self.assertIn("ollama pull", out)
        cfg = load_config(target)
        self.assertEqual(set(cfg.agents), {"llama", "qwen", "gemma"})
        self.assertTrue(all(a.provider == "ollama" for a in cfg.agents.values()))

class ExamplesTests(unittest.TestCase):
    def test_example_configs_load(self) -> None:
        root = Path(__file__).resolve().parent.parent / "examples"
        full = load_config(root / "aic.toml")
        self.assertEqual({"claude", "gpt", "gemini", "local"} <= set(full.agents), True)
        demo = load_config(root / "offline-demo.toml")
        self.assertTrue(demo.agents)
        self.assertTrue(all(a.provider == "mock" for a in demo.agents.values()))
        local = load_config(root / "local.toml")
        self.assertTrue(local.agents)
        self.assertTrue(all(a.provider == "ollama" for a in local.agents.values()))

    def test_offline_demo_runs(self) -> None:
        root = Path(__file__).resolve().parent.parent / "examples"
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--config", str(root / "offline-demo.toml"), "chat", "captain", "robot",
                         "--topic", "hello", "--rounds", "2"])
        self.assertEqual(code, 0)
        self.assertIn("captain (mock:captain-1)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
