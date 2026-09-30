"""Tests for the claude-code and codex providers, using fake CLIs (no login, no network)."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from aiconnect.errors import ProviderError
from aiconnect.providers import get_provider
from aiconnect.providers.cli import (
    REPLY_INSTRUCTION, ClaudeCodeProvider, CodexProvider, render_prompt,
)
from aiconnect.types import Message

# Each fake records how it was called into $FAKE_LOG, then answers like the real CLI.
FAKE_CLAUDE = """
import json, os, sys
log = {"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": sys.stdin.read(),
       "cwd_files": os.listdir("."), "api_key": os.environ.get("ANTHROPIC_API_KEY")}
json.dump(log, open(os.environ["FAKE_LOG"], "w"))
if os.environ.get("FAKE_FAIL"):
    print(json.dumps({"is_error": True, "result": "Not logged in"})); sys.exit(1)
print(json.dumps({"is_error": False, "result": "  hello from claude  "}))
"""

FAKE_CODEX = """
import json, os, sys
args = sys.argv[1:]
if args[:2] == ["features", "list"]:
    print("shell_tool   stable  true\\nplugins   stable  true\\nmemories  stable  false")
    sys.exit(0)
home = os.environ.get("CODEX_HOME", "")
log = {"argv": args, "cwd": os.getcwd(), "stdin": sys.stdin.read(), "home": home,
       "home_files": sorted(os.listdir(home)) if home else [],
       "api_key": os.environ.get("OPENAI_API_KEY")}
json.dump(log, open(os.environ["FAKE_LOG"], "w"))
if os.environ.get("FAKE_REFRESH"):
    open(os.path.join(home, "auth.json"), "w").write('{"token": "refreshed"}')
if os.environ.get("FAKE_FAIL"):
    print("boom", file=sys.stderr); sys.exit(2)
out = args[args.index("-o") + 1]
open(out, "w").write("hello from chatgpt\\n")
"""


def make_fake(directory: Path, name: str, body: str) -> str:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class RenderPromptTests(unittest.TestCase):
    def test_single_message_is_passed_as_is(self) -> None:
        self.assertEqual(render_prompt([Message("user", "hi")]), "hi")

    def test_history_marks_own_messages(self) -> None:
        text = render_prompt([
            Message("user", "[Moderator]: topic"),
            Message("assistant", "my point"),
            Message("user", "[gpt]: rebuttal"),
        ])
        self.assertIn("[Moderator]: topic", text)
        self.assertIn("[You]: my point", text)
        self.assertIn("[gpt]: rebuttal", text)
        self.assertTrue(text.endswith(REPLY_INSTRUCTION))


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / "log.json"
        env = {"FAKE_LOG": str(self.log), "ANTHROPIC_API_KEY": "sk-secret", "OPENAI_API_KEY": "sk-secret"}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        for var in ("FAKE_FAIL", "FAKE_REFRESH"):
            os.environ.pop(var, None)
        CodexProvider._features_cache.clear()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def called(self) -> dict:
        return json.loads(self.log.read_text(encoding="utf-8"))


class ClaudeCodeTests(CLITestCase):
    def provider(self, **options) -> ClaudeCodeProvider:
        return ClaudeCodeProvider({"command": make_fake(self.dir, "claude", FAKE_CLAUDE), **options})

    def test_registered(self) -> None:
        self.assertIsInstance(get_provider("claude-code"), ClaudeCodeProvider)

    def test_isolated_run(self) -> None:
        reply = self.provider().complete("sonnet", [Message("user", "hi")], system="Be brief.")
        self.assertEqual(reply, "hello from claude")
        call = self.called()
        argv = call["argv"]
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--system-prompt") + 1], "Be brief.")
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")
        self.assertEqual(call["stdin"], "hi")            # prompt never in argv
        self.assertNotIn("hi", argv)
        self.assertEqual(call["cwd_files"], [])          # empty throwaway directory
        self.assertNotEqual(Path(call["cwd"]).resolve(), Path.cwd().resolve())
        self.assertFalse(Path(call["cwd"]).exists())     # cleaned up
        self.assertIsNone(call["api_key"])               # subscription login, not API billing

    def test_default_model_is_not_forced(self) -> None:
        self.provider().complete("default", [Message("user", "hi")])
        self.assertNotIn("--model", self.called()["argv"])

    def test_use_api_key_keeps_env(self) -> None:
        self.provider(use_api_key=True).complete("sonnet", [Message("user", "hi")])
        self.assertEqual(self.called()["api_key"], "sk-secret")

    def test_error_result(self) -> None:
        os.environ["FAKE_FAIL"] = "1"
        with self.assertRaisesRegex(ProviderError, "Not logged in"):
            self.provider().complete("sonnet", [Message("user", "hi")])

    def test_missing_binary(self) -> None:
        p = ClaudeCodeProvider({"command": str(self.dir / "nope")})
        with self.assertRaisesRegex(ProviderError, "not found.*log in"):
            p.complete("sonnet", [Message("user", "hi")])


class CodexTests(CLITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.real_home = self.dir / "real-codex-home"
        self.real_home.mkdir()
        (self.real_home / "auth.json").write_text('{"token": "original"}', encoding="utf-8")
        (self.real_home / "AGENTS.md").write_text("private notes", encoding="utf-8")
        (self.real_home / "config.toml").write_text("model = 'x'", encoding="utf-8")
        os.environ["CODEX_HOME"] = str(self.real_home)

    def provider(self, **options) -> CodexProvider:
        return CodexProvider({"command": make_fake(self.dir, "codex", FAKE_CODEX), **options})

    def test_registered(self) -> None:
        self.assertIsInstance(get_provider("codex"), CodexProvider)

    def test_isolated_run(self) -> None:
        reply = self.provider().complete("gpt-5", [Message("user", "hi")], system="Be brief.")
        self.assertEqual(reply, "hello from chatgpt")
        call = self.called()
        argv = call["argv"]
        self.assertEqual(argv[0], "exec")
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertIn("project_doc_max_bytes=0", argv)
        self.assertIn('developer_instructions="Be brief."', argv)
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5")
        self.assertEqual(argv[-1], "-")
        # Only features this codex version knows are disabled (unknown ones would crash it).
        disabled = [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]
        self.assertEqual(sorted(disabled), ["memories", "plugins", "shell_tool"])
        self.assertEqual(call["stdin"], "hi")
        # A throwaway CODEX_HOME holding only the login: no AGENTS.md, config or history.
        self.assertNotEqual(call["home"], str(self.real_home))
        self.assertEqual(call["home_files"], ["auth.json"])
        self.assertIsNone(call["api_key"])
        self.assertFalse(Path(call["home"]).exists())

    def test_refreshed_login_is_copied_back(self) -> None:
        os.environ["FAKE_REFRESH"] = "1"
        self.provider().complete("default", [Message("user", "hi")])
        self.assertEqual((self.real_home / "auth.json").read_text(), '{"token": "refreshed"}')
        self.assertNotIn("-m", self.called()["argv"])

    def test_unchanged_login_is_left_alone(self) -> None:
        before = (self.real_home / "auth.json").stat().st_mtime_ns
        self.provider().complete("default", [Message("user", "hi")])
        self.assertEqual((self.real_home / "auth.json").stat().st_mtime_ns, before)

    def test_missing_login(self) -> None:
        (self.real_home / "auth.json").unlink()
        with self.assertRaisesRegex(ProviderError, "codex login"):
            self.provider().complete("default", [Message("user", "hi")])

    def test_isolate_home_false_uses_real_home(self) -> None:
        self.provider(isolate_home=False).complete("default", [Message("user", "hi")])
        self.assertEqual(self.called()["home"], str(self.real_home))

    def test_failure(self) -> None:
        os.environ["FAKE_FAIL"] = "1"
        with self.assertRaisesRegex(ProviderError, "code 2: boom"):
            self.provider().complete("default", [Message("user", "hi")])


if __name__ == "__main__":
    unittest.main()
