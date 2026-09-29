"""Providers that drive official AI command-line tools with your existing subscription login.

No API key: `claude-code` runs the Claude Code CLI (Claude Pro/Max login) and `codex` runs the
OpenAI Codex CLI (ChatGPT login). Each call is one isolated, non-interactive run in an empty
temporary directory, with tools, MCP servers, skills, project instruction files and session
history turned off, so the model sees only the conversation aic sends it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from ..errors import ProviderError
from ..types import Message
from .base import Provider

DEFAULT_TIMEOUT = 300.0
DEFAULT_SYSTEM = "You are a helpful assistant taking part in a conversation."
REPLY_INSTRUCTION = "Write only your next message in this conversation, without any speaker tag."


def render_prompt(messages: list[Message]) -> str:
    """Flatten a chat history into one prompt, since each CLI run is a single turn."""
    if len(messages) == 1:
        return messages[0].content
    lines = ["Conversation so far (your own earlier messages are marked [You]):", ""]
    for m in messages:
        lines.append(f"[You]: {m.content}" if m.role == "assistant" else m.content)
        lines.append("")
    lines.append(REPLY_INSTRUCTION)
    return "\n".join(lines)


def _tail(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


class CLIProvider(Provider):
    """Base class: run `command` in a throwaway directory, prompt on stdin (never in argv)."""

    api_key_env = None
    default_command: ClassVar[str]
    install_hint: ClassVar[str]

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        self.command = str(self.options.get("command") or self.default_command)
        self.timeout = float(self.options.get("timeout", DEFAULT_TIMEOUT))
        self.extra_args = [str(a) for a in self.options.get("extra_args", [])]

    def api_key(self) -> str:
        return ""  # authentication is the CLI's own login

    def run(self, args: list[str], prompt: str, cwd: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.command, *args], input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=cwd, env=env, timeout=self.timeout,
            )
        except FileNotFoundError:
            raise ProviderError(f"'{self.command}' not found. {self.install_hint}") from None
        except subprocess.TimeoutExpired:
            raise ProviderError(f"'{self.command}' did not answer within {self.timeout:.0f}s.") from None


class ClaudeCodeProvider(CLIProvider):
    """Claude through the Claude Code CLI (`claude -p`), using your Claude subscription."""

    name = "claude-code"
    default_command = "claude"
    install_hint = "Install Claude Code (https://claude.com/claude-code), then run `claude` once to log in."

    def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> str:
        args = [
            "-p", "--output-format", "json",
            "--no-session-persistence",   # nothing written to ~/.claude/projects
            "--tools", "",                # no file, shell or web access
            "--strict-mcp-config",        # no MCP servers
            "--disable-slash-commands",   # no skills
            "--setting-sources", "",      # no user/project settings, hooks or CLAUDE.md
            "--system-prompt", system or DEFAULT_SYSTEM,
        ]
        if model and model != "default":
            args += ["--model", model]
        env = dict(os.environ)
        if not self.options.get("use_api_key"):
            env.pop("ANTHROPIC_API_KEY", None)  # use the subscription login, never API billing
        with tempfile.TemporaryDirectory(prefix="aic-claude-") as cwd:
            proc = self.run(args + self.extra_args, render_prompt(messages), cwd, env)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise ProviderError(
                f"claude exited with code {proc.returncode}: {_tail(proc.stderr or proc.stdout)}"
            ) from None
        if not isinstance(data, dict) or data.get("is_error") or proc.returncode != 0:
            detail = data.get("result") if isinstance(data, dict) else None
            raise ProviderError(f"claude failed: {_tail(str(detail or proc.stderr or proc.stdout))}")
        return str(data.get("result") or "").strip()


# Codex features that give the model tools or outside context; disabled when the installed
# version knows them (an unknown name makes `codex exec` fail, so we check first).
CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "memories", "browser_use",
    "computer_use", "multi_agent", "goals", "view_image", "image_generation",
)


class CodexProvider(CLIProvider):
    """ChatGPT models through the OpenAI Codex CLI (`codex exec`), using your ChatGPT login.

    By default Codex runs with a temporary CODEX_HOME that holds only a copy of your login
    (auth.json), so your ~/.codex/AGENTS.md, skills, config and history are never sent.
    A refreshed login is copied back. Set `isolate_home = false` to use ~/.codex directly
    (needed if Codex keeps credentials in the OS keyring).
    """

    name = "codex"
    default_command = "codex"
    install_hint = "Install Codex (npm i -g @openai/codex), then run `codex login` with your ChatGPT account."
    _features_cache: ClassVar[dict[str, set[str]]] = {}

    def real_home(self) -> Path:
        return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()

    def supported_features(self, env: dict[str, str]) -> set[str]:
        if self.command not in self._features_cache:
            with tempfile.TemporaryDirectory(prefix="aic-codex-") as cwd:
                proc = self.run(["features", "list"], "", cwd, env)
            self._features_cache[self.command] = {
                line.split()[0] for line in proc.stdout.splitlines() if line.strip()
            }
        return self._features_cache[self.command]

    def build_args(self, model: str, system: str, workdir: str, outfile: str, features: set[str]) -> list[str]:
        args = [
            "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--color", "never", "-C", workdir, "-o", outfile,
            "-c", "project_doc_max_bytes=0",       # no AGENTS.md from the working directory
            "-c", 'web_search="disabled"',
            "-c", f"developer_instructions={json.dumps(system or DEFAULT_SYSTEM)}",
        ]
        for feature in CODEX_DISABLED_FEATURES:
            if feature in features:
                args += ["--disable", feature]
        if model and model != "default":
            args += ["-m", model]
        return args + self.extra_args + ["-"]

    def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> str:
        env = dict(os.environ)
        if not self.options.get("use_api_key"):
            env.pop("OPENAI_API_KEY", None)  # use the ChatGPT login, never API billing
        isolate = self.options.get("isolate_home", True)
        with tempfile.TemporaryDirectory(prefix="aic-codex-") as tmp:
            workdir, home = Path(tmp, "work"), Path(tmp, "home")
            workdir.mkdir()
            home.mkdir()
            auth = self.real_home() / "auth.json"
            original = b""
            if isolate:
                if not auth.is_file():
                    raise ProviderError(
                        f"No Codex login found at {auth}. Run `codex login` with your ChatGPT account "
                        "(or set isolate_home = false if Codex stores it in your OS keyring)."
                    )
                original = auth.read_bytes()
                (home / "auth.json").write_bytes(original)
                os.chmod(home / "auth.json", 0o600)
                env["CODEX_HOME"] = str(home)
            outfile = Path(tmp, "reply.txt")
            args = self.build_args(model, system, str(workdir), str(outfile), self.supported_features(env))
            proc = self.run(args, render_prompt(messages), str(workdir), env)
            if isolate:
                self._sync_login_back(home / "auth.json", auth, original)
            reply = outfile.read_text(encoding="utf-8").strip() if outfile.exists() else ""
        if proc.returncode != 0 or not reply:
            raise ProviderError(f"codex exited with code {proc.returncode}: {_tail(proc.stderr or proc.stdout)}")
        return reply

    @staticmethod
    def _sync_login_back(copy: Path, auth: Path, original: bytes) -> None:
        """Codex may refresh the login during the run; keep ~/.codex/auth.json current."""
        if not copy.is_file():
            return
        refreshed = copy.read_bytes()
        if refreshed == original or not auth.is_file() or auth.read_bytes() != original:
            return  # unchanged, or the real file changed meanwhile (never clobber it)
        tmp = auth.with_name(".auth.json.aic-tmp")
        tmp.write_bytes(refreshed)
        os.chmod(tmp, 0o600)
        os.replace(tmp, auth)


def command_available(command: str) -> bool:
    return shutil.which(command) is not None
