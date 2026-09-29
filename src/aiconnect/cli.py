"""Command line interface: `aic ask | chat | relay | debate | panel | agents | providers | init | replay`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

from . import __version__
from .config import Config, load_config, resolve_agents
from .errors import AICError, ConfigError, ProviderError
from .orchestrator import Agent, run_chat, run_debate, run_panel, run_relay
from .providers import REGISTRY, provider_class
from .transcript import Transcript, Turn
from .types import AgentSpec, Message

DEFAULT_CHAT_ROUNDS = 3
DEFAULT_DEBATE_ROUNDS = 2

STARTER_CONFIG = '''\
# aic configuration. Save as ./aic.toml or ~/.config/aic/config.toml (or pass --config PATH).
# API keys are read from environment variables only; run `aic providers` to see which are set.
#
# Try it:
#   aic chat claude gpt --topic "Is free will an illusion?" --rounds 3
#   aic debate claude gpt --motion "Tabs beat spaces" --judge gemini
#   aic panel claude gpt gemini local --question "Best first programming language?" --synthesizer claude
#   aic ask claude "Explain the halting problem in two sentences"

[defaults]
rounds = 3                    # default for --rounds
stream = true                 # false = print whole turns instead of live tokens
# save_dir = "transcripts"    # auto-save a Markdown transcript of every run here

[agents.claude]
provider = "anthropic"        # required. Key: ANTHROPIC_API_KEY
model = "claude-sonnet-5-5"   # required
system = "You are Claude, a curious philosopher. Answer in a few sentences and ask a sharp follow-up question."
temperature = 0.8             # optional
max_tokens = 600              # optional, default 1024

[agents.gpt]
provider = "openai"           # Key: OPENAI_API_KEY
model = "gpt-5"
system = "You are GPT, a pragmatic engineer. Be concrete, challenge vague claims, keep replies short."

[agents.gemini]
provider = "gemini"           # Key: GEMINI_API_KEY
model = "gemini-2.5-flash"
system = "You are Gemini, a witty science communicator who loves analogies."

[agents.local]
provider = "ollama"           # No key; needs `ollama serve` running
model = "llama3.2"
system = "You are a laid-back local model. Be brief and honest about what you don't know."
base_url = "http://localhost:11434"   # any extra key is passed to the provider as an option

# Other providers: openrouter, groq, mistral, deepseek, xai (see `aic providers`).
# Long personas can live in a file next to this config:
#   system_file = "persona.md"
'''

LOCAL_STARTER_CONFIG = '''\
# aic configuration for 100% local models: no API key, no internet once the models are downloaded.
# Save as ./aic.toml or ~/.config/aic/config.toml (or pass --config PATH).
#
# Setup (once):
#   1. Install Ollama from https://ollama.com (Windows, macOS, Linux) and leave it running.
#   2. Download the models used below:
#        ollama pull llama3.2
#        ollama pull qwen2.5:3b
#        ollama pull gemma3:4b
#
# Try it:
#   aic chat llama qwen --topic "Is free will an illusion?" --rounds 3
#   aic debate llama qwen --motion "Cats are better than dogs" --judge gemma
#   aic panel llama qwen gemma --question "Best first programming language?" --synthesizer gemma
#
# Model size vs. your PC (RAM needed is roughly the download size + 1-2 GB):
#   tiny, any PC:      qwen2.5:0.5b (0.4 GB), llama3.2:1b (1.3 GB)
#   small, 8 GB RAM:   llama3.2 (2 GB), qwen2.5:3b (1.9 GB), gemma3:4b (3.3 GB)
#   medium, 16 GB RAM: qwen2.5:7b (4.7 GB), llama3.1:8b (4.9 GB), mistral (4.1 GB)

[defaults]
rounds = 3
stream = true
# save_dir = "transcripts"    # auto-save a Markdown transcript of every run here

[agents.llama]
provider = "ollama"
model = "llama3.2"
system = "You are Llama, a curious philosopher. Answer in a few sentences and ask a sharp follow-up question."
temperature = 0.8
max_tokens = 400

[agents.qwen]
provider = "ollama"
model = "qwen2.5:3b"
system = "You are Qwen, a pragmatic engineer. Be concrete, challenge vague claims, keep replies short."
max_tokens = 400

[agents.gemma]
provider = "ollama"
model = "gemma3:4b"
system = "You are Gemma, a fair and witty referee who loves analogies."
max_tokens = 400

# Ollama on another machine of your network:
#   base_url = "http://192.168.1.20:11434"
'''

# -- Rendering --------------------------------------------------------------------------------

_PALETTE = ("36", "33", "35", "32", "34", "91", "96", "93", "95", "92")
_SEED_LABELS = {"chat": "Topic", "debate": "Motion", "panel": "Question"}
_ROUND_LABELS = {"chat": "round", "debate": "round", "relay": "step"}


def use_color(out: TextIO, no_color: bool = False) -> bool:
    """Colour only on a TTY, unless NO_COLOR is set or --no-color was given."""
    if no_color or os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(out.isatty())
    except (AttributeError, ValueError):
        return False


def agent_color(name: str) -> str:
    """A stable ANSI colour code per agent name (crc32, not hash(): that is randomised per run)."""
    return _PALETTE[zlib.crc32(name.encode("utf-8")) % len(_PALETTE)]


class Renderer:
    """Prints turns as they happen. Handles both live token streaming and whole-turn output."""

    def __init__(
        self,
        out: TextIO,
        *,
        color: bool,
        quiet: bool = False,
        mode: str = "",
        round_size: int | None = None,
        show_rounds: bool = False,
    ):
        self.out = out
        self.color = color
        self.quiet = quiet
        self.mode = mode
        self.round_size = round_size  # agent turns per round, used to label streamed headers
        self.show_rounds = show_rounds
        self.labels: dict[str, str] = {}
        self.special: dict[str, str] = {}  # speaker name -> "judge" | "synthesis"
        self._open: str | None = None  # speaker whose tokens are currently streaming
        self._fresh = False  # nothing but whitespace printed yet for the open turn
        self._pending = ""  # trailing whitespace held back until more text arrives
        self._agent_turns = 0
        enc = getattr(out, "encoding", None) or "utf-8"
        try:
            "──═".encode(enc)
            self._rule, self._dbl = "──", "══"
        except (UnicodeEncodeError, LookupError):
            self._rule, self._dbl = "--", "=="

    # -- helpers ---------------------------------------------------------------------------

    def paint(self, text: str, *codes: str) -> str:
        if not self.color or not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    def register(self, agents: list[Agent], judge: Agent | None = None, synth: Agent | None = None) -> None:
        for agent in agents:
            self.labels[agent.spec.name] = agent.label
        for agent, role in ((judge, "judge"), (synth, "synthesis")):
            if agent is not None:
                self.labels[agent.spec.name] = agent.label
                self.special[agent.spec.name] = role

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _header(self, speaker: str, label: str, role: str, round_no: int) -> str:
        if role == "judge":
            return self.paint(f"{self._dbl} JUDGE  {label} {self._dbl}", "1", "30", "43")
        if role == "synthesis":
            return self.paint(f"{self._dbl} SYNTHESIS  {label} {self._dbl}", "1", "30", "46")
        text = self.paint(f"{self._rule} {label} {self._rule}", "1", agent_color(speaker))
        if self.show_rounds and round_no > 0 and self.mode in _ROUND_LABELS:
            text += self.paint(f"  {_ROUND_LABELS[self.mode]} {round_no}", "2")
        return text

    def _role_of(self, speaker: str) -> str:
        return self.special.get(speaker, "agent")

    # -- events ----------------------------------------------------------------------------

    def seed(self, turn: Turn) -> None:
        if self.quiet:
            return
        label = _SEED_LABELS.get(self.mode, "Prompt")
        self._write(self.paint(f"{label}: {turn.content}", "2") + "\n\n")

    def token(self, speaker: str, chunk: str) -> None:
        if self._open != speaker:
            self.abort()
            role = self._role_of(speaker)
            round_no = 0
            if self.round_size:
                round_no = self._agent_turns // self.round_size + 1
            label = self.labels.get(speaker, speaker)
            self._write(self._header(speaker, label, role, round_no) + "\n")
            self._open = speaker
            self._fresh = True
        if self._fresh:
            chunk = chunk.lstrip()
            if not chunk:
                return
            self._fresh = False
        text = self._pending + chunk
        body = text.rstrip()
        self._pending = text[len(body):]
        if body:
            self._write(body)

    def turn(self, turn: Turn) -> None:
        if turn.role == "seed":
            self.seed(turn)
            return
        if turn.role == "agent":
            self._agent_turns += 1
        if self._open == turn.speaker:
            self._write("\n")
            self._open = None
            self._pending = ""
        else:
            self.abort()
            label = f"{turn.speaker} ({turn.model})" if turn.model else turn.speaker
            self._write(self._header(turn.speaker, label, turn.role, turn.round) + "\n")
            self._write(turn.content + "\n")
        if turn.elapsed >= 0.1:
            self._write(self.paint(f"({turn.elapsed:.1f}s)", "2") + "\n")
        self._write("\n")

    def note(self, text: str) -> None:
        if not self.quiet:
            self._write(self.paint(text, "2") + "\n\n")

    def abort(self) -> None:
        """Close a half-streamed turn (after an error or Ctrl-C) so the next line starts clean."""
        if self._open is not None:
            self._write("\n")
            self._open = None
            self._pending = ""


# -- Saving -----------------------------------------------------------------------------------


def _auto_save_path(save_dir: str, mode: str) -> Path:
    folder = Path(save_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{mode}-{stamp}.md"
    n = 2
    while path.exists():
        path = folder / f"{mode}-{stamp}-{n}.md"
        n += 1
    return path


class _Run:
    """One CLI invocation of a conversation mode: rendering, saving and error handling."""

    def __init__(self, args: argparse.Namespace, config: Config, out: TextIO, err: TextIO):
        self.args = args
        self.config = config
        self.out = out
        self.err = err
        save = getattr(args, "save", None)
        save_dir = config.defaults.get("save_dir")
        self.save_path: Path | None = None
        self._save_dir = str(save_dir) if save_dir and not save else None
        if save:
            self.save_path = Path(save).expanduser()

    def streaming(self, allowed: bool = True) -> bool:
        if not allowed or getattr(self.args, "no_stream", False):
            return False
        return bool(self.config.defaults.get("stream", True))

    def execute(
        self,
        mode: str,
        runner: Callable[[Callable[[Turn], None], Callable[[str, str], None] | None], Transcript],
        *,
        topic: str,
        agents: list[Agent],
        judge: Agent | None = None,
        synth: Agent | None = None,
        stream: bool = True,
        round_size: int | None = None,
        show_rounds: bool = False,
    ) -> int:
        renderer = Renderer(
            self.out,
            color=use_color(self.out, getattr(self.args, "no_color", False)),
            quiet=getattr(self.args, "quiet", False),
            mode=mode,
            round_size=round_size,
            show_rounds=show_rounds,
        )
        renderer.register(agents, judge, synth)
        seen: list[Turn] = []

        def on_turn(turn: Turn) -> None:
            seen.append(turn)
            renderer.turn(turn)

        on_token = renderer.token if stream else None
        try:
            transcript = runner(on_turn, on_token)
        except KeyboardInterrupt:
            renderer.abort()
            print("interrupted", file=self.err)
            self._save(Transcript(mode=mode, topic=topic, turns=list(seen)))
            return 130
        except AICError as exc:
            renderer.abort()
            partial = getattr(exc, "transcript", None)
            if partial is None:
                partial = Transcript(mode=mode, topic=topic, turns=list(seen))
            self._save(partial)
            print(f"error: {exc}", file=self.err)
            return 1
        return 0 if self._save(transcript) else 1

    def _save(self, transcript: Transcript) -> bool:
        path = self.save_path
        if path is None and self._save_dir:
            path = _auto_save_path(self._save_dir, transcript.mode)
        if path is None:
            return True
        try:
            written = transcript.save(path)
        except OSError as exc:
            print(f"error: cannot save transcript to {path}: {exc}", file=self.err)
            return False
        print(f"saved transcript: {written}", file=self.err)
        return True


# -- Command helpers --------------------------------------------------------------------------


def _build_agents(specs: list[AgentSpec]) -> list[Agent]:
    agents: list[Agent] = []
    for spec in specs:
        try:
            agents.append(Agent(spec))
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"Cannot load provider '{spec.provider}' for agent '{spec.name}': {exc}") from None
    return agents


def _resolve(config: Config, refs: list[str], extra: str | None = None) -> tuple[list[Agent], Agent | None]:
    """Resolve agent refs plus an optional extra role (judge/synthesizer) with unique names."""
    specs = resolve_agents(config, refs + ([extra] if extra else []))
    agents = _build_agents(specs)
    if extra:
        return agents[:-1], agents[-1]
    return agents, None


def _rounds(args: argparse.Namespace, config: Config, fallback: int) -> int:
    if args.rounds is not None:
        return int(args.rounds)
    value = config.defaults.get("rounds", fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"defaults.rounds must be an integer, got {value!r}")
    return value


def cmd_ask(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    prompt = " ".join(args.prompt)
    agents, _ = _resolve(config, [args.agent])
    agent = agents[0]
    run = _Run(args, config, out, err)

    def runner(on_turn: Callable[[Turn], None], on_token: Any) -> Transcript:
        transcript = Transcript(mode="ask", topic=prompt)
        on_turn(transcript.add(Turn(speaker="user", content=prompt, round=0, role="seed")))
        started = time.monotonic()
        try:
            text = agent.respond([Message("user", prompt)], on_token)
        except ProviderError as exc:
            exc.transcript = transcript  # type: ignore[attr-defined]
            raise
        model = f"{agent.spec.provider}:{agent.spec.model}"
        turn = Turn(agent.spec.name, text, round=1, model=model, elapsed=time.monotonic() - started)
        on_turn(transcript.add(turn))
        return transcript

    return run.execute("ask", runner, topic=prompt, agents=agents, stream=run.streaming())


def cmd_chat(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    agents, _ = _resolve(config, args.agents)
    rounds = _rounds(args, config, DEFAULT_CHAT_ROUNDS)
    run = _Run(args, config, out, err)

    def runner(on_turn: Callable[[Turn], None], on_token: Any) -> Transcript:
        return run_chat(
            agents, args.topic, rounds, on_turn=on_turn, on_token=on_token, stop_phrase=args.stop
        )

    return run.execute(
        "chat", runner, topic=args.topic, agents=agents, stream=run.streaming(),
        round_size=len(agents), show_rounds=rounds > 1,
    )


def cmd_relay(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    agents, _ = _resolve(config, args.agents)
    run = _Run(args, config, out, err)

    def runner(on_turn: Callable[[Turn], None], on_token: Any) -> Transcript:
        return run_relay(agents, args.prompt, on_turn=on_turn, on_token=on_token)

    return run.execute(
        "relay", runner, topic=args.prompt, agents=agents, stream=run.streaming(),
        round_size=1, show_rounds=len(agents) > 1,
    )


def cmd_debate(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    agents, judge = _resolve(config, args.agents, args.judge)
    rounds = _rounds(args, config, DEFAULT_DEBATE_ROUNDS)
    run = _Run(args, config, out, err)

    def runner(on_turn: Callable[[Turn], None], on_token: Any) -> Transcript:
        return run_debate(
            agents, args.motion, rounds, judge=judge, on_turn=on_turn, on_token=on_token
        )

    return run.execute(
        "debate", runner, topic=args.motion, agents=agents, judge=judge, stream=run.streaming(),
        round_size=len(agents), show_rounds=rounds > 1,
    )


def cmd_panel(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    agents, synth = _resolve(config, args.agents, args.synthesizer)
    parallel = not args.sequential
    run = _Run(args, config, out, err)

    def runner(on_turn: Callable[[Turn], None], on_token: Any) -> Transcript:
        return run_panel(
            agents, args.question, synthesizer=synth, parallel=parallel,
            on_turn=on_turn, on_token=on_token,
        )

    # Parallel panelists cannot stream, so parallel mode prints whole turns.
    stream = run.streaming(allowed=not (parallel and len(agents) > 1))
    if not stream and parallel and len(agents) > 1 and not args.quiet:
        names = ", ".join(a.spec.name for a in agents)
        print(f"(asking {names} in parallel...)", file=err)
    return run.execute(
        "panel", runner, topic=args.question, agents=agents, synth=synth, stream=stream
    )


def cmd_agents(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    if not config.agents:
        where = f" in {config.path}" if config.path else ""
        print(f"No agents configured{where}. Run 'aic init' to create a starter aic.toml.", file=out)
        return 0
    if config.path:
        print(f"Config: {config.path}", file=out)
    width = max(len(name) for name in config.agents)
    for name, spec in config.agents.items():
        persona = " ".join(spec.system.split())
        if len(persona) > 60:
            persona = persona[:57] + "..."
        print(f"{name.ljust(width)}  {spec.provider}:{spec.model}  {persona}".rstrip(), file=out)
    return 0


def cmd_providers(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    color = use_color(out, getattr(args, "no_color", False))
    rows: list[tuple[str, str, str, str]] = []
    for name in REGISTRY:
        try:
            env = provider_class(name).api_key_env
        except (AICError, ImportError, AttributeError) as exc:
            rows.append((name, "-", f"unavailable ({exc})", "31"))
            continue
        if env is None:
            rows.append((name, "-", "no key needed", "2"))
        elif os.environ.get(env):
            rows.append((name, env, "key set", "32"))
        else:
            rows.append((name, env, "missing", "33"))
    w_name = max(len("PROVIDER"), *(len(r[0]) for r in rows))
    w_env = max(len("ENV VAR"), *(len(r[1]) for r in rows))
    print(f"{'PROVIDER'.ljust(w_name)}  {'ENV VAR'.ljust(w_env)}  STATUS", file=out)
    for name, env, status, code in rows:
        shown = f"\x1b[{code}m{status}\x1b[0m" if color else status
        print(f"{name.ljust(w_name)}  {env.ljust(w_env)}  {shown}", file=out)
    return 0


def cmd_init(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    path = Path(args.path).expanduser()
    if path.exists() and not args.force:
        raise ConfigError(f"{path} already exists. Use --force to overwrite it.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(LOCAL_STARTER_CONFIG if args.local else STARTER_CONFIG, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot write {path}: {exc}") from None
    if args.local:
        print(f"Wrote {path}. Install Ollama (https://ollama.com), pull the models, then try:", file=out)
        print("  ollama pull llama3.2 && ollama pull qwen2.5:3b && ollama pull gemma3:4b", file=out)
        print('  aic chat llama qwen --topic "Is free will an illusion?"', file=out)
        return 0
    print(f"Wrote {path}. Set your API keys (see 'aic providers'), then try:", file=out)
    print('  aic chat claude gpt --topic "Is free will an illusion?"', file=out)
    return 0


def cmd_replay(args: argparse.Namespace, config: Config, out: TextIO, err: TextIO) -> int:
    path = Path(args.file).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        transcript = Transcript.from_dict(data)
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from None
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"{path} is not a valid aic JSON transcript ({exc!r})") from None
    agent_turns = [t for t in transcript.turns if t.role == "agent"]
    renderer = Renderer(
        out,
        color=use_color(out, getattr(args, "no_color", False)),
        quiet=getattr(args, "quiet", False),
        mode=transcript.mode,
        show_rounds=any(t.round > 1 for t in agent_turns) or transcript.mode == "relay" and len(agent_turns) > 1,
    )
    renderer.note(f"aic {transcript.mode} - {transcript.started_at}")
    for turn in transcript.turns:
        renderer.turn(turn)
    return 0


# -- Parser -----------------------------------------------------------------------------------


def _add_globals(parser: argparse.ArgumentParser, sub: bool) -> None:
    """Global flags. On subcommands they default to SUPPRESS so they work before or after the command."""

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if sub else value

    parser.add_argument("--config", metavar="PATH", default=default(None), help="config file (default: search aic.toml)")
    parser.add_argument("--no-stream", action="store_true", default=default(False), help="print whole turns instead of live tokens")
    parser.add_argument("--no-color", action="store_true", default=default(False), help="disable ANSI colours")
    parser.add_argument("--save", metavar="PATH", default=default(None), help="save the transcript (.json or .md)")
    parser.add_argument("--quiet", action="store_true", default=default(False), help="do not echo the seed prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aic",
        description="Let different AI models talk to each other.",
        epilog="Agents are config names or inline specs: claude, gpt=openai:gpt-5, ollama:llama3.2",
    )
    parser.add_argument("--version", action="version", version=f"aic {__version__}")
    _add_globals(parser, sub=False)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        _add_globals(p, sub=True)
        return p

    p = add("ask", "ask one agent a question (streamed)")
    p.add_argument("agent", metavar="AGENT")
    p.add_argument("prompt", nargs="+", metavar="PROMPT")
    p.set_defaults(func=cmd_ask)

    p = add("chat", "round-robin conversation between agents")
    p.add_argument("agents", nargs="+", metavar="AGENT")
    p.add_argument("--topic", required=True, help="opening topic")
    p.add_argument("--rounds", type=int, default=None, help=f"rounds (default: config or {DEFAULT_CHAT_ROUNDS})")
    p.add_argument("--stop", metavar="PHRASE", default=None, help="stop early when a reply contains PHRASE")
    p.set_defaults(func=cmd_chat)

    p = add("relay", "pipeline: each agent transforms the previous agent's output")
    p.add_argument("agents", nargs="+", metavar="AGENT")
    p.add_argument("--prompt", required=True, help="input for the first agent")
    p.set_defaults(func=cmd_relay)

    p = add("debate", "agents argue a motion, optionally with a judge")
    p.add_argument("agents", nargs="+", metavar="AGENT")
    p.add_argument("--motion", required=True, help="the motion to debate")
    p.add_argument("--rounds", type=int, default=None, help=f"rounds (default: config or {DEFAULT_DEBATE_ROUNDS})")
    p.add_argument("--judge", metavar="AGENT", default=None, help="agent that delivers a verdict")
    p.set_defaults(func=cmd_debate)

    p = add("panel", "everyone answers independently, optionally synthesized")
    p.add_argument("agents", nargs="+", metavar="AGENT")
    p.add_argument("--question", required=True, help="question for the panel")
    p.add_argument("--synthesizer", metavar="AGENT", default=None, help="agent that merges the answers")
    p.add_argument("--sequential", action="store_true", help="ask one agent at a time (enables streaming)")
    p.set_defaults(func=cmd_panel)

    p = add("agents", "list configured agents")
    p.set_defaults(func=cmd_agents)

    p = add("providers", "list providers and whether their API key is set")
    p.set_defaults(func=cmd_providers)

    p = add("init", "write a starter aic.toml")
    p.add_argument("path", nargs="?", default="aic.toml", metavar="PATH")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--local", action="store_true",
                   help="only local Ollama models: no API key needed")
    p.set_defaults(func=cmd_init)

    p = add("replay", "pretty-print a saved JSON transcript")
    p.add_argument("file", metavar="FILE.json")
    p.set_defaults(func=cmd_replay)
    return parser


# Commands that work without loading a config file.
_NO_CONFIG = {cmd_providers, cmd_init, cmd_replay}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --version, --help and usage errors
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)

    out, err = sys.stdout, sys.stderr
    if getattr(args, "func", None) is None:
        parser.print_help(out)
        return 0
    try:
        config = Config() if args.func in _NO_CONFIG else load_config(args.config)
        return args.func(args, config, out, err)
    except KeyboardInterrupt:
        print("interrupted", file=err)
        return 130
    except AICError as exc:
        print(f"error: {exc}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
