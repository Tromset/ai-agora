# PLAN — ai-connected-to-ai (`aic`)

## Goal
A zero-dependency Python CLI, `aic`, that lets different AI models (Claude, GPT, Gemini,
Mistral, Groq, local Ollama models...) talk to each other: free conversation, relay
pipelines, debates with a judge, and panels with a synthesis, all from the terminal.

## Refined brief (the "better prompt")
> Build `aic`, a Python 3.11+ command-line tool using **only the standard library**
> (urllib, argparse, tomllib, concurrent.futures, unittest). A user declares *agents*
> (name + provider + model + persona/system prompt) in a TOML file or inline on the
> command line, then picks a *mode* that decides who speaks when and what each agent sees.
> Every run streams to the terminal with coloured speaker labels and can be saved as a
> Markdown or JSON transcript. API keys come from environment variables only. A `mock`
> provider makes everything testable offline. Each module ships with unittest tests
> that never touch the network.

## Stack & why
- **Python 3.11 stdlib only**: `tomllib` for config, `urllib` for HTTP (respects `HTTPS_PROXY`),
  no install friction, `pipx install .` just works.
- **unittest** for tests: `python -m unittest discover -s tests -t .`.
- src layout, entry point `aic = aiconnect.cli:main`.

## Folder tree
```
pyproject.toml
src/aiconnect/
  __init__.py            version                         [core, done]
  errors.py              AICError / ConfigError / ProviderError [core, done]
  types.py               Message, AgentSpec              [core, done]
  http.py                post_json, post_lines, iter_sse_data [core, done]
  providers/
    __init__.py          registry + get_provider()       [core, done]
    base.py              Provider ABC                    [core, done]
    mock.py              offline deterministic provider  [core, done]
    anthropic.py         Messages API                    [agent A]
    openai.py            Chat Completions + compatible presets [agent A]
    gemini.py            generateContent                 [agent B]
    ollama.py            local /api/chat                 [agent B]
  transcript.py          Turn, Transcript, exports       [agent C]
  orchestrator.py        Agent, perspective, modes       [agent C]
  config.py              TOML + inline agent parsing     [agent D]
  cli.py                 argparse commands, rendering    [agent D]
tests/                   test_<module>.py per module
examples/                sample aic.toml files          [agent D]
.github/workflows/ci.yml                                 [agent D]
README.md                                                [agent D]
```

## Shared contracts (do not change without the integrator)

### Core (already written, read-only for agents)
- `Message(role: "system"|"user"|"assistant", content: str)`
- `AgentSpec(name, provider, model, system="", temperature=None, max_tokens=1024, options={})`
- `Provider(options)`: `.complete(model, messages, *, system, temperature, max_tokens) -> str`,
  `.stream(...) -> Iterator[str]`, `.api_key()`, `.base_url`, `.timeout`.
  `messages` never contain `system` role; they start with `user` and alternate.
- `get_provider(name, options) -> Provider` (registry in `providers/__init__.py`).
- `http.post_json`, `http.post_lines`, `http.iter_sse_data`; all raise `ProviderError`.

### Providers (agents A, B)
Each provider subclass sets `name`, `api_key_env`, `default_base_url`, implements
`complete` and real token `stream`ing, and maps HTTP/JSON problems to `ProviderError`
with a readable message. Tests monkeypatch `aiconnect.http.post_json` / `post_lines`
(import the module as `from .. import http` and call `http.post_json(...)` so patching works).

| name | module:class | key env | base url |
|---|---|---|---|
| anthropic | anthropic:AnthropicProvider | ANTHROPIC_API_KEY | https://api.anthropic.com/v1 |
| openai | openai:OpenAIProvider | OPENAI_API_KEY | https://api.openai.com/v1 |
| openrouter | openai:OpenRouterProvider | OPENROUTER_API_KEY | https://openrouter.ai/api/v1 |
| groq | openai:GroqProvider | GROQ_API_KEY | https://api.groq.com/openai/v1 |
| mistral | openai:MistralProvider | MISTRAL_API_KEY | https://api.mistral.ai/v1 |
| deepseek | openai:DeepSeekProvider | DEEPSEEK_API_KEY | https://api.deepseek.com/v1 |
| xai | openai:XAIProvider | XAI_API_KEY | https://api.x.ai/v1 |
| gemini | gemini:GeminiProvider | GEMINI_API_KEY | https://generativelanguage.googleapis.com/v1beta |
| ollama | ollama:OllamaProvider | (none) | http://localhost:11434 (or $OLLAMA_HOST) |

### Transcript (agent C) — `aiconnect/transcript.py`
```python
@dataclass
class Turn:
    speaker: str          # agent name, or "user" for the seed prompt
    content: str
    round: int = 0
    role: str = "agent"   # "seed" | "agent" | "judge" | "synthesis"
    model: str = ""       # "provider:model", empty for seed
    elapsed: float = 0.0  # seconds

@dataclass
class Transcript:
    mode: str
    topic: str
    turns: list[Turn] = field(default_factory=list)
    started_at: str = <ISO-8601 UTC now>
    def add(self, turn: Turn) -> Turn
    def to_markdown(self) -> str
    def to_dict(self) -> dict
    def to_json(self) -> str
    @classmethod
    def from_dict(cls, data: dict) -> "Transcript"
    def save(self, path: str | Path) -> Path   # .json -> JSON, anything else -> Markdown
```

### Orchestrator (agent C) — `aiconnect/orchestrator.py`
```python
OnTurn  = Callable[[Turn], None]          # called after each turn is complete
OnToken = Callable[[str, str], None]      # (speaker, chunk) while streaming

class Agent:
    def __init__(self, spec: AgentSpec, provider: Provider | None = None)  # provider defaults to get_provider(spec.provider, spec.options)
    spec: AgentSpec; provider: Provider
    @property label -> str                # "name (provider:model)"
    def respond(self, messages: list[Message], on_token: OnToken | None = None) -> str

def build_view(transcript: Transcript, agent_name: str, instruction: str = "") -> list[Message]
    # Perspective: this agent's own turns -> "assistant"; every other turn (seed included)
    # -> "user" prefixed with "[Speaker]: ". Consecutive same-role messages are merged
    # (joined with "\n\n"); the list must start with "user" (inject the topic if needed).
    # `instruction` (if any) is appended to the final user message.

def run_chat(agents, topic, rounds=3, *, on_turn=None, on_token=None, stop_phrase=None) -> Transcript
    # Round-robin; each round every agent speaks once. Stops early if a reply contains stop_phrase.
def run_relay(agents, prompt, *, on_turn=None, on_token=None) -> Transcript
    # Pipeline: agent[i] receives only agent[i-1]'s output (agent[0] gets the prompt).
def run_debate(agents, motion, rounds=2, *, judge=None, on_turn=None, on_token=None) -> Transcript
    # Agents argue in turns (positions assigned: agent i gets "for"/"against"/... hint in instruction);
    # optional judge (Agent) reads the whole debate and delivers a verdict (role="judge").
def run_panel(agents, question, *, synthesizer=None, parallel=True, on_turn=None, on_token=None) -> Transcript
    # Everyone answers the same question independently (thread pool when parallel; no token
    # streaming in parallel mode), then optional synthesizer merges answers (role="synthesis").
```
All modes put a `Turn(speaker="user", role="seed", ...)` first. Provider errors propagate
as `ProviderError` (the CLI prints them); the partial transcript is attached as `err.transcript`.

### Config (agent D) — `aiconnect/config.py`
```python
@dataclass
class Config:
    agents: dict[str, AgentSpec]
    defaults: dict[str, Any]          # e.g. rounds, stream, save_dir
    path: Path | None

def default_config_paths() -> list[Path]    # ./aic.toml, $AIC_CONFIG, ~/.config/aic/config.toml
def load_config(path: str | Path | None = None) -> Config   # missing default file -> empty Config
def parse_agent_arg(value: str) -> AgentSpec
    # "name=provider:model" | "provider:model" (name = model) ; optional ",system=..." not needed
def resolve_agents(config: Config, refs: list[str]) -> list[AgentSpec]
    # each ref is either a name from config or an inline spec; duplicate names get suffixed #2
```
TOML shape:
```toml
[defaults]
rounds = 3
save_dir = "transcripts"

[agents.claude]
provider = "anthropic"
model = "claude-sonnet-5-5"
system = "You are a curious philosopher."
temperature = 0.8

[agents.local]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"   # any extra key goes to AgentSpec.options
```

### CLI (agent D) — `aiconnect/cli.py`, `main(argv=None) -> int`
```
aic ask     AGENT "prompt"                         single agent, streamed
aic chat    AGENT AGENT [...] --topic T [--rounds N] [--stop PHRASE]
aic relay   AGENT AGENT [...] --prompt P
aic debate  AGENT AGENT [...] --motion M [--rounds N] [--judge AGENT]
aic panel   AGENT AGENT [...] --question Q [--synthesizer AGENT] [--sequential]
aic agents                                         list configured agents
aic providers                                      list providers + whether key is set
aic init    [PATH]                                 write a starter aic.toml
aic replay  FILE.json                              pretty-print a saved transcript
Global: --config PATH, --no-stream, --no-color, --save PATH, --quiet, --version
```
Coloured speaker headers (ANSI, disabled when not a TTY / `NO_COLOR` / `--no-color`).
Exit codes: 0 ok, 1 AICError (message on stderr, no traceback), 130 on Ctrl-C.

## Task list
- [x] Core skeleton: pyproject, errors, types, http, provider base, registry, mock
- [x] A: anthropic.py + openai.py (+ presets) + tests/test_providers_anthropic_openai.py
- [x] B: gemini.py + ollama.py + tests/test_providers_gemini_ollama.py (+ tests for mock/registry)
- [x] C: transcript.py + orchestrator.py + tests/test_transcript.py + tests/test_orchestrator.py
- [x] D: config.py + cli.py + examples/ + README.md + CI + tests/test_config.py + tests/test_cli.py
- [x] Integration: full test suite green, `aic chat` end-to-end with mock agents, commit, PR

## Risks
- Role alternation: Anthropic/Gemini reject consecutive same-role messages → `build_view` merges.
- Gemini uses role "model" and `systemInstruction`; Anthropic uses top-level `system`.
- Newer OpenAI models reject `max_tokens` → send `max_completion_tokens` for `openai`, `max_tokens` for the compatible presets.
- Streaming formats differ: SSE (Anthropic, OpenAI, Gemini `alt=sse`) vs NDJSON (Ollama).
- Never print or log API keys.
