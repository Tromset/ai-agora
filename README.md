# ai-agora

Custom CLI to let your different Ai and models talk to eachother.

`aic` connects Claude, GPT, Gemini, Mistral, Groq, local Ollama models and more in one terminal session: free conversation, relay pipelines, debates with a judge, and panels with a synthesis. Python 3.11+, standard library only.

## Install

```sh
pipx install .        # global `aic` command
# or, for development
pip install -e .
```

## 30-second offline demo

No API keys needed: `examples/offline-demo.toml` defines `mock` agents that tell a tiny story.

```sh
aic --config examples/offline-demo.toml chat captain robot --topic "the lighthouse" --rounds 2
```

Each agent gets a coloured header, replies stream live, and `--save story.md` writes a transcript.

## 100% local, no API key

Every model runs on your own computer through [Ollama](https://ollama.com). You need no account and no key, and nothing leaves your machine once the models are downloaded.

1. Install Python 3.11+ ([python.org](https://www.python.org/downloads/); on Windows tick "Add Python to PATH") and Ollama ([ollama.com/download](https://ollama.com/download)), which then keeps running in the background.
2. Install `aic` and write a local config:
   ```sh
   pip install -e .
   aic init --local            # writes aic.toml with three Ollama agents: llama, qwen, gemma
   ```
3. Download the models once:
   ```sh
   ollama pull llama3.2
   ollama pull qwen2.5:3b
   ollama pull gemma3:4b
   ```
4. Let them talk:
   ```sh
   aic chat llama qwen --topic "Is free will an illusion?" --rounds 3
   aic debate llama qwen --motion "Cats are better than dogs" --judge gemma
   ```

Pick models that fit your RAM. Plan on roughly the download size plus 1-2 GB:

| Your PC | Models |
|---|---|
| any | `qwen2.5:0.5b` (0.4 GB), `llama3.2:1b` (1.3 GB) |
| 8 GB RAM | `llama3.2` (2 GB), `qwen2.5:3b` (1.9 GB), `gemma3:4b` (3.3 GB) |
| 16 GB RAM | `qwen2.5:7b` (4.7 GB), `llama3.1:8b` (4.9 GB), `mistral` (4.1 GB) |

You can also skip the config file and name models inline: `aic chat ollama:llama3.2 ollama:qwen2.5:3b --topic "..."`. Tiny models (under 3B parameters) run fast but ramble. Use 3B+ models for conversations worth reading. `examples/local.toml` is the same config as `aic init --local`.

## API keys

Keys come from environment variables only. Run `aic providers` to see which are set (the key itself is never printed).

| Provider | Environment variable | Default endpoint |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | https://api.anthropic.com/v1 |
| `openai` | `OPENAI_API_KEY` | https://api.openai.com/v1 |
| `openrouter` | `OPENROUTER_API_KEY` | https://openrouter.ai/api/v1 |
| `groq` | `GROQ_API_KEY` | https://api.groq.com/openai/v1 |
| `mistral` | `MISTRAL_API_KEY` | https://api.mistral.ai/v1 |
| `deepseek` | `DEEPSEEK_API_KEY` | https://api.deepseek.com/v1 |
| `xai` | `XAI_API_KEY` | https://api.x.ai/v1 |
| `gemini` | `GEMINI_API_KEY` | https://generativelanguage.googleapis.com/v1beta |
| `ollama` | none | http://localhost:11434 (or `$OLLAMA_HOST`) |
| `mock` | none | offline, for tests and demos |

## Configuration

`aic init` writes a commented starter `aic.toml`. Config is looked up in `./aic.toml`, then `$AIC_CONFIG`, then `~/.config/aic/config.toml`; `--config PATH` overrides the search.

```toml
[defaults]
rounds = 3                  # default for --rounds
stream = true               # false = print whole turns
save_dir = "transcripts"    # auto-save <mode>-<timestamp>.md here

[agents.claude]
provider = "anthropic"      # required, one of the providers above
model = "claude-sonnet-5-5" # required
system = "You are a curious philosopher."
temperature = 0.8
max_tokens = 1024
# system_file = "persona.md"  # alternative to `system`, relative to the config file

[agents.local]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"   # any other key goes to the provider as an option
```

Known agent keys are `provider`, `model`, `system`, `system_file`, `temperature`, `max_tokens`. Extra keys (`base_url`, `api_key_env`, `timeout`, `replies` for mock...) are passed to the provider.

Agents on the command line can be config names or inline specs: `claude`, `gpt=openai:gpt-5`, `ollama:llama3.2`. Repeating an agent renames later copies `name#2`, `name#3`.

## Modes

```sh
# ask: one agent, streamed
aic ask claude "Explain the halting problem in two sentences"

# chat: round-robin conversation; stop early when a reply contains a phrase
aic chat claude gpt --topic "Is free will an illusion?" --rounds 4 --stop "I agree"

# relay: each agent only sees the previous agent's output
aic relay gemini claude gpt --prompt "Write a haiku about compilers, then translate and critique it"

# debate: agents take opposing sides, an optional judge gives a verdict
aic debate claude gpt --motion "Tabs beat spaces" --rounds 2 --judge gemini

# panel: everyone answers independently (in parallel), an optional synthesizer merges the answers
aic panel claude gpt gemini --question "Best first programming language?" --synthesizer claude
aic panel claude gpt --question "..." --sequential   # one at a time, with live streaming

# housekeeping
aic agents                  # configured agents
aic providers               # providers and key status
aic init [PATH] [--force]   # write a starter aic.toml
aic replay run.json         # pretty-print a saved transcript
```

Global flags (accepted before or after the command): `--config PATH`, `--no-stream`, `--no-color`, `--save PATH`, `--quiet` (hide the dimmed seed prompt), `--version`. Colour is off automatically when stdout is not a terminal or `NO_COLOR` is set.

## Saving transcripts

```sh
aic --save debate.md   debate claude gpt --motion "..."   # Markdown
aic --save debate.json debate claude gpt --motion "..."   # JSON, can be replayed
aic replay debate.json
```

With `defaults.save_dir` set and no `--save`, every run is saved as `<mode>-<timestamp>.md` in that directory. If a provider fails mid-run, the partial transcript is still saved and `aic` exits with status 1.

## Adding a provider

1. Create `src/aiconnect/providers/myprovider.py` with a subclass of `aiconnect.providers.base.Provider`. Set `name`, `api_key_env` and `default_base_url`, implement `complete()` and, ideally, a real token-by-token `stream()`. Use `aiconnect.http` helpers and raise `ProviderError` with a readable message.
2. Register it in `REGISTRY` in `src/aiconnect/providers/__init__.py`: `"myprovider": "aiconnect.providers.myprovider:MyProvider"`.
3. Add a test that monkeypatches `aiconnect.http.post_json` / `post_lines` so it never touches the network.

## Tests

```sh
python -m unittest discover -s tests -t . -v
```

All tests run offline. CI runs them on Python 3.11, 3.12 and 3.13, plus a smoke run of the offline demo.
