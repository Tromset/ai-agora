"""Conversation modes: who speaks when, and what each agent gets to see."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .errors import ConfigError, ProviderError
from .providers import get_provider
from .providers.base import Provider
from .transcript import Transcript, Turn
from .types import AgentSpec, Message

OnTurn = Callable[[Turn], None]  # called after each turn is complete
OnToken = Callable[[str, str], None]  # (speaker, chunk) while streaming

# -- Prompts (tweak freely; {placeholders} are filled with str.format) -----------------------

# The seed turn is shown to models as coming from this speaker.
SEED_SPEAKER_LABEL = "Moderator"
# Small (often local) models tend to parrot the "[Speaker]: " tags they see in the history.
_SPEAKER_TAG = re.compile(r"^(?:\[[^\]\n]{1,40}\]:\s*)+")

CHAT_INSTRUCTION = (
    "You are {name} in a group conversation with {others}. Reply to the latest messages in "
    "1-3 short paragraphs; do not speak for others and do not prefix your reply with your name."
)

RELAY_FIRST_INSTRUCTION = (
    "You are step {step} of {total} in a pipeline of AI models; your reply is handed to the "
    "next step. Reply with your result only."
)
RELAY_INSTRUCTION = (
    "Above is the output of the previous step (step {prev} of {total}). Improve, extend or "
    "transform this for the next step. Reply with your result only."
)

DEBATE_STANCES = ("for", "against")  # agent i argues DEBATE_STANCES[i % 2]
DEBATE_OPENING_INSTRUCTION = (
    'You are {name} in a debate on the motion: "{motion}". You argue {stance} the motion. '
    "Your opponents: {others}. Open with your strongest arguments in 1-3 short paragraphs."
)
DEBATE_REBUTTAL_INSTRUCTION = (
    'You are {name} in a debate on the motion: "{motion}". You argue {stance} the motion. '
    "Your opponents: {others}. Rebut your opponents' latest points, then reinforce your own "
    "case, in 1-3 short paragraphs. Stay in your stance."
)
JUDGE_INSTRUCTION = (
    'You are {name}, the impartial judge of the debate above on the motion: "{motion}". Weigh '
    "the arguments of every side, declare a winner (by name) and give your reasons."
)

PANEL_INSTRUCTION = (
    "You are {name} on an expert panel. Answer the question independently and concisely; you "
    "cannot see the other panelists' answers."
)
SYNTHESIS_INSTRUCTION = (
    "You are {name}, the synthesizer of the panel above. Merge the panelists' answers into one "
    "best answer, and note where they disagree."
)


class Agent:
    """A participant: an AgentSpec bound to a Provider instance."""

    def __init__(self, spec: AgentSpec, provider: Provider | None = None):
        self.spec = spec
        self.provider = provider if provider is not None else get_provider(spec.provider, spec.options)

    @property
    def label(self) -> str:
        return f"{self.spec.name} ({self.spec.provider}:{self.spec.model})"

    @property
    def model_id(self) -> str:
        return f"{self.spec.provider}:{self.spec.model}"

    def respond(self, messages: list[Message], on_token: OnToken | None = None) -> str:
        """Ask the provider; streams (calling on_token(name, chunk)) when on_token is given."""
        spec = self.spec
        kwargs = dict(system=spec.system, temperature=spec.temperature, max_tokens=spec.max_tokens)
        if on_token is None:
            return _clean_reply(self.provider.complete(spec.model, messages, **kwargs))
        chunks: list[str] = []
        pending: str | None = ""  # held back while it could still be a parroted speaker tag
        for chunk in self.provider.stream(spec.model, messages, **kwargs):
            chunks.append(chunk)
            if pending is None:
                on_token(spec.name, chunk)
                continue
            pending += chunk
            head = pending.lstrip()
            tag = _SPEAKER_TAG.match(head)
            if tag and tag.end() < len(head):
                on_token(spec.name, head[tag.end():])
                pending = None
            elif head and not tag and (not head.startswith("[") or "]" in head or len(head) > 45):
                on_token(spec.name, pending)
                pending = None
        if pending:
            on_token(spec.name, _clean_reply(pending))
        return _clean_reply("".join(chunks))


def _clean_reply(text: str) -> str:
    """Trim whitespace and any leading "[Speaker]: " tags the model copied from its view."""
    return _SPEAKER_TAG.sub("", text.strip()).strip()


# -- Perspective ------------------------------------------------------------------------------


def _view(transcript: Transcript, own: str | None, instruction: str) -> list[Message]:
    messages: list[Message] = []

    def push(role: str, content: str) -> None:
        if messages and messages[-1].role == role:
            messages[-1].content += "\n\n" + content
        else:
            messages.append(Message(role, content))  # type: ignore[arg-type]

    for turn in transcript.turns:
        if own is not None and turn.role != "seed" and turn.speaker == own:
            push("assistant", turn.content)
        else:
            label = SEED_SPEAKER_LABEL if turn.role == "seed" else turn.speaker
            push("user", f"[{label}]: {turn.content}")

    if not messages or messages[0].role != "user":
        messages.insert(0, Message("user", f"[{SEED_SPEAKER_LABEL}]: {transcript.topic}"))
    if instruction:
        if messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction
        else:
            messages.append(Message("user", instruction))
    return messages


def build_view(transcript: Transcript, agent_name: str, instruction: str = "") -> list[Message]:
    """The conversation as `agent_name` should see it.

    The agent's own turns become "assistant"; every other turn becomes "user" prefixed with
    "[Speaker]: " (the seed prompt is shown as "[Moderator]: ..."). Consecutive same-role
    messages are merged with a blank line, the list always starts with "user" (the topic is
    injected if needed), `instruction` is appended to the final user message (or added as a
    new user message if the last one is "assistant"), and no "system" message is ever emitted.
    """
    return _view(transcript, agent_name, instruction)


# -- Helpers ----------------------------------------------------------------------------------


def _check(agents: list[Agent], minimum: int, mode: str, rounds: int | None = None) -> None:
    if len(agents) < minimum:
        raise ConfigError(f"'{mode}' needs at least {minimum} agent(s), got {len(agents)}.")
    if rounds is not None and rounds < 1:
        raise ConfigError(f"'{mode}' needs rounds >= 1, got {rounds}.")


def _check_unique(agents: list[Agent], mode: str) -> None:
    names = [a.spec.name for a in agents]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"'{mode}' needs unique agent names, duplicated: {', '.join(dupes)}.")


def _start(mode: str, topic: str, on_turn: OnTurn | None) -> Transcript:
    transcript = Transcript(mode=mode, topic=topic)
    seed = transcript.add(Turn(speaker="user", content=topic, round=0, role="seed"))
    if on_turn:
        on_turn(seed)
    return transcript


def _ask(agent: Agent, messages: list[Message], on_token: OnToken | None) -> tuple[str, float]:
    started = time.monotonic()
    text = agent.respond(messages, on_token)
    return text, time.monotonic() - started


def _speak(
    transcript: Transcript,
    agent: Agent,
    messages: list[Message],
    *,
    round: int,
    role: str = "agent",
    on_turn: OnTurn | None,
    on_token: OnToken | None,
) -> Turn:
    text, elapsed = _ask(agent, messages, on_token)
    turn = transcript.add(
        Turn(agent.spec.name, text, round=round, role=role, model=agent.model_id, elapsed=elapsed)
    )
    if on_turn:
        on_turn(turn)
    return turn


def _others(agents: list[Agent], me: Agent) -> str:
    return ", ".join(a.spec.name for a in agents if a is not me)


# -- Modes ------------------------------------------------------------------------------------


def run_chat(
    agents: list[Agent],
    topic: str,
    rounds: int = 3,
    *,
    on_turn: OnTurn | None = None,
    on_token: OnToken | None = None,
    stop_phrase: str | None = None,
) -> Transcript:
    """Round-robin conversation; stops early when a reply contains `stop_phrase` (case-insensitive)."""
    _check(agents, 2, "chat", rounds)
    _check_unique(agents, "chat")
    transcript = _start("chat", topic, on_turn)
    try:
        for rnd in range(1, rounds + 1):
            for agent in agents:
                instruction = CHAT_INSTRUCTION.format(name=agent.spec.name, others=_others(agents, agent))
                turn = _speak(
                    transcript, agent, build_view(transcript, agent.spec.name, instruction),
                    round=rnd, on_turn=on_turn, on_token=on_token,
                )
                if stop_phrase and stop_phrase.lower() in turn.content.lower():
                    return transcript
    except ProviderError as err:
        err.transcript = transcript  # type: ignore[attr-defined]
        raise
    return transcript


def run_relay(
    agents: list[Agent],
    prompt: str,
    *,
    on_turn: OnTurn | None = None,
    on_token: OnToken | None = None,
) -> Transcript:
    """Pipeline: agent i sees only agent i-1's output (agent 0 sees the prompt). Round = step number."""
    _check(agents, 1, "relay")
    transcript = _start("relay", prompt, on_turn)
    total = len(agents)
    try:
        previous = prompt
        for step, agent in enumerate(agents, start=1):
            if step == 1:
                instruction = RELAY_FIRST_INSTRUCTION.format(step=step, total=total)
            else:
                instruction = RELAY_INSTRUCTION.format(prev=step - 1, total=total)
            messages = [Message("user", f"{previous}\n\n{instruction}")]
            previous = _speak(
                transcript, agent, messages, round=step, on_turn=on_turn, on_token=on_token
            ).content
    except ProviderError as err:
        err.transcript = transcript  # type: ignore[attr-defined]
        raise
    return transcript


def run_debate(
    agents: list[Agent],
    motion: str,
    rounds: int = 2,
    *,
    judge: Agent | None = None,
    on_turn: OnTurn | None = None,
    on_token: OnToken | None = None,
) -> Transcript:
    """Agents argue in turns (agent i takes DEBATE_STANCES[i % 2]); an optional judge gives a verdict."""
    _check(agents, 2, "debate", rounds)
    _check_unique(agents, "debate")
    transcript = _start("debate", motion, on_turn)
    try:
        for rnd in range(1, rounds + 1):
            for i, agent in enumerate(agents):
                template = DEBATE_OPENING_INSTRUCTION if rnd == 1 else DEBATE_REBUTTAL_INSTRUCTION
                instruction = template.format(
                    name=agent.spec.name,
                    motion=motion,
                    stance=DEBATE_STANCES[i % len(DEBATE_STANCES)],
                    others=_others(agents, agent),
                )
                _speak(
                    transcript, agent, build_view(transcript, agent.spec.name, instruction),
                    round=rnd, on_turn=on_turn, on_token=on_token,
                )
        if judge is not None:
            instruction = JUDGE_INSTRUCTION.format(name=judge.spec.name, motion=motion)
            # own=None: the judge sees every turn as coming from someone else.
            _speak(
                transcript, judge, _view(transcript, None, instruction),
                round=rounds + 1, role="judge", on_turn=on_turn, on_token=on_token,
            )
    except ProviderError as err:
        err.transcript = transcript  # type: ignore[attr-defined]
        raise
    return transcript


def run_panel(
    agents: list[Agent],
    question: str,
    *,
    synthesizer: Agent | None = None,
    parallel: bool = True,
    on_turn: OnTurn | None = None,
    on_token: OnToken | None = None,
) -> Transcript:
    """Everyone answers independently (round 1); an optional synthesizer merges (round 2).

    In parallel mode a thread pool is used, on_token is ignored, and turns are appended (and
    on_turn is called) in agent order from the calling thread.
    """
    _check(agents, 1, "panel")
    transcript = _start("panel", question, on_turn)
    seed_only = Transcript(mode="panel", topic=question, turns=list(transcript.turns))
    views = [
        _view(seed_only, a.spec.name, PANEL_INSTRUCTION.format(name=a.spec.name)) for a in agents
    ]
    try:
        if parallel and len(agents) > 1:
            with ThreadPoolExecutor(max_workers=len(agents)) as pool:
                futures = [pool.submit(_ask, a, v, None) for a, v in zip(agents, views)]
                try:
                    for agent, future in zip(agents, futures):
                        text, elapsed = future.result()
                        turn = transcript.add(
                            Turn(agent.spec.name, text, round=1, model=agent.model_id, elapsed=elapsed)
                        )
                        if on_turn:
                            on_turn(turn)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        else:
            for agent, view in zip(agents, views):
                _speak(
                    transcript, agent, view, round=1,
                    on_turn=on_turn, on_token=None if parallel else on_token,
                )
        if synthesizer is not None:
            instruction = SYNTHESIS_INSTRUCTION.format(name=synthesizer.spec.name)
            _speak(
                transcript, synthesizer, _view(transcript, None, instruction),
                round=2, role="synthesis", on_turn=on_turn, on_token=on_token,
            )
    except ProviderError as err:
        err.transcript = transcript  # type: ignore[attr-defined]
        raise
    return transcript
