"""Conversation record: ordered turns plus Markdown / JSON export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Turn:
    """One utterance in a run."""

    speaker: str  # agent name, or "user" for the seed prompt
    content: str
    round: int = 0
    role: str = "agent"  # "seed" | "agent" | "judge" | "synthesis"
    model: str = ""  # "provider:model", empty for seed
    elapsed: float = 0.0  # seconds


@dataclass
class Transcript:
    """Everything said in one run, in order."""

    mode: str
    topic: str
    turns: list[Turn] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)

    def add(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        return turn

    # -- export ---------------------------------------------------------------------------

    def participants(self) -> list[tuple[str, str]]:
        """Distinct (speaker, model) pairs of non-seed turns, in order of first appearance."""
        seen: list[tuple[str, str]] = []
        for turn in self.turns:
            if turn.role == "seed":
                continue
            pair = (turn.speaker, turn.model)
            if pair not in seen:
                seen.append(pair)
        return seen

    @staticmethod
    def _heading(turn: Turn) -> str:
        if turn.role == "seed":
            return "### Prompt"
        who = turn.speaker + (f" · {turn.model}" if turn.model else "")
        if turn.role == "judge":
            return f"### Judge: {who}"
        if turn.role == "synthesis":
            return f"### Synthesis: {who}"
        return f"### {who} — round {turn.round}"

    def to_markdown(self) -> str:
        parts = [f"# aic {self.mode}: {self.topic}", ""]
        people = ", ".join(f"{name} ({model})" if model else name for name, model in self.participants())
        meta = f"*Started: {self.started_at}*"
        if people:
            meta += f" · *Participants: {people}*"
        parts += [meta, ""]
        for turn in self.turns:
            parts += [self._heading(turn), "", turn.content, ""]
        return "\n".join(parts).rstrip("\n") + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "topic": self.topic,
            "started_at": self.started_at,
            "turns": [asdict(t) for t in self.turns],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        turns = [
            Turn(
                speaker=str(t["speaker"]),
                content=str(t["content"]),
                round=int(t.get("round", 0)),
                role=str(t.get("role", "agent")),
                model=str(t.get("model", "")),
                elapsed=float(t.get("elapsed", 0.0)),
            )
            for t in data.get("turns", [])
        ]
        kwargs: dict[str, Any] = {}
        if data.get("started_at"):
            kwargs["started_at"] = str(data["started_at"])
        return cls(mode=str(data["mode"]), topic=str(data["topic"]), turns=turns, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Write to `path` (parent dirs created): `.json` -> JSON, anything else -> Markdown."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_json() if target.suffix.lower() == ".json" else self.to_markdown()
        target.write_text(text, encoding="utf-8")
        return target
