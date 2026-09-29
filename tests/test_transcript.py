from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aiconnect.transcript import Transcript, Turn


def sample() -> Transcript:
    t = Transcript(mode="debate", topic="Tabs vs spaces", started_at="2026-01-02T03:04:05+00:00")
    t.add(Turn("user", "Tabs vs spaces", 0, "seed"))
    t.add(Turn("alice", "Tabs.\n\n```py\nx = 1\n```\n  indented", 1, "agent", "mock:a", 0.25))
    t.add(Turn("bob", "Spaces – café ☕", 1, "agent", "mock:b", 1.5))
    t.add(Turn("carol", "Bob wins.", 2, "judge", "mock:c", 0.1))
    t.add(Turn("dave", "Merged.", 2, "synthesis", "mock:d", 0.2))
    return t


class TurnTests(unittest.TestCase):
    def test_defaults(self) -> None:
        turn = Turn("a", "hi")
        self.assertEqual((turn.round, turn.role, turn.model, turn.elapsed), (0, "agent", "", 0.0))


class TranscriptTests(unittest.TestCase):
    def test_add_returns_turn_and_appends(self) -> None:
        t = Transcript("chat", "x")
        turn = Turn("a", "hi")
        self.assertIs(t.add(turn), turn)
        self.assertEqual(t.turns, [turn])

    def test_started_at_default_is_iso_utc(self) -> None:
        t = Transcript("chat", "x")
        self.assertTrue(t.started_at.endswith("+00:00"))
        self.assertNotEqual(Transcript("chat", "x").started_at, "")

    def test_markdown_structure(self) -> None:
        md = sample().to_markdown()
        self.assertTrue(md.startswith("# aic debate: Tabs vs spaces\n"))
        self.assertIn("2026-01-02T03:04:05+00:00", md)
        self.assertIn("alice (mock:a)", md)
        self.assertIn("bob (mock:b)", md)
        self.assertIn("### Prompt", md)
        self.assertIn("### alice · mock:a — round 1", md)
        self.assertIn("### bob · mock:b — round 1", md)
        self.assertIn("### Judge: carol · mock:c", md)
        self.assertIn("### Synthesis: dave · mock:d", md)
        self.assertNotIn("user (", md)
        self.assertTrue(md.endswith("\n"))
        # Order of sections follows the turns.
        positions = [md.index(h) for h in ("### Prompt", "### alice", "### bob", "### Judge", "### Synthesis")]
        self.assertEqual(positions, sorted(positions))

    def test_markdown_content_verbatim(self) -> None:
        md = sample().to_markdown()
        self.assertIn("Tabs.\n\n```py\nx = 1\n```\n  indented", md)
        self.assertIn("Spaces – café ☕", md)

    def test_markdown_without_model(self) -> None:
        t = Transcript("chat", "x")
        t.add(Turn("a", "hello", 1))
        self.assertIn("### a — round 1", t.to_markdown())

    def test_empty_transcript_markdown(self) -> None:
        md = Transcript("chat", "x", started_at="S").to_markdown()
        self.assertIn("# aic chat: x", md)

    def test_dict_roundtrip_exact(self) -> None:
        t = sample()
        again = Transcript.from_dict(t.to_dict())
        self.assertEqual(again, t)
        self.assertEqual(again.to_dict(), t.to_dict())

    def test_json_roundtrip_exact(self) -> None:
        t = sample()
        again = Transcript.from_dict(json.loads(t.to_json()))
        self.assertEqual(again, t)
        self.assertIn("café ☕", t.to_json())  # not ASCII-escaped

    def test_to_dict_shape(self) -> None:
        d = sample().to_dict()
        self.assertEqual(set(d), {"mode", "topic", "started_at", "turns"})
        self.assertEqual(
            set(d["turns"][1]), {"speaker", "content", "round", "role", "model", "elapsed"}
        )

    def test_from_dict_defaults(self) -> None:
        t = Transcript.from_dict({"mode": "chat", "topic": "x", "turns": [{"speaker": "a", "content": "c"}]})
        self.assertEqual(t.turns, [Turn("a", "c")])
        self.assertTrue(t.started_at)

    def test_save_markdown_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Transcript.save(sample(), Path(tmp) / "a" / "b" / "run.md")
            self.assertIsInstance(path, Path)
            self.assertEqual(path.read_text(encoding="utf-8"), sample().to_markdown())

    def test_save_json_and_other_suffix(self) -> None:
        t = sample()
        with tempfile.TemporaryDirectory() as tmp:
            jpath = t.save(str(Path(tmp) / "x" / "run.json"))
            self.assertEqual(Transcript.from_dict(json.loads(jpath.read_text(encoding="utf-8"))), t)
            other = t.save(Path(tmp) / "run.txt")
            self.assertTrue(other.read_text(encoding="utf-8").startswith("# aic debate"))
            upper = t.save(Path(tmp) / "RUN.JSON")
            json.loads(upper.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
