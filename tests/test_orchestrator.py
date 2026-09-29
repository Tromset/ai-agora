from __future__ import annotations

import threading
import time
import unittest

from aiconnect.errors import ConfigError, ProviderError
from aiconnect.orchestrator import (
    Agent,
    build_view,
    run_chat,
    run_debate,
    run_panel,
    run_relay,
)
from aiconnect.providers.mock import MockProvider
from aiconnect.transcript import Transcript, Turn
from aiconnect.types import AgentSpec, Message


def make(name: str, replies: list[str] | None = None, **spec_kw) -> Agent:
    options = {"replies": replies} if replies else {}
    spec = AgentSpec(name=name, provider="mock", model=f"m-{name}", **spec_kw)
    return Agent(spec, provider=MockProvider(options))


def calls(agent: Agent) -> list[list[Message]]:
    return agent.provider.calls  # type: ignore[attr-defined]


class FailingProvider(MockProvider):
    def __init__(self, fail_on: int, options=None):
        super().__init__(options)
        self.fail_on = fail_on

    def complete(self, model, messages, **kw):
        if len(self.calls) + 1 == self.fail_on:
            self.calls.append(list(messages))
            raise ProviderError("boom")
        return super().complete(model, messages, **kw)


def failing(name: str, fail_on: int) -> Agent:
    return Agent(AgentSpec(name, "mock", f"m-{name}"), provider=FailingProvider(fail_on))


class InvariantMixin:
    def assertValid(self, messages: list[Message]) -> None:
        self.assertTrue(messages)  # type: ignore[attr-defined]
        self.assertEqual(messages[0].role, "user")  # type: ignore[attr-defined]
        for a, b in zip(messages, messages[1:]):
            self.assertNotEqual(a.role, b.role)  # type: ignore[attr-defined]
        for m in messages:
            self.assertIn(m.role, ("user", "assistant"))  # type: ignore[attr-defined]
            self.assertTrue(m.content)  # type: ignore[attr-defined]

    def assertAllCallsValid(self, *agents: Agent) -> None:
        for agent in agents:
            self.assertTrue(calls(agent))  # type: ignore[attr-defined]
            for msgs in calls(agent):
                self.assertValid(msgs)


class AgentTests(unittest.TestCase):
    def test_label_and_default_provider(self) -> None:
        agent = Agent(AgentSpec("a", "mock", "m1"))
        self.assertEqual(agent.label, "a (mock:m1)")
        self.assertIsInstance(agent.provider, MockProvider)

    def test_respond_strips_and_passes_settings(self) -> None:
        seen: dict = {}

        class Spy(MockProvider):
            def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024):
                seen.update(model=model, system=system, temperature=temperature, max_tokens=max_tokens)
                return "  padded \n"

        agent = Agent(AgentSpec("a", "mock", "m", system="SYS", temperature=0.3, max_tokens=77), Spy())
        self.assertEqual(agent.respond([Message("user", "hi")]), "padded")
        self.assertEqual(seen, {"model": "m", "system": "SYS", "temperature": 0.3, "max_tokens": 77})

    def test_respond_drops_parroted_speaker_tags(self) -> None:
        agent = make("a", ["[llama]: [Moderator]:  I agree.", "[not a tag] stays", "x [b]: y"])
        self.assertEqual(agent.respond([Message("user", "hi")]), "I agree.")
        self.assertEqual(agent.respond([Message("user", "hi")]), "[not a tag] stays")
        self.assertEqual(agent.respond([Message("user", "hi")]), "x [b]: y")

    def test_respond_streams_chunks(self) -> None:
        agent = make("a", ["hello big world"])
        chunks: list[tuple[str, str]] = []
        text = agent.respond([Message("user", "hi")], on_token=lambda s, c: chunks.append((s, c)))
        self.assertEqual(text, "hello big world")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(s == "a" for s, _ in chunks))
        self.assertEqual("".join(c for _, c in chunks).strip(), "hello big world")

    def test_respond_without_on_token_uses_complete(self) -> None:
        class NoStream(MockProvider):
            def stream(self, *a, **k):
                raise AssertionError("stream must not be used")

        agent = Agent(AgentSpec("a", "mock", "m"), NoStream({"replies": ["ok"]}))
        self.assertEqual(agent.respond([Message("user", "x")]), "ok")


class BuildViewTests(InvariantMixin, unittest.TestCase):
    def transcript(self) -> Transcript:
        t = Transcript("chat", "Topic")
        t.add(Turn("user", "Topic", 0, "seed"))
        t.add(Turn("a", "A1", 1, model="mock:a"))
        t.add(Turn("b", "B1", 1, model="mock:b"))
        t.add(Turn("c", "C1", 1, model="mock:c"))
        t.add(Turn("a", "A2", 2, model="mock:a"))
        return t

    def test_perspective_and_merge(self) -> None:
        view = build_view(self.transcript(), "a")
        self.assertEqual([m.role for m in view], ["user", "assistant", "user", "assistant"])
        self.assertEqual(view[0].content, "[Moderator]: Topic")
        self.assertEqual(view[1].content, "A1")
        self.assertEqual(view[2].content, "[b]: B1\n\n[c]: C1")
        self.assertEqual(view[3].content, "A2")
        self.assertValid(view)

    def test_instruction_added_as_new_user_message_after_assistant(self) -> None:
        view = build_view(self.transcript(), "a", "DO IT")
        self.assertEqual(view[-1], Message("user", "DO IT"))
        self.assertValid(view)

    def test_instruction_appended_to_last_user_message(self) -> None:
        view = build_view(self.transcript(), "b", "DO IT")
        self.assertEqual([m.role for m in view], ["user", "assistant", "user", "assistant", "user"][: len(view)])
        self.assertTrue(view[-1].content.endswith("\n\nDO IT"))
        self.assertIn("[a]: A2", view[-1].content)
        self.assertValid(view)

    def test_each_perspective_is_valid(self) -> None:
        for name in ("a", "b", "c", "nobody"):
            self.assertValid(build_view(self.transcript(), name, "go"))

    def test_first_message_user_even_without_seed(self) -> None:
        t = Transcript("chat", "Topic")
        t.add(Turn("a", "I start"))
        view = build_view(t, "a")
        self.assertEqual(view[0], Message("user", "[Moderator]: Topic"))
        self.assertEqual(view[1], Message("assistant", "I start"))
        self.assertValid(view)

    def test_empty_transcript(self) -> None:
        view = build_view(Transcript("chat", "Topic"), "a", "go")
        self.assertEqual(len(view), 1)
        self.assertIn("go", view[0].content)

    def test_seed_from_speaker_named_like_agent_is_still_user(self) -> None:
        t = Transcript("chat", "Topic")
        t.add(Turn("a", "Topic", 0, "seed"))
        self.assertEqual(build_view(t, "a")[0].role, "user")

    def test_does_not_mutate_transcript(self) -> None:
        t = self.transcript()
        before = t.to_dict()
        build_view(t, "b", "x")
        self.assertEqual(t.to_dict(), before)


class ChatTests(InvariantMixin, unittest.TestCase):
    def test_turn_counts_order_and_metadata(self) -> None:
        a, b, c = make("a"), make("b"), make("c")
        seen: list[Turn] = []
        t = run_chat([a, b, c], "Hello", rounds=2, on_turn=seen.append)
        self.assertEqual(t.mode, "chat")
        self.assertEqual(t.topic, "Hello")
        self.assertEqual(len(t.turns), 7)
        self.assertEqual((t.turns[0].speaker, t.turns[0].role, t.turns[0].round), ("user", "seed", 0))
        self.assertEqual([x.speaker for x in t.turns[1:]], ["a", "b", "c"] * 2)
        self.assertEqual([x.round for x in t.turns[1:]], [1, 1, 1, 2, 2, 2])
        self.assertTrue(all(x.role == "agent" for x in t.turns[1:]))
        self.assertEqual(t.turns[1].model, "mock:m-a")
        self.assertEqual(t.turns[0].model, "")
        self.assertTrue(all(x.elapsed >= 0 for x in t.turns))
        self.assertEqual(seen, t.turns)  # on_turn includes the seed, in order
        self.assertAllCallsValid(a, b, c)

    def test_default_rounds_is_three(self) -> None:
        t = run_chat([make("a"), make("b")], "x")
        self.assertEqual(len(t.turns), 1 + 3 * 2)

    def test_first_speaker_sees_topic_and_instruction(self) -> None:
        a, b = make("a"), make("b")
        run_chat([a, b], "Cats?", rounds=1)
        first = calls(a)[0]
        self.assertEqual(len(first), 1)
        self.assertIn("[Moderator]: Cats?", first[0].content)
        self.assertIn("You are a", first[0].content)
        self.assertIn("b", first[0].content)
        second = calls(b)[0]
        self.assertEqual([m.role for m in second], ["user"])
        self.assertIn("[a]:", second[0].content)

    def test_own_history_is_assistant(self) -> None:
        a, b = make("a", ["A1", "A2"]), make("b", ["B1", "B2"])
        run_chat([a, b], "x", rounds=2)
        last = calls(a)[1]
        self.assertEqual([m.role for m in last], ["user", "assistant", "user"])
        self.assertEqual(last[1].content, "A1")
        self.assertIn("[b]: B1", last[2].content)

    def test_stop_phrase_ends_early(self) -> None:
        a = make("a", ["hi", "all done. agreed!"])
        b = make("b", ["yo"])
        t = run_chat([a, b], "x", rounds=5, stop_phrase="AGREED")
        self.assertEqual([x.speaker for x in t.turns[1:]], ["a", "b", "a"])
        self.assertIn("agreed", t.turns[-1].content)

    def test_stop_phrase_in_middle_of_round(self) -> None:
        a, b, c = make("a", ["STOP"]), make("b"), make("c")
        t = run_chat([a, b, c], "x", rounds=3, stop_phrase="STOP")
        self.assertEqual(len(t.turns), 2)
        self.assertEqual(calls(b), [])

    def test_streaming(self) -> None:
        a, b = make("a", ["one two three"]), make("b", ["four five"])
        events: list[tuple[str, str]] = []
        t = run_chat([a, b], "x", rounds=1, on_token=lambda s, c: events.append((s, c)))
        self.assertEqual(t.turns[1].content, "one two three")
        self.assertEqual("".join(c for s, c in events if s == "a").strip(), "one two three")
        self.assertEqual("".join(c for s, c in events if s == "b").strip(), "four five")
        speakers = [s for s, _ in events]
        self.assertEqual(speakers, sorted(speakers))  # a's chunks all before b's

    def test_whitespace_stripped(self) -> None:
        t = run_chat([make("a", ["  x \n"]), make("b")], "t", rounds=1)
        self.assertEqual(t.turns[1].content, "x")

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            run_chat([make("a")], "x")
        with self.assertRaises(ConfigError):
            run_chat([], "x")
        with self.assertRaises(ConfigError):
            run_chat([make("a"), make("b")], "x", rounds=0)
        with self.assertRaises(ConfigError):
            run_chat([make("a"), make("a")], "x")

    def test_provider_error_carries_partial_transcript(self) -> None:
        a, b = make("a"), failing("b", fail_on=1)
        with self.assertRaises(ProviderError) as ctx:
            run_chat([a, b], "x", rounds=2)
        partial = ctx.exception.transcript  # type: ignore[attr-defined]
        self.assertIsInstance(partial, Transcript)
        self.assertEqual([x.speaker for x in partial.turns], ["user", "a"])

    def test_system_prompt_is_not_in_messages(self) -> None:
        a = make("a", system="Be a pirate")
        run_chat([a, make("b")], "x", rounds=1)
        self.assertNotIn("Be a pirate", calls(a)[0][0].content)


class RelayTests(InvariantMixin, unittest.TestCase):
    def test_pipeline(self) -> None:
        a, b, c = make("a", ["ONE"]), make("b", ["TWO"]), make("c", ["THREE"])
        seen: list[Turn] = []
        t = run_relay([a, b, c], "start", on_turn=seen.append)
        self.assertEqual(t.mode, "relay")
        self.assertEqual([x.speaker for x in t.turns], ["user", "a", "b", "c"])
        self.assertEqual([x.round for x in t.turns], [0, 1, 2, 3])
        self.assertEqual([x.content for x in t.turns], ["start", "ONE", "TWO", "THREE"])
        self.assertEqual(seen, t.turns)
        self.assertAllCallsValid(a, b, c)

    def test_only_previous_output_is_visible(self) -> None:
        a, b, c = make("a", ["ONE"]), make("b", ["TWO"]), make("c", ["THREE"])
        run_relay([a, b, c], "start")
        self.assertIn("start", calls(a)[0][0].content)
        self.assertEqual(len(calls(b)[0]), 1)
        self.assertIn("ONE", calls(b)[0][0].content)
        self.assertNotIn("start", calls(b)[0][0].content)
        self.assertIn("TWO", calls(c)[0][0].content)
        self.assertNotIn("ONE", calls(c)[0][0].content)
        self.assertIn("Improve, extend or transform", calls(b)[0][0].content)

    def test_single_agent_and_streaming(self) -> None:
        a = make("a", ["x y"])
        tokens: list[str] = []
        t = run_relay([a], "p", on_token=lambda s, c: tokens.append(c))
        self.assertEqual(len(t.turns), 2)
        self.assertEqual("".join(tokens).strip(), "x y")

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            run_relay([], "x")

    def test_provider_error(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            run_relay([make("a"), failing("b", 1), make("c")], "x")
        self.assertEqual([x.speaker for x in ctx.exception.transcript.turns], ["user", "a"])  # type: ignore[attr-defined]


class DebateTests(InvariantMixin, unittest.TestCase):
    def test_turns_order_and_stances(self) -> None:
        a, b = make("a"), make("b")
        t = run_debate([a, b], "AI is good", rounds=2)
        self.assertEqual(t.mode, "debate")
        self.assertEqual([x.speaker for x in t.turns[1:]], ["a", "b", "a", "b"])
        self.assertEqual([x.round for x in t.turns[1:]], [1, 1, 2, 2])
        self.assertEqual(t.turns[0].role, "seed")
        self.assertIn("for the motion", calls(a)[0][-1].content)
        self.assertIn("against the motion", calls(b)[0][-1].content)
        self.assertIn("Rebut", calls(a)[1][-1].content)
        self.assertNotIn("Rebut", calls(a)[0][-1].content)
        self.assertAllCallsValid(a, b)

    def test_three_agents_alternate_stances(self) -> None:
        agents = [make("a"), make("b"), make("c")]
        run_debate(agents, "m", rounds=1)
        stances = ["for the motion", "against the motion", "for the motion"]
        for agent, stance in zip(agents, stances):
            self.assertIn(stance, calls(agent)[0][-1].content)
        self.assertAllCallsValid(*agents)

    def test_judge(self) -> None:
        a, b = make("a", ["ARG A"]), make("b", ["ARG B"])
        judge = make("j", ["b wins because reasons"])
        seen: list[Turn] = []
        t = run_debate([a, b], "m", rounds=2, judge=judge, on_turn=seen.append)
        self.assertEqual(len(t.turns), 6)
        verdict = t.turns[-1]
        self.assertEqual((verdict.speaker, verdict.role, verdict.round), ("j", "judge", 3))
        self.assertEqual(verdict.model, "mock:m-j")
        self.assertEqual(seen, t.turns)
        jview = calls(judge)[0]
        self.assertAllCallsValid(judge)
        self.assertEqual(len(jview), 1)  # whole debate merged into one user message
        for needle in ("ARG A", "ARG B", "[a]:", "[b]:", "declare a winner", "m"):
            self.assertIn(needle, jview[0].content)

    def test_judge_with_same_name_as_debater_still_valid(self) -> None:
        a, b = make("a"), make("b")
        judge = make("a")
        run_debate([a, b], "m", rounds=1, judge=judge)
        self.assertAllCallsValid(judge)
        self.assertEqual([m.role for m in calls(judge)[0]], ["user"])

    def test_no_judge_means_no_judge_turn(self) -> None:
        t = run_debate([make("a"), make("b")], "m")
        self.assertFalse(any(x.role == "judge" for x in t.turns))
        self.assertEqual(len(t.turns), 1 + 2 * 2)

    def test_streaming_includes_judge(self) -> None:
        events: list[str] = []
        run_debate(
            [make("a"), make("b")], "m", rounds=1, judge=make("j"),
            on_token=lambda s, c: events.append(s),
        )
        self.assertEqual(sorted(set(events)), ["a", "b", "j"])

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            run_debate([make("a")], "m")
        with self.assertRaises(ConfigError):
            run_debate([make("a"), make("b")], "m", rounds=0)

    def test_judge_error_carries_transcript(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            run_debate([make("a"), make("b")], "m", rounds=1, judge=failing("j", 1))
        self.assertEqual(len(ctx.exception.transcript.turns), 3)  # type: ignore[attr-defined]


class PanelTests(InvariantMixin, unittest.TestCase):
    def test_parallel_answers_in_agent_order(self) -> None:
        class Slow(MockProvider):
            def __init__(self, delay: float, options=None):
                super().__init__(options)
                self.delay = delay

            def complete(self, model, messages, **kw):
                time.sleep(self.delay)
                return super().complete(model, messages, **kw)

        a = Agent(AgentSpec("a", "mock", "ma"), Slow(0.15, {"replies": ["A"]}))
        b = Agent(AgentSpec("b", "mock", "mb"), Slow(0.0, {"replies": ["B"]}))
        c = Agent(AgentSpec("c", "mock", "mc"), Slow(0.05, {"replies": ["C"]}))
        seen: list[Turn] = []
        t = run_panel([a, b, c], "Q?", on_turn=seen.append)
        self.assertEqual(t.mode, "panel")
        self.assertEqual([x.speaker for x in t.turns], ["user", "a", "b", "c"])
        self.assertEqual([x.content for x in t.turns[1:]], ["A", "B", "C"])
        self.assertEqual([x.round for x in t.turns[1:]], [1, 1, 1])
        self.assertEqual(seen, t.turns)
        self.assertGreaterEqual(t.turns[1].elapsed, 0.1)
        self.assertAllCallsValid(a, b, c)

    def test_runs_concurrently(self) -> None:
        barrier = threading.Barrier(3, timeout=5)

        class Meet(MockProvider):
            def complete(self, model, messages, **kw):
                barrier.wait()  # deadlocks (BrokenBarrierError) if not concurrent
                return super().complete(model, messages, **kw)

        agents = [Agent(AgentSpec(n, "mock", "m"), Meet()) for n in "abc"]
        t = run_panel(agents, "Q?")
        self.assertEqual(len(t.turns), 4)

    def test_agents_answer_independently(self) -> None:
        a, b = make("a", ["ANS-A"]), make("b", ["ANS-B"])
        run_panel([a, b], "Q?", parallel=True)
        run_panel([a, b], "Q2?", parallel=False)
        for agent, other in ((a, "ANS-B"), (b, "ANS-A")):
            for msgs in calls(agent):
                self.assertEqual(len(msgs), 1)
                self.assertNotIn(other, msgs[0].content)
                self.assertIn("independently", msgs[0].content)

    def test_synthesizer(self) -> None:
        a, b = make("a", ["ANS-A"]), make("b", ["ANS-B"])
        s = make("s", ["merged answer"])
        t = run_panel([a, b], "Q?", synthesizer=s)
        last = t.turns[-1]
        self.assertEqual((last.speaker, last.role, last.content, last.round), ("s", "synthesis", "merged answer", 2))
        self.assertEqual(len(t.turns), 4)
        view = calls(s)[0]
        self.assertAllCallsValid(s)
        for needle in ("[a]: ANS-A", "[b]: ANS-B", "disagree"):
            self.assertIn(needle, view[0].content)

    def test_sequential_streams_tokens_parallel_does_not(self) -> None:
        events: list[str] = []
        agents = [make("a", ["x y"]), make("b", ["z w"])]
        run_panel(agents, "Q?", parallel=False, on_token=lambda s, c: events.append(s))
        self.assertEqual(sorted(set(events)), ["a", "b"])
        events.clear()
        run_panel(agents, "Q?", parallel=True, on_token=lambda s, c: events.append(s))
        self.assertEqual(events, [])

    def test_sequential_order(self) -> None:
        t = run_panel([make("a"), make("b")], "Q?", parallel=False)
        self.assertEqual([x.speaker for x in t.turns], ["user", "a", "b"])

    def test_single_agent(self) -> None:
        t = run_panel([make("a", ["only"])], "Q?")
        self.assertEqual([x.content for x in t.turns], ["Q?", "only"])

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            run_panel([], "Q?")

    def test_provider_error_parallel(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            run_panel([make("a"), failing("b", 1), make("c")], "Q?")
        partial = ctx.exception.transcript  # type: ignore[attr-defined]
        self.assertEqual([x.speaker for x in partial.turns], ["user", "a"])

    def test_provider_error_sequential_and_synth(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            run_panel([make("a"), failing("b", 1)], "Q?", parallel=False)
        self.assertEqual([x.speaker for x in ctx.exception.transcript.turns], ["user", "a"])  # type: ignore[attr-defined]
        with self.assertRaises(ProviderError) as ctx:
            run_panel([make("a")], "Q?", synthesizer=failing("s", 1))
        self.assertEqual(len(ctx.exception.transcript.turns), 2)  # type: ignore[attr-defined]


class ExportIntegrationTests(unittest.TestCase):
    def test_run_roundtrips_and_renders(self) -> None:
        t = run_debate([make("a"), make("b")], "Motion", rounds=1, judge=make("j"))
        self.assertEqual(Transcript.from_dict(t.to_dict()), t)
        md = t.to_markdown()
        self.assertIn("# aic debate: Motion", md)
        self.assertIn("### Judge: j · mock:m-j", md)


if __name__ == "__main__":
    unittest.main()
