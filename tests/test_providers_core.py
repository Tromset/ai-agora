"""Tests for the provider base class, MockProvider and the provider registry."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from aiconnect.errors import ConfigError
from aiconnect.providers import REGISTRY, get_provider, provider_class
from aiconnect.providers.base import Provider
from aiconnect.providers.mock import MockProvider
from aiconnect.types import Message


class MockProviderTests(unittest.TestCase):
    def test_default_reply_format(self) -> None:
        provider = MockProvider()
        reply = provider.complete("m1", [Message("user", "What is life?")])
        self.assertEqual(reply, "[m1] reply #1 to: What is life?")

    def test_default_reply_counter_increments(self) -> None:
        provider = MockProvider()
        provider.complete("m", [Message("user", "a")])
        self.assertEqual(provider.complete("m", [Message("user", "b")]), "[m] reply #2 to: b")

    def test_default_reply_truncates_to_60_chars(self) -> None:
        reply = MockProvider().complete("m", [Message("user", "x" * 100)])
        self.assertEqual(reply, "[m] reply #1 to: " + "x" * 60)

    def test_default_reply_uses_last_user_message(self) -> None:
        messages = [Message("user", "first"), Message("assistant", "mid"), Message("user", "last")]
        self.assertTrue(MockProvider().complete("m", messages).endswith("to: last"))

    def test_default_reply_without_user_message(self) -> None:
        self.assertEqual(MockProvider().complete("m", []), "[m] reply #1 to: ")

    def test_replies_cycle(self) -> None:
        provider = MockProvider({"replies": ["one", "two", "three"]})
        got = [provider.complete("m", [Message("user", "q")]) for _ in range(5)]
        self.assertEqual(got, ["one", "two", "three", "one", "two"])

    def test_replies_are_per_instance(self) -> None:
        opts = {"replies": ["a", "b"]}
        first, second = MockProvider(opts), MockProvider(opts)
        first.complete("m", [Message("user", "q")])
        self.assertEqual(second.complete("m", [Message("user", "q")]), "a")

    def test_stream_words_rejoin_to_reply(self) -> None:
        provider = MockProvider({"replies": ["the quick brown fox"]})
        chunks = list(provider.stream("m", [Message("user", "q")]))
        self.assertEqual(len(chunks), 4)
        self.assertEqual("".join(chunks).strip(), "the quick brown fox")

    def test_stream_default_reply_rejoins(self) -> None:
        chunks = list(MockProvider().stream("m", [Message("user", "hello there")]))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).strip(), "[m] reply #1 to: hello there")

    def test_calls_are_recorded(self) -> None:
        provider = MockProvider()
        first = [Message("user", "one")]
        second = [Message("user", "one"), Message("assistant", "r"), Message("user", "two")]
        provider.complete("m", first)
        list(provider.stream("m", second))
        self.assertEqual(provider.calls, [first, second])

    def test_calls_hold_a_snapshot(self) -> None:
        provider = MockProvider()
        messages = [Message("user", "one")]
        provider.complete("m", messages)
        messages.append(Message("assistant", "later"))
        self.assertEqual(len(provider.calls[0]), 1)

    def test_no_api_key_needed_for_construction(self) -> None:
        provider = MockProvider()
        self.assertEqual(provider.name, "mock")
        self.assertIsNone(provider.api_key_env)


class _KeyedProvider(Provider):
    name = "keyed"
    api_key_env = "AIC_TEST_CLASS_KEY"
    default_base_url = "https://example.test/v1/"

    def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> str:
        return "ok"


class _KeylessProvider(Provider):
    name = "keyless"

    def complete(self, model, messages, *, system="", temperature=None, max_tokens=1024) -> str:
        return "ok"


class ProviderBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("AIC_TEST_CLASS_KEY", "AIC_TEST_OTHER_KEY"):
            os.environ.pop(var, None)

    def test_options_api_key_has_top_priority(self) -> None:
        os.environ["AIC_TEST_CLASS_KEY"] = "from-class-env"
        os.environ["AIC_TEST_OTHER_KEY"] = "from-option-env"
        provider = _KeyedProvider({"api_key": "literal", "api_key_env": "AIC_TEST_OTHER_KEY"})
        self.assertEqual(provider.api_key(), "literal")

    def test_options_api_key_env_beats_class_env(self) -> None:
        os.environ["AIC_TEST_CLASS_KEY"] = "from-class-env"
        os.environ["AIC_TEST_OTHER_KEY"] = "from-option-env"
        provider = _KeyedProvider({"api_key_env": "AIC_TEST_OTHER_KEY"})
        self.assertEqual(provider.api_key(), "from-option-env")

    def test_class_env_is_the_default(self) -> None:
        os.environ["AIC_TEST_CLASS_KEY"] = "from-class-env"
        self.assertEqual(_KeyedProvider().api_key(), "from-class-env")

    def test_missing_key_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            _KeyedProvider().api_key()
        self.assertIn("AIC_TEST_CLASS_KEY", str(ctx.exception))

    def test_option_env_unset_does_not_fall_back_to_class_env(self) -> None:
        os.environ["AIC_TEST_CLASS_KEY"] = "from-class-env"
        with self.assertRaises(ConfigError):
            _KeyedProvider({"api_key_env": "AIC_TEST_OTHER_KEY"}).api_key()

    def test_empty_env_value_counts_as_missing(self) -> None:
        os.environ["AIC_TEST_CLASS_KEY"] = ""
        with self.assertRaises(ConfigError):
            _KeyedProvider().api_key()

    def test_provider_without_key_env_raises(self) -> None:
        with self.assertRaises(ConfigError):
            _KeylessProvider().api_key()

    def test_keyless_provider_accepts_explicit_key(self) -> None:
        self.assertEqual(_KeylessProvider({"api_key": "k"}).api_key(), "k")

    def test_base_url_defaults_and_strips_trailing_slash(self) -> None:
        self.assertEqual(_KeyedProvider().base_url, "https://example.test/v1")
        self.assertEqual(_KeyedProvider({"base_url": "http://x/y/"}).base_url, "http://x/y")

    def test_timeout_default_and_override(self) -> None:
        self.assertEqual(_KeyedProvider().timeout, 120.0)
        self.assertEqual(_KeyedProvider({"timeout": "5"}).timeout, 5.0)

    def test_options_are_copied(self) -> None:
        opts = {"api_key": "k"}
        provider = _KeyedProvider(opts)
        opts["api_key"] = "changed"
        self.assertEqual(provider.api_key(), "k")

    def test_default_stream_yields_complete_once(self) -> None:
        self.assertEqual(list(_KeyedProvider().stream("m", [Message("user", "q")])), ["ok"])


class RegistryTests(unittest.TestCase):
    def test_get_mock_provider(self) -> None:
        provider = get_provider("mock")
        self.assertIsInstance(provider, MockProvider)

    def test_get_provider_passes_options(self) -> None:
        provider = get_provider("mock", {"replies": ["hi"]})
        self.assertEqual(provider.complete("m", [Message("user", "q")]), "hi")

    def test_unknown_provider_lists_known_names(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            get_provider("nope")
        message = str(ctx.exception)
        self.assertIn("nope", message)
        for name in REGISTRY:
            self.assertIn(name, message)

    def test_unknown_provider_class_lookup(self) -> None:
        with self.assertRaises(ConfigError):
            provider_class("")

    def test_provider_class_returns_class(self) -> None:
        self.assertIs(provider_class("mock"), MockProvider)

    def test_expected_names_registered(self) -> None:
        expected = {
            "anthropic", "openai", "openrouter", "groq", "mistral",
            "deepseek", "xai", "gemini", "ollama", "mock",
        }
        self.assertEqual(set(REGISTRY), expected)

    def test_gemini_and_ollama_resolve(self) -> None:
        self.assertEqual(provider_class("gemini").name, "gemini")
        self.assertEqual(provider_class("ollama").name, "ollama")


if __name__ == "__main__":
    unittest.main()
