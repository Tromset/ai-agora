"""Offline tests for the Anthropic and OpenAI(-compatible) providers."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from aiconnect.errors import ConfigError, ProviderError
from aiconnect.providers import get_provider
from aiconnect.providers.anthropic import AnthropicProvider
from aiconnect.providers.openai import (
    DeepSeekProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    XAIProvider,
)
from aiconnect.types import Message

MSGS = [Message("user", "hi"), Message("assistant", "hello"), Message("user", "again")]
WIRE_MSGS = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "again"},
]

OPENAI_PRESETS = [
    (OpenAIProvider, "openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    (OpenRouterProvider, "openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    (GroqProvider, "groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    (MistralProvider, "mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    (DeepSeekProvider, "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    (XAIProvider, "xai", "XAI_API_KEY", "https://api.x.ai/v1"),
]


def sse(*events: dict | str) -> list[str]:
    """Build realistic SSE lines (blank lines are dropped by http.post_lines)."""
    lines: list[str] = []
    for ev in events:
        if isinstance(ev, dict):
            if "type" in ev:
                lines.append(f"event: {ev['type']}")
            lines.append("data: " + json.dumps(ev))
        else:
            lines.append(ev)
    return lines


class AnthropicTests(unittest.TestCase):
    def setUp(self):
        self.p = AnthropicProvider({"api_key": "sk-test"})

    def test_metadata(self):
        self.assertEqual(AnthropicProvider.name, "anthropic")
        self.assertEqual(AnthropicProvider.api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(AnthropicProvider.default_base_url, "https://api.anthropic.com/v1")

    def test_complete_request_shape(self):
        reply = {"content": [{"type": "text", "text": "ok"}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
            out = self.p.complete(
                "claude-x", MSGS, system="be brief", temperature=0.3, max_tokens=77
            )
        self.assertEqual(out, "ok")
        url, headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "sk-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(
            payload,
            {
                "model": "claude-x",
                "max_tokens": 77,
                "messages": WIRE_MSGS,
                "system": "be brief",
                "temperature": 0.3,
            },
        )

    def test_optional_fields_omitted(self):
        with mock.patch(
            "aiconnect.http.post_json", return_value={"content": [{"type": "text", "text": "x"}]}
        ) as post:
            self.p.complete("m", MSGS)
        payload = post.call_args.args[2]
        self.assertNotIn("system", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("stream", payload)
        self.assertEqual(payload["max_tokens"], 1024)

    def test_temperature_zero_is_sent(self):
        with mock.patch(
            "aiconnect.http.post_json", return_value={"content": [{"type": "text", "text": "x"}]}
        ) as post:
            self.p.complete("m", MSGS, temperature=0.0)
        self.assertEqual(post.call_args.args[2]["temperature"], 0.0)

    def test_custom_base_url(self):
        p = AnthropicProvider({"api_key": "k", "base_url": "http://proxy.local/v1/"})
        with mock.patch(
            "aiconnect.http.post_json", return_value={"content": [{"type": "text", "text": "x"}]}
        ) as post:
            p.complete("m", MSGS)
        self.assertEqual(post.call_args.args[0], "http://proxy.local/v1/messages")

    def test_complete_joins_text_blocks_only(self):
        reply = {
            "content": [
                {"type": "text", "text": "Hello, "},
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                {"type": "text", "text": "world"},
            ]
        }
        with mock.patch("aiconnect.http.post_json", return_value=reply):
            self.assertEqual(self.p.complete("m", MSGS), "Hello, world")

    def test_complete_malformed(self):
        for bad in ({}, {"content": None}, {"content": [1, 2]}, {"error": "x"}):
            with self.subTest(bad=bad), mock.patch("aiconnect.http.post_json", return_value=bad):
                with self.assertRaises(ProviderError):
                    self.p.complete("m", MSGS)

    def test_missing_key(self):
        p = AnthropicProvider({})
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "aiconnect.http.post_json"
        ) as post:
            with self.assertRaises(ConfigError) as ctx:
                p.complete("m", MSGS)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        post.assert_not_called()

    def test_key_from_env(self):
        p = AnthropicProvider({})
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}), mock.patch(
            "aiconnect.http.post_json", return_value={"content": [{"type": "text", "text": "x"}]}
        ) as post:
            p.complete("m", MSGS)
        self.assertEqual(post.call_args.args[1]["x-api-key"], "from-env")

    def test_stream_parsing(self):
        lines = sse(
            {"type": "message_start", "message": {"id": "msg_1", "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "ping"},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        )
        with mock.patch("aiconnect.http.post_lines", return_value=iter(lines)) as post:
            chunks = list(self.p.stream("m", MSGS, system="s", temperature=0.5, max_tokens=9))
        self.assertEqual(chunks, ["Hel", "lo"])
        url, headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "sk-test")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["system"], "s")
        self.assertEqual(payload["max_tokens"], 9)

    def test_stream_error_event(self):
        lines = sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a"}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        got: list[str] = []
        with mock.patch("aiconnect.http.post_lines", return_value=iter(lines)):
            with self.assertRaises(ProviderError) as ctx:
                for chunk in self.p.stream("m", MSGS):
                    got.append(chunk)
        self.assertEqual(got, ["a"])
        self.assertIn("Overloaded", str(ctx.exception))

    def test_stream_bad_json(self):
        with mock.patch("aiconnect.http.post_lines", return_value=iter(["data: {not json"])):
            with self.assertRaises(ProviderError):
                list(self.p.stream("m", MSGS))

    def test_stream_malformed_delta(self):
        lines = sse({"type": "content_block_delta", "index": 0})
        with mock.patch("aiconnect.http.post_lines", return_value=iter(lines)):
            with self.assertRaises(ProviderError):
                list(self.p.stream("m", MSGS))

    def test_key_not_in_error_message(self):
        with mock.patch("aiconnect.http.post_json", return_value={}):
            with self.assertRaises(ProviderError) as ctx:
                self.p.complete("m", MSGS)
        self.assertNotIn("sk-test", str(ctx.exception))


class OpenAITests(unittest.TestCase):
    def test_preset_metadata(self):
        for cls, name, env, base in OPENAI_PRESETS:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.name, name)
                self.assertEqual(cls.api_key_env, env)
                self.assertEqual(cls.default_base_url, base)
                self.assertEqual(cls({"api_key": "k"}).base_url, base)

    def test_max_tokens_field_per_class(self):
        self.assertEqual(OpenAIProvider.max_tokens_field, "max_completion_tokens")
        for cls, name, _env, _base in OPENAI_PRESETS:
            with self.subTest(cls=cls.__name__):
                expected = "max_completion_tokens" if name == "openai" else "max_tokens"
                self.assertEqual(cls.max_tokens_field, expected)
                reply = {"choices": [{"message": {"content": "x"}}]}
                with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
                    cls({"api_key": "k"}).complete("m", MSGS, max_tokens=55)
                payload = post.call_args.args[2]
                self.assertEqual(payload[expected], 55)
                other = "max_tokens" if expected == "max_completion_tokens" else "max_completion_tokens"
                self.assertNotIn(other, payload)

    def test_complete_request_shape(self):
        p = OpenAIProvider({"api_key": "sk-oa"})
        reply = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
            out = p.complete("gpt-x", MSGS, system="sys", temperature=0.9, max_tokens=10)
        self.assertEqual(out, "pong")
        url, headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-oa")
        self.assertEqual(
            payload,
            {
                "model": "gpt-x",
                "messages": [{"role": "system", "content": "sys"}] + WIRE_MSGS,
                "max_completion_tokens": 10,
                "temperature": 0.9,
            },
        )

    def test_optional_fields_omitted(self):
        p = OpenAIProvider({"api_key": "k"})
        reply = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
            p.complete("m", MSGS)
        payload = post.call_args.args[2]
        self.assertNotIn("temperature", payload)
        self.assertNotIn("stream", payload)
        self.assertEqual(payload["messages"], WIRE_MSGS)  # no system message

    def test_temperature_zero_is_sent(self):
        p = OpenAIProvider({"api_key": "k"})
        reply = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
            p.complete("m", MSGS, temperature=0)
        self.assertEqual(post.call_args.args[2]["temperature"], 0)

    def test_custom_base_url_compatible_server(self):
        p = OpenAIProvider({"api_key": "k", "base_url": "http://localhost:8000/v1/"})
        reply = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
            p.complete("m", MSGS)
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/v1/chat/completions")

    def test_preset_url_and_auth(self):
        for cls, _name, _env, base in OPENAI_PRESETS[1:]:
            with self.subTest(cls=cls.__name__):
                reply = {"choices": [{"message": {"content": "x"}}]}
                with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
                    cls({"api_key": "tok"}).complete("m", MSGS)
                self.assertEqual(post.call_args.args[0], f"{base}/chat/completions")
                self.assertEqual(post.call_args.args[1]["Authorization"], "Bearer tok")

    def test_complete_none_content_is_empty(self):
        p = OpenAIProvider({"api_key": "k"})
        reply = {"choices": [{"message": {"content": None}}]}
        with mock.patch("aiconnect.http.post_json", return_value=reply):
            self.assertEqual(p.complete("m", MSGS), "")

    def test_complete_malformed(self):
        p = OpenAIProvider({"api_key": "k"})
        for bad in ({}, {"choices": []}, {"choices": None}, {"choices": [{}]}, {"error": {}}):
            with self.subTest(bad=bad), mock.patch("aiconnect.http.post_json", return_value=bad):
                with self.assertRaises(ProviderError):
                    p.complete("m", MSGS)

    def test_missing_key(self):
        for cls, _name, env, _base in OPENAI_PRESETS:
            with self.subTest(cls=cls.__name__):
                with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                    "aiconnect.http.post_json"
                ) as post:
                    with self.assertRaises(ConfigError) as ctx:
                        cls({}).complete("m", MSGS)
                self.assertIn(env, str(ctx.exception))
                post.assert_not_called()

    def test_key_from_env_and_custom_env_name(self):
        reply = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "g", "MY_KEY": "custom"}):
            with mock.patch("aiconnect.http.post_json", return_value=reply) as post:
                GroqProvider({}).complete("m", MSGS)
                self.assertEqual(post.call_args.args[1]["Authorization"], "Bearer g")
                OpenAIProvider({"api_key_env": "MY_KEY"}).complete("m", MSGS)
                self.assertEqual(post.call_args.args[1]["Authorization"], "Bearer custom")

    def test_stream_parsing(self):
        p = OpenAIProvider({"api_key": "k"})

        def chunk(delta: dict, finish=None) -> dict:
            return {
                "id": "c1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        lines = sse(
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, "stop"),
            {"id": "c1", "choices": [], "usage": {"total_tokens": 3}},
            "data: [DONE]",
            "data: " + json.dumps(chunk({"content": "ignored"})),
        )
        with mock.patch("aiconnect.http.post_lines", return_value=iter(lines)) as post:
            chunks = list(p.stream("m", MSGS, system="s", max_tokens=5))
        self.assertEqual(chunks, ["Hel", "lo"])
        url, headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "s"})
        self.assertEqual(payload["max_completion_tokens"], 5)

    def test_stream_compatible_uses_max_tokens(self):
        p = GroqProvider({"api_key": "k"})
        with mock.patch("aiconnect.http.post_lines", return_value=iter([])) as post:
            self.assertEqual(list(p.stream("m", MSGS, max_tokens=8)), [])
        payload = post.call_args.args[2]
        self.assertEqual(payload["max_tokens"], 8)
        self.assertNotIn("max_completion_tokens", payload)

    def test_stream_bad_json_and_shape(self):
        p = OpenAIProvider({"api_key": "k"})
        for line in ("data: {oops", 'data: {"nochoices": 1}', 'data: {"choices": [{"delta": 5}]}'):
            with self.subTest(line=line):
                with mock.patch("aiconnect.http.post_lines", return_value=iter([line])):
                    with self.assertRaises(ProviderError):
                        list(p.stream("m", MSGS))

    def test_stream_error_payload(self):
        p = OpenAIProvider({"api_key": "k"})
        line = 'data: {"error": {"message": "rate limited"}}'
        with mock.patch("aiconnect.http.post_lines", return_value=iter([line])):
            with self.assertRaises(ProviderError) as ctx:
                list(p.stream("m", MSGS))
        self.assertIn("rate limited", str(ctx.exception))


class RegistryTests(unittest.TestCase):
    def test_all_reachable_via_get_provider(self):
        expected = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "openrouter": OpenRouterProvider,
            "groq": GroqProvider,
            "mistral": MistralProvider,
            "deepseek": DeepSeekProvider,
            "xai": XAIProvider,
        }
        for name, cls in expected.items():
            with self.subTest(name=name):
                provider = get_provider(name, {"api_key": "k"})
                self.assertIsInstance(provider, cls)
                self.assertEqual(provider.name, name)

    def test_get_provider_passes_options(self):
        p = get_provider("openai", {"base_url": "http://x/v1", "timeout": 5, "api_key": "k"})
        self.assertEqual(p.base_url, "http://x/v1")
        self.assertEqual(p.timeout, 5.0)


if __name__ == "__main__":
    unittest.main()
