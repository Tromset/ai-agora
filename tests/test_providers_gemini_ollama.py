"""Tests for the Gemini and Ollama providers. Never touches the network."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from aiconnect.errors import ConfigError, ProviderError
from aiconnect.providers.gemini import GeminiProvider
from aiconnect.providers.ollama import OllamaProvider
from aiconnect.types import Message

POST_JSON = "aiconnect.http.post_json"
POST_LINES = "aiconnect.http.post_lines"

CONVERSATION = [
    Message("user", "hi"),
    Message("assistant", "hello"),
    Message("user", "how are you?"),
]


def gemini_reply(*texts: str, finish: str = "STOP") -> dict:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": t} for t in texts]}, "finishReason": finish}
        ]
    }


def sse(obj: dict) -> str:
    return "data: " + json.dumps(obj)


class EnvTestCase(unittest.TestCase):
    """Runs each test with a scrubbed set of provider-related environment variables."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_HOST"):
            os.environ.pop(var, None)


class GeminiCompleteTests(EnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "secret-key-123"
        self.provider = GeminiProvider()

    def complete(self, reply=None, **kwargs):
        with mock.patch(POST_JSON, return_value=gemini_reply("ok") if reply is None else reply) as post:
            result = self.provider.complete("gemini-2.0-flash", CONVERSATION, **kwargs)
        return result, post

    def test_class_attributes(self) -> None:
        self.assertEqual(GeminiProvider.name, "gemini")
        self.assertEqual(GeminiProvider.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(
            self.provider.base_url, "https://generativelanguage.googleapis.com/v1beta"
        )

    def test_url_and_key_in_header_not_url(self) -> None:
        _, post = self.complete()
        url, headers, _payload = post.call_args.args[:3]
        self.assertEqual(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        )
        self.assertEqual(headers["x-goog-api-key"], "secret-key-123")
        self.assertNotIn("secret-key-123", url)
        self.assertNotIn("key=", url)

    def test_models_prefix_is_not_duplicated(self) -> None:
        with mock.patch(POST_JSON, return_value=gemini_reply("ok")) as post:
            self.provider.complete("models/gemini-2.0-flash", CONVERSATION)
        self.assertIn("/models/gemini-2.0-flash:generateContent", post.call_args.args[0])
        self.assertNotIn("models/models", post.call_args.args[0])

    def test_role_mapping(self) -> None:
        _, post = self.complete()
        contents = post.call_args.args[2]["contents"]
        self.assertEqual([c["role"] for c in contents], ["user", "model", "user"])
        self.assertEqual(contents[1]["parts"], [{"text": "hello"}])
        self.assertEqual(contents[2]["parts"], [{"text": "how are you?"}])

    def test_system_instruction_present(self) -> None:
        _, post = self.complete(system="Be brief.")
        payload = post.call_args.args[2]
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "Be brief."}]})

    def test_system_instruction_absent_when_empty(self) -> None:
        _, post = self.complete(system="")
        self.assertNotIn("systemInstruction", post.call_args.args[2])

    def test_generation_config(self) -> None:
        _, post = self.complete(temperature=0.3, max_tokens=77)
        self.assertEqual(
            post.call_args.args[2]["generationConfig"],
            {"maxOutputTokens": 77, "temperature": 0.3},
        )

    def test_temperature_omitted_when_none(self) -> None:
        _, post = self.complete(max_tokens=50)
        self.assertEqual(post.call_args.args[2]["generationConfig"], {"maxOutputTokens": 50})

    def test_temperature_zero_is_sent(self) -> None:
        _, post = self.complete(temperature=0.0)
        self.assertEqual(post.call_args.args[2]["generationConfig"]["temperature"], 0.0)

    def test_timeout_is_forwarded(self) -> None:
        provider = GeminiProvider({"timeout": 9})
        with mock.patch(POST_JSON, return_value=gemini_reply("ok")) as post:
            provider.complete("m", CONVERSATION)
        self.assertEqual(post.call_args.kwargs["timeout"], 9.0)

    def test_joins_multiple_parts(self) -> None:
        result, _ = self.complete(gemini_reply("Hello, ", "world"))
        self.assertEqual(result, "Hello, world")

    def test_custom_base_url(self) -> None:
        provider = GeminiProvider({"base_url": "http://proxy.local/v1beta/"})
        with mock.patch(POST_JSON, return_value=gemini_reply("ok")) as post:
            provider.complete("m", CONVERSATION)
        self.assertEqual(post.call_args.args[0], "http://proxy.local/v1beta/models/m:generateContent")

    def test_blocked_prompt(self) -> None:
        blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
        with self.assertRaises(ProviderError) as ctx:
            self.complete(blocked)
        self.assertIn("SAFETY", str(ctx.exception))

    def test_error_object(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            self.complete({"error": {"code": 400, "message": "API key not valid"}})
        self.assertIn("API key not valid", str(ctx.exception))

    def test_non_stop_finish_without_text(self) -> None:
        reply = {"candidates": [{"finishReason": "SAFETY"}]}
        with self.assertRaises(ProviderError) as ctx:
            self.complete(reply)
        self.assertIn("SAFETY", str(ctx.exception))

    def test_malformed_responses(self) -> None:
        for bad in ({}, {"foo": "bar"}, [], "text", {"candidates": ["oops"]}, {"candidates": [[]]}):
            with self.subTest(bad=bad), self.assertRaises(ProviderError):
                self.complete(bad)

    def test_provider_error_from_http_propagates(self) -> None:
        with mock.patch(POST_JSON, side_effect=ProviderError("HTTP 403 from x")):
            with self.assertRaises(ProviderError):
                self.provider.complete("m", CONVERSATION)


class GeminiKeyTests(EnvTestCase):
    def test_gemini_key(self) -> None:
        os.environ["GEMINI_API_KEY"] = "g"
        self.assertEqual(GeminiProvider().api_key(), "g")

    def test_google_key_fallback(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "goog"
        self.assertEqual(GeminiProvider().api_key(), "goog")

    def test_gemini_key_wins_over_google_key(self) -> None:
        os.environ["GEMINI_API_KEY"] = "g"
        os.environ["GOOGLE_API_KEY"] = "goog"
        self.assertEqual(GeminiProvider().api_key(), "g")

    def test_fallback_key_reaches_header(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "goog"
        with mock.patch(POST_JSON, return_value=gemini_reply("ok")) as post:
            GeminiProvider().complete("m", CONVERSATION)
        self.assertEqual(post.call_args.args[1]["x-goog-api-key"], "goog")

    def test_options_api_key_wins(self) -> None:
        os.environ["GEMINI_API_KEY"] = "g"
        self.assertEqual(GeminiProvider({"api_key": "explicit"}).api_key(), "explicit")

    def test_options_api_key_env_wins_over_defaults(self) -> None:
        os.environ["GEMINI_API_KEY"] = "g"
        os.environ["MY_KEY"] = "mine"
        self.assertEqual(GeminiProvider({"api_key_env": "MY_KEY"}).api_key(), "mine")

    def test_options_api_key_env_unset_is_error(self) -> None:
        os.environ["GEMINI_API_KEY"] = "g"
        with self.assertRaises(ConfigError):
            GeminiProvider({"api_key_env": "SURELY_NOT_SET_XYZ"}).api_key()

    def test_missing_key_is_config_error_without_leaking(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            GeminiProvider().api_key()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_missing_key_stops_before_network(self) -> None:
        with mock.patch(POST_JSON) as post, self.assertRaises(ConfigError):
            GeminiProvider().complete("m", CONVERSATION)
        post.assert_not_called()


class GeminiStreamTests(EnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "secret-key-123"
        self.provider = GeminiProvider()

    def stream(self, lines, **kwargs):
        with mock.patch(POST_LINES, return_value=iter(lines)) as post:
            chunks = list(self.provider.stream("gemini-2.0-flash", CONVERSATION, **kwargs))
        return chunks, post

    def test_sse_chunks(self) -> None:
        lines = [
            sse(gemini_reply("Hel")),
            "",
            sse(gemini_reply("lo ", "wor")),
            sse(gemini_reply("ld", finish="STOP")),
        ]
        chunks, post = self.stream(lines)
        self.assertEqual(chunks, ["Hel", "lo wor", "ld"])
        url, headers, payload = post.call_args.args[:3]
        self.assertTrue(url.endswith("/models/gemini-2.0-flash:streamGenerateContent?alt=sse"))
        self.assertEqual(headers["x-goog-api-key"], "secret-key-123")
        self.assertNotIn("secret-key-123", url)
        self.assertEqual([c["role"] for c in payload["contents"]], ["user", "model", "user"])

    def test_ignores_non_data_lines_and_empty_chunks(self) -> None:
        lines = [
            ": keep-alive",
            "event: message",
            sse(gemini_reply("a")),
            sse({"candidates": [{"finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 3}}),
        ]
        chunks, _ = self.stream(lines)
        self.assertEqual(chunks, ["a"])

    def test_stream_payload_matches_complete_payload(self) -> None:
        _, post = self.stream([sse(gemini_reply("x"))], system="sys", temperature=0.5, max_tokens=9)
        payload = post.call_args.args[2]
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "sys"}]})
        self.assertEqual(payload["generationConfig"], {"maxOutputTokens": 9, "temperature": 0.5})

    def test_stream_blocked_prompt(self) -> None:
        lines = [sse({"promptFeedback": {"blockReason": "OTHER"}})]
        with self.assertRaises(ProviderError) as ctx:
            self.stream(lines)
        self.assertIn("OTHER", str(ctx.exception))

    def test_stream_error_object(self) -> None:
        lines = [sse(gemini_reply("partial")), sse({"error": {"message": "quota exceeded"}})]
        with self.assertRaises(ProviderError) as ctx:
            self.stream(lines)
        self.assertIn("quota exceeded", str(ctx.exception))

    def test_stream_invalid_json(self) -> None:
        with self.assertRaises(ProviderError):
            self.stream(["data: {not json"])

    def test_stream_malformed_chunk(self) -> None:
        with self.assertRaises(ProviderError):
            self.stream([sse({"weird": True})])


class OllamaTests(EnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.provider = OllamaProvider()

    def complete(self, reply=None, **kwargs):
        reply = reply if reply is not None else {"message": {"role": "assistant", "content": "ok"}, "done": True}
        with mock.patch(POST_JSON, return_value=reply) as post:
            result = self.provider.complete("llama3.2", CONVERSATION, **kwargs)
        return result, post

    def test_class_attributes(self) -> None:
        self.assertEqual(OllamaProvider.name, "ollama")
        self.assertIsNone(OllamaProvider.api_key_env)

    def test_default_base_url(self) -> None:
        self.assertEqual(OllamaProvider().base_url, "http://localhost:11434")

    def test_ollama_host_without_scheme(self) -> None:
        os.environ["OLLAMA_HOST"] = "10.0.0.5:11500"
        self.assertEqual(OllamaProvider().base_url, "http://10.0.0.5:11500")

    def test_ollama_host_with_scheme_and_trailing_slash(self) -> None:
        os.environ["OLLAMA_HOST"] = "https://ollama.example.com/"
        self.assertEqual(OllamaProvider().base_url, "https://ollama.example.com")

    def test_ollama_host_empty_uses_default(self) -> None:
        os.environ["OLLAMA_HOST"] = "  "
        self.assertEqual(OllamaProvider().base_url, "http://localhost:11434")

    def test_options_base_url_beats_env(self) -> None:
        os.environ["OLLAMA_HOST"] = "somewhere:1"
        self.assertEqual(OllamaProvider({"base_url": "http://x:2/"}).base_url, "http://x:2")

    def test_host_is_read_at_init_time(self) -> None:
        first = OllamaProvider()
        os.environ["OLLAMA_HOST"] = "later:9"
        self.assertEqual(first.base_url, "http://localhost:11434")
        self.assertEqual(OllamaProvider().base_url, "http://later:9")

    def test_needs_no_api_key(self) -> None:
        _, post = self.complete()
        self.assertNotIn("Authorization", post.call_args.args[1])
        self.assertEqual(self.provider.api_key(), "")

    def test_complete_payload(self) -> None:
        result, post = self.complete(system="Be brief.", temperature=0.2, max_tokens=64)
        self.assertEqual(result, "ok")
        url, _headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "llama3.2")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["options"], {"num_predict": 64, "temperature": 0.2})
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you?"},
            ],
        )

    def test_no_system_message_when_empty(self) -> None:
        _, post = self.complete(system="")
        roles = [m["role"] for m in post.call_args.args[2]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_temperature_omitted_when_none(self) -> None:
        _, post = self.complete(max_tokens=10)
        self.assertEqual(post.call_args.args[2]["options"], {"num_predict": 10})

    def test_uses_env_host_for_requests(self) -> None:
        os.environ["OLLAMA_HOST"] = "box:1234"
        provider = OllamaProvider()
        with mock.patch(POST_JSON, return_value={"message": {"content": "x"}}) as post:
            provider.complete("m", CONVERSATION)
        self.assertEqual(post.call_args.args[0], "http://box:1234/api/chat")

    def test_error_object(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            self.complete({"error": "model 'nope' not found"})
        self.assertIn("not found", str(ctx.exception))

    def test_malformed_responses(self) -> None:
        for bad in ({}, {"message": {}}, {"message": None}, [], "x", {"message": "str"}):
            with self.subTest(bad=bad), self.assertRaises(ProviderError):
                self.complete(bad)

    def test_unreachable_server_hint(self) -> None:
        err = ProviderError("Cannot reach http://localhost:11434/api/chat: [Errno 111] Connection refused")
        with mock.patch(POST_JSON, side_effect=err):
            with self.assertRaises(ProviderError) as ctx:
                self.provider.complete("m", CONVERSATION)
        self.assertIn("Is Ollama running? (ollama serve)", str(ctx.exception))

    def test_model_not_found_hint(self) -> None:
        err = ProviderError('HTTP 404 from http://localhost:11434/api/chat: {"error":"model \'x\' not found"}')
        with mock.patch(POST_JSON, side_effect=err):
            with self.assertRaises(ProviderError) as ctx:
                self.provider.complete("x", CONVERSATION)
        self.assertIn("ollama pull x", str(ctx.exception))

    def test_other_http_errors_pass_through(self) -> None:
        err = ProviderError("HTTP 500 from http://localhost:11434/api/chat: boom")
        with mock.patch(POST_JSON, side_effect=err):
            with self.assertRaises(ProviderError) as ctx:
                self.provider.complete("m", CONVERSATION)
        self.assertEqual(str(ctx.exception), str(err))


class OllamaStreamTests(EnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.provider = OllamaProvider()

    def stream(self, lines, **kwargs):
        with mock.patch(POST_LINES, return_value=iter(lines)) as post:
            chunks = list(self.provider.stream("llama3.2", CONVERSATION, **kwargs))
        return chunks, post

    @staticmethod
    def line(content: str, done: bool = False) -> str:
        return json.dumps({"message": {"role": "assistant", "content": content}, "done": done})

    def test_ndjson_stream(self) -> None:
        lines = [self.line("Hel"), self.line("lo"), self.line("", done=True)]
        chunks, post = self.stream(lines, system="sys", temperature=0.1, max_tokens=5)
        self.assertEqual(chunks, ["Hel", "lo"])
        url, _headers, payload = post.call_args.args[:3]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(payload["options"], {"num_predict": 5, "temperature": 0.1})

    def test_stops_at_done(self) -> None:
        lines = [self.line("a"), self.line("", done=True), self.line("never")]
        chunks, _ = self.stream(lines)
        self.assertEqual(chunks, ["a"])

    def test_done_line_content_is_kept(self) -> None:
        chunks, _ = self.stream([self.line("a"), self.line("b", done=True)])
        self.assertEqual(chunks, ["a", "b"])

    def test_skips_blank_lines(self) -> None:
        chunks, _ = self.stream([self.line("a"), "   ", self.line("", done=True)])
        self.assertEqual(chunks, ["a"])

    def test_error_object(self) -> None:
        lines = [self.line("a"), json.dumps({"error": "out of memory"})]
        with self.assertRaises(ProviderError) as ctx:
            self.stream(lines)
        self.assertIn("out of memory", str(ctx.exception))

    def test_invalid_json_line(self) -> None:
        with self.assertRaises(ProviderError):
            self.stream(["{nope"])

    def test_malformed_line(self) -> None:
        with self.assertRaises(ProviderError):
            self.stream([json.dumps({"unexpected": 1})])

    def test_unreachable_server_hint(self) -> None:
        def boom(*_args, **_kwargs):
            raise ProviderError("Cannot reach http://localhost:11434/api/chat: refused")
            yield  # pragma: no cover  (makes this a generator like post_lines)

        with mock.patch(POST_LINES, side_effect=boom):
            with self.assertRaises(ProviderError) as ctx:
                list(self.provider.stream("m", CONVERSATION))
        self.assertIn("ollama serve", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
