"""Tests for the chat turn fields and the context-budget guard.

Mocked only: no MCP server, no Open WebUI, no network.

Run from the server directory:

    python -m pytest tests/test_llm_context.py -q
"""
import json
import os
import unittest
from unittest.mock import patch

from api.services import llm_service, mcp_service


def _tool(name: str, description: str = "", required=None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": list(required or []),
            },
        },
    }


CATALOG = [
    _tool("search", "Search Spotify for tracks, albums and artists.", ["q"]),
    _tool("get_now_playing", "Show what is playing right now."),
    _tool("play", "Start or resume playback.", ["uri"]),
    _tool("library_stats", "Library statistics: albums, artists, genres."),
]


class ContextGuardTests(unittest.TestCase):
    """The turn must fit the local model's window, and say so if it does not."""

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENWEBUI_URL": "http://webui.test",
                "OPENWEBUI_API_KEY": "sk-test",
                "LLM_CONTEXT_TOKENS": "16384",
            },
        )
        self.env.start()
        os.environ.pop("OPENWEBUI_MODEL", None)
        os.environ.pop("LLM_TOOL_ROUTER", None)
        os.environ.pop("LLM_TOOL_REVEAL", None)
        os.environ.pop("LLM_TOOL_PREREVEAL", None)
        os.environ.pop("LLM_TOOL_PREREVEAL_TOKENS", None)

    def tearDown(self):
        self.env.stop()

    def _patches(self, tools):
        return [
            patch.object(mcp_service, "is_configured", return_value=True),
            patch.object(mcp_service, "list_tools", return_value=tools),
            patch.object(mcp_service, "tool_servers", return_value={}),
        ]

    def _run(self, tools, responses, text="which artists are in my library?", **kwargs):
        calls = []

        def fake_request(method, path, *, json_body=None, timeout=None, token=None):
            calls.append(json_body)
            return responses.pop(0)

        patches = self._patches(tools)
        for p in patches:
            p.start()
        try:
            with patch.object(llm_service, "_request", side_effect=fake_request):
                result = llm_service.chat(
                    [{"role": "user", "content": text}],
                    model="model-a",
                    **kwargs,
                )
        finally:
            for p in patches:
                p.stop()
        return result, calls

    @staticmethod
    def _answer(text="done"):
        return {
            "model": "model-a",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }

    def test_chat_id_is_sent_top_level_for_compaction(self):
        _, calls = self._run(
            CATALOG,
            [self._answer()],
            chat_id="chat-1",
            user_message_id="u-1",
            assistant_message_id="a-1",
            parent_id="p-1",
        )
        body = calls[0]
        self.assertEqual(body.get("chat_id"), "chat-1")
        user_message = body.get("user_message") or {}
        self.assertEqual(user_message.get("id"), "u-1")
        self.assertEqual(user_message.get("parentId"), "p-1")
        self.assertEqual(user_message.get("models"), ["model-a"])
        self.assertEqual(
            body.get("message_ids"),
            [{"model_id": "model-a", "message_id": "a-1"}],
        )
        self.assertNotIn("metadata", body)

    def test_chat_id_without_persisted_turn_has_no_user_message(self):
        _, calls = self._run(CATALOG, [self._answer()], chat_id="chat-1")
        body = calls[0]
        self.assertEqual(body.get("chat_id"), "chat-1")
        self.assertNotIn("user_message", body)
        self.assertNotIn("message_ids", body)

    def test_a_tool_follow_up_does_not_repeat_the_chat_turn(self):
        """A second request must not make Open WebUI insert the turn again.

        Open WebUI answers a repeated turn with "409 Message already exists
        or has an invalid ID", because the assistant reply of the first
        request is already stored.
        """
        call = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": '{"query": "library"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        _, calls = self._run(
            CATALOG,
            [call, self._answer("done")],
            chat_id="chat-1",
            user_message_id="u-1",
            assistant_message_id="a-1",
            parent_id="p-1",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].get("chat_id"), "chat-1")
        self.assertEqual(
            calls[0].get("message_ids"),
            [{"model_id": "model-a", "message_id": "a-1"}],
        )
        # The follow-up carries the tool result in its own messages instead.
        self.assertNotIn("chat_id", calls[1])
        self.assertNotIn("user_message", calls[1])
        self.assertNotIn("message_ids", calls[1])
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(calls[1]["messages"][-1]["tool_call_id"], "call-1")

    def test_no_chat_id_sends_no_chat_fields(self):
        _, calls = self._run(CATALOG, [self._answer()])
        self.assertNotIn("chat_id", calls[0])
        self.assertNotIn("user_message", calls[0])
        self.assertNotIn("metadata", calls[0])

    def test_a_huge_prompt_is_refused_before_it_reaches_the_model(self):
        with patch.dict(os.environ, {"LLM_CONTEXT_TOKENS": "256"}):
            with self.assertRaises(llm_service.LLMContextExceeded) as ctx:
                self._run(
                    CATALOG,
                    [self._answer()],
                    text="library " * 1000,
                    tool_router="off",
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(ctx.exception.code, "llm_context_exceeded")
        self.assertIn("messages", str(ctx.exception))

    def test_the_estimate_covers_tools_and_messages(self):
        result, _ = self._run(CATALOG, [self._answer()])
        self.assertGreater(result["tool_route"]["estimated_tokens"], 0)
        self.assertEqual(result["tool_route"]["registered"], len(CATALOG))

    def test_the_stored_chain_is_measured_even_when_the_window_is_small(self):
        """Open WebUI replays the whole chain, so the guard must see it."""
        with patch.dict(os.environ, {"LLM_CONTEXT_TOKENS": "256"}):
            with self.assertRaises(llm_service.LLMContextExceeded):
                self._run(
                    CATALOG,
                    [self._answer()],
                    text="short question",
                    tool_router="off",
                    estimate_messages=[{"role": "user", "content": "library " * 1000}],
                )


class TextToolCallTests(unittest.TestCase):
    """A model that writes tool-call markup as text must still run the tool."""

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENWEBUI_URL": "http://webui.test",
                "OPENWEBUI_API_KEY": "sk-test",
                "LLM_CONTEXT_TOKENS": "16384",
            },
        )
        self.env.start()
        for name in (
            "OPENWEBUI_MODEL",
            "LLM_TOOL_ROUTER",
            "LLM_TOOL_REVEAL",
            "LLM_TOOL_PREREVEAL",
            "LLM_TOOL_PREREVEAL_TOKENS",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()

    MARKUP = (
        "All tracks located. Now let me build the playlist and start playing it.\n\n"
        "<tool_call>\n<function=play>\n<parameter=uris>\n"
        '["spotify:track:7xRemq7GLu0Tbqe9OckG87", "spotify:track:5IyL3XOaRPpTgxVjRIAxXU"]\n'
        "</parameter>\n</function>\n</tool_call>"
    )

    def _run(self, responses):
        calls = []
        results = []

        def fake_request(method, path, *, json_body=None, timeout=None, token=None):
            calls.append(json_body)
            return responses.pop(0)

        def fake_tool(name, arguments):
            results.append((name, arguments))
            return "Playback started."

        patches = [
            patch.object(mcp_service, "is_configured", return_value=True),
            patch.object(mcp_service, "list_tools", return_value=CATALOG),
            patch.object(mcp_service, "tool_servers", return_value={}),
            patch.object(mcp_service, "call_tool", side_effect=fake_tool),
        ]
        for item in patches:
            item.start()
        try:
            with patch.object(llm_service, "_request", side_effect=fake_request):
                answer = llm_service.chat(
                    [{"role": "user", "content": "make the playlist"}],
                    model="model-a",
                )
        finally:
            for item in patches:
                item.stop()
        return answer, calls, results

    @staticmethod
    def _markup(text):
        return {
            "model": "model-a",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }

    @staticmethod
    def _answer(text="done"):
        return {
            "model": "model-a",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }

    def test_markup_tool_call_is_executed_and_the_loop_continues(self):
        answer, calls, results = self._run(
            [self._markup(self.MARKUP), self._answer("The playlist is playing.")]
        )

        # The tool actually ran, with the list parsed out of the markup.
        self.assertEqual(results[0][0], "play")
        self.assertEqual(
            results[0][1]["uris"],
            ["spotify:track:7xRemq7GLu0Tbqe9OckG87", "spotify:track:5IyL3XOaRPpTgxVjRIAxXU"],
        )
        # A second request carried the result back to the model.
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(calls[1]["messages"][-1]["content"], "Playback started.")
        # The answer is the model's prose, without the markup.
        self.assertEqual(answer["content"], "The playlist is playing.")
        self.assertEqual(answer["tool_route"]["text_calls"], 1)

    def test_markup_is_removed_from_a_final_answer(self):
        answer, calls, _ = self._run(
            [self._markup(self.MARKUP), self._answer("The playlist is playing.")]
        )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("<tool_call>", answer["content"])
        self.assertNotIn("<function=", answer["content"])

    def test_a_plain_parameter_stays_a_string(self):
        markup = (
            "<tool_call>\n<function=play>\n<parameter=context_uri>\n"
            "spotify:playlist:1DhX9l474TRN0aHjdqCBxv\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, _, results = self._run([self._markup(markup), self._answer("ok")])
        self.assertEqual(
            results[0][1]["context_uri"], "spotify:playlist:1DhX9l474TRN0aHjdqCBxv"
        )

    def test_two_calls_in_one_reply_both_run(self):
        markup = (
            "<tool_call>\n<function=search>\n<parameter=q>\nwinter\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=play>\n<parameter=context_uri>\n"
            "spotify:playlist:abc\n</parameter>\n</function>\n</tool_call>"
        )
        _, calls, results = self._run([self._markup(markup), self._answer("ok")])
        self.assertEqual([name for name, _ in results], ["search", "play"])
        tool_rows = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_rows), 2)
        # The ids have to match the assistant's calls, or the upstream rejects
        # the exchange.
        asked = [c["id"] for c in calls[1]["messages"][-3]["tool_calls"]]
        self.assertEqual([row["tool_call_id"] for row in tool_rows], asked)

    def test_a_native_call_wins_over_markup(self):
        native = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "<tool_call><function=search><parameter=q>x"
                                   "</parameter></function></tool_call>",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "play",
                                             "arguments": '{"context_uri": "spotify:playlist:z"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        _, calls, results = self._run([native, self._answer("ok")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "play")


class TextToolCallParsingTests(unittest.TestCase):
    """The markup parser on its own."""

    def test_parses_the_template_shape(self):
        calls = llm_service._parse_text_tool_calls(
            "<tool_call>\n<function=playlist_create>\n<parameter=name>\n"
            "Gloomy Winter\n</parameter>\n<parameter=public>\nfalse\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "playlist_create")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"name": "Gloomy Winter", "public": False},
        )

    def test_ignores_ordinary_prose(self):
        self.assertEqual(llm_service._parse_text_tool_calls("The queue is empty."), [])

    def test_strip_leaves_the_prose(self):
        text = "Starting.\n<tool_call>\n<function=play>\n<parameter=uris>\n[]\n"
        text += "</parameter>\n</function>\n</tool_call>"
        self.assertEqual(llm_service._strip_text_tool_calls(text), "Starting.")


class ToolLoopBudgetTests(unittest.TestCase):
    """The loop is bounded, does not repeat a call, and ends with a report."""

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENWEBUI_URL": "http://webui.test",
                "OPENWEBUI_API_KEY": "sk-test",
                "LLM_CONTEXT_TOKENS": "32768",
            },
        )
        self.env.start()
        for name in (
            "OPENWEBUI_MODEL",
            "LLM_TOOL_ROUTER",
            "LLM_TOOL_REVEAL",
            "LLM_TOOL_PREREVEAL",
            "LLM_TOOL_PREREVEAL_TOKENS",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()

    def _run(self, responses):
        calls = []
        results = []

        def fake_request(method, path, *, json_body=None, timeout=None, token=None):
            calls.append(json_body)
            return responses.pop(0)

        def fake_tool(name, arguments):
            results.append((name, arguments))
            return f"{name} ok"

        patches = [
            patch.object(mcp_service, "is_configured", return_value=True),
            patch.object(mcp_service, "list_tools", return_value=CATALOG),
            patch.object(mcp_service, "tool_servers", return_value={}),
            patch.object(mcp_service, "call_tool", side_effect=fake_tool),
        ]
        for item in patches:
            item.start()
        try:
            with patch.object(llm_service, "_request", side_effect=fake_request):
                answer = llm_service.chat(
                    [{"role": "user", "content": "do the whole thing"}],
                    model="model-a",
                )
        finally:
            for item in patches:
                item.stop()
        return answer, calls, results

    @staticmethod
    def _markup(name, parameter, value):
        return {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": (
                            f"<tool_call>\n<function={name}>\n"
                            f"<parameter={parameter}>\n{value}\n</parameter>\n"
                            "</function>\n</tool_call>"
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    @staticmethod
    def _answer(text="done"):
        return {
            "model": "model-a",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        }

    def test_an_identical_repeated_call_is_not_run_twice(self):
        answer, calls, results = self._run(
            [
                self._markup("get_now_playing", "response_format", "concise"),
                self._markup("get_now_playing", "response_format", "concise"),
                self._answer("Checked."),
            ]
        )
        self.assertEqual(len(results), 1)
        second = [m for m in calls[2]["messages"] if m.get("role") == "tool"][-1]
        self.assertIn("Already called", second["content"])
        self.assertEqual(answer["content"], "Checked.")

    def test_exhausted_rounds_end_with_a_report_instruction(self):
        rounds = 12
        responses = [
            self._markup("get_now_playing", "response_format", f"r{i}")
            for i in range(rounds)
        ]
        responses.append(self._answer("Three steps ran; one is unfinished."))
        answer, calls, results = self._run(responses)

        self.assertEqual(len(results), rounds)
        # rounds + the final no-tools request
        self.assertEqual(len(calls), rounds + 1)
        last_messages = calls[-1]["messages"]
        self.assertNotIn("tools", calls[-1])
        self.assertEqual(last_messages[-1]["role"], "user")
        self.assertIn("Stop calling tools", last_messages[-1]["content"])
        self.assertEqual(answer["content"], "Three steps ran; one is unfinished.")

    def test_the_default_round_budget_covers_a_playlist_pipeline(self):
        """search, create, add, play plus reading the results must all fit."""
        import inspect

        signature = inspect.signature(llm_service.chat)
        self.assertGreaterEqual(signature.parameters["tool_rounds"].default, 10)


if __name__ == "__main__":
    unittest.main()
