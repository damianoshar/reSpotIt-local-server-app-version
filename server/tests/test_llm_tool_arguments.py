"""Tool-call arguments must stay valid JSON across a turn.

A local model that hits its token limit mid-call returns an arguments string
that is not valid JSON (a ``spotify:track:`` URI cut before its closing quote).
Open WebUI parses every replayed tool call, so forwarding the raw string made
the *next* request fail with 500:

    Failed to parse tool call arguments as JSON ... missing closing quote

The loop now re-serialises parsed arguments, refuses to run a half-parsed call,
and sanitises stored output items on replay.

Mocked only: no MCP server, no Open WebUI, no network.

Run from the server directory:

    python -m pytest tests/test_llm_tool_arguments.py -q
"""
import json
import os
import unittest
from unittest.mock import patch

from api.services import llm_service, mcp_service, openwebui_service


def _tool(name: str, required=None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": list(required or []),
            },
        },
    }


CATALOG = [
    _tool("playlist_add_tracks", ["playlist_id", "uris"]),
    _tool("search", ["q"]),
    _tool("play", ["uri"]),
]

TRUNCATED = '{"uris": ["spotify:track:13s6Fcu6Jl6'


def _tool_call(arguments, name="playlist_add_tracks", call_id="call-1",
               finish_reason="length"):
    return {
        "model": "model-a",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def _answer(text="done"):
    return {
        "model": "model-a",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
    }


class ToolArgumentLoopTests(unittest.TestCase):
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

    def tearDown(self):
        self.env.stop()

    def _run(self, responses, run_tool=None):
        calls = []

        def fake_request(method, path, *, json_body=None, timeout=None, token=None):
            calls.append(json_body)
            return responses.pop(0)

        patches = [
            patch.object(mcp_service, "is_configured", return_value=True),
            patch.object(mcp_service, "list_tools", return_value=CATALOG),
            patch.object(mcp_service, "tool_servers", return_value={}),
            patch.object(llm_service, "_request", side_effect=fake_request),
        ]
        if run_tool is not None:
            patches.append(patch.object(llm_service, "_run_tool", side_effect=run_tool))
        for p in patches:
            p.start()
        try:
            result = llm_service.chat(
                [{"role": "user", "content": "add these tracks"}],
                model="model-a",
            )
        finally:
            for p in patches:
                p.stop()
        return result, calls

    def test_a_truncated_call_is_not_run_and_not_replayed_raw(self):
        executed = []

        def run_tool(name, args, *rest):
            executed.append((name, args))
            return "ran"

        result, calls = self._run(
            [_tool_call(TRUNCATED), _answer("retry with fewer tracks")],
            run_tool=run_tool,
        )

        self.assertEqual(executed, [])

        # The follow-up request must carry parseable JSON, or Open WebUI 500s.
        assistant_rows = [
            row for row in calls[1]["messages"] if row.get("tool_calls")
        ]
        self.assertEqual(len(assistant_rows), 1)
        forwarded = assistant_rows[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(forwarded), {})

        self.assertEqual(result["tool_calls"][0]["arguments"], {})
        self.assertEqual(result["tool_calls"][0]["error"], "invalid_arguments")
        tool_row = calls[1]["messages"][-1]
        self.assertEqual(tool_row["role"], "tool")
        self.assertIn("incomplete JSON arguments", tool_row["content"])

    def test_a_valid_call_still_runs_with_its_arguments(self):
        seen = []

        def run_tool(name, args, *rest):
            seen.append((name, args))
            return "2 tracks added"

        result, calls = self._run(
            [
                _tool_call('{"playlist_id": "p1", "uris": ["spotify:track:a"]}',
                           finish_reason="tool_calls"),
                _answer("added"),
            ],
            run_tool=run_tool,
        )

        self.assertEqual(
            seen,
            [("playlist_add_tracks",
              {"playlist_id": "p1", "uris": ["spotify:track:a"]})],
        )
        assistant_rows = [
            row for row in calls[1]["messages"] if row.get("tool_calls")
        ]
        forwarded = assistant_rows[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(
            json.loads(forwarded),
            {"playlist_id": "p1", "uris": ["spotify:track:a"]},
        )
        self.assertNotIn("error", result["tool_calls"][0])


class StoredArgumentsTests(unittest.TestCase):
    def test_replayed_output_items_never_carry_invalid_json(self):
        rows = openwebui_service._output_rows(
            [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "status": "completed",
                    "name": "playlist_add_tracks",
                    "arguments": TRUNCATED,
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "input_text", "text": "nothing ran"}],
                },
            ]
        )

        self.assertEqual(len(rows), 2)
        arguments = rows[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(arguments), {})

    def test_new_output_items_sanitise_a_truncated_string(self):
        items = openwebui_service._tool_output_items(
            [{"id": "c1", "name": "playlist_add_tracks", "arguments": TRUNCATED,
              "result": "nothing ran"}]
        )

        self.assertEqual(items[0]["arguments"], "{}")
        self.assertEqual(json.loads(items[0]["arguments"]), {})

    def test_normalise_arguments_marks_truncation(self):
        self.assertEqual(llm_service._normalise_tool_arguments('{"a": 1}'), ({"a": 1}, True))
        self.assertEqual(llm_service._normalise_tool_arguments({"a": 1}), ({"a": 1}, True))
        self.assertEqual(llm_service._normalise_tool_arguments(None), ({}, True))
        self.assertEqual(llm_service._normalise_tool_arguments(TRUNCATED), ({}, False))
        self.assertEqual(llm_service._normalise_tool_arguments('["a"]'), ({}, False))


if __name__ == "__main__":
    unittest.main()
