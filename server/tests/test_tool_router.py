"""Tests for the tool registry and the L0 tool-routing ladder.

Mocked only: no MCP server, no Open WebUI, no network.

Run from the server directory:

    python -m pytest tests/test_tool_router.py -q
"""
import json
import os
import unittest
from unittest.mock import patch

from api.services import llm_service, mcp_service, tool_registry


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


def _call(name: str, arguments: dict) -> dict:
    """One assistant turn that calls ``name``."""
    return {
        "model": "model-a",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


CATALOG = [
    _tool("search", "Search Spotify for tracks, albums and artists.", ["q"]),
    _tool("get_now_playing", "Show what is playing right now."),
    _tool("play", "Start or resume playback.", ["uri"]),
    _tool("pause", "Pause playback."),
    _tool("skip_next", "Skip to the next track."),
    _tool("set_volume", "Set the playback volume.", ["percent"]),
    _tool("get_queue", "Show the current play queue."),
    _tool("add_to_queue", "Add a track to the play queue.", ["uri"]),
    _tool("get_devices", "List the available playback devices."),
    _tool("library_stats", "Library statistics: albums, artists, genres."),
    _tool("library_list_albums", "List saved albums.", ["limit"]),
    _tool("library_genres", "Break the library down by genre."),
    _tool("playlist_create", "Create a new playlist.", ["name"]),
    _tool("playlist_add_tracks", "Add tracks to a playlist.", ["playlist_id", "uris"]),
    _tool("playlist_list", "List the user's playlists."),
    _tool("catalog_search", "Search the local catalog for albums.", ["q"]),
    _tool("create_playlist", "Create a new playlist for the current user.", ["name"]),
    _tool("playlist_delete", "Delete a playlist from the account.", ["playlist_id"]),
    _tool("get_saved_albums", "Get albums saved in the user's library."),
    _tool("get_saved_shows", "Get shows saved in the user's library."),
    _tool(
        "call_tool",
        "Call any registered tool by name - the dispatcher that makes a "
        "trimmed surface usable.",
    ),
    _tool("playlist_retention_set", "Set or replace a playlist expiry.",
          ["playlist_id", "expires_at"]),
]

# A slice of the real surface with the two tools a library request confuses:
# the hot catalog `search` and the local-library readers.
LIBRARY_CATALOG = [
    _tool("search", "Search Spotify's catalog for tracks, artists, albums and playlists.", ["q"]),
    _tool("play_from_search", "Search Spotify by name and immediately play the best match.", ["q"]),
    _tool("play", "Start or resume playback.", ["uri"]),
    _tool("library_list_albums",
          "Paginated, filterable list of albums in the local library.", ["search"]),
    _tool("library_list_artists",
          "Paginated, searchable list of artists in the local library.", ["search"]),
    _tool("library_stats", "Canonical library stats: album/artist/genre totals."),
    _tool("search_saved_albums",
          "Search saved albums (client-side filter over bounded walk).", ["q"]),
    _tool("search_saved_tracks",
          "Search your Liked Songs (saved tracks) by text query.", ["q"]),
    _tool("catalog_search", "Search Spotify artists and albums.", ["q"]),
    _tool("playlist_list", "List the user's playlists."),
]


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.reg = tool_registry.build(CATALOG, {"library_stats": "respotit"})

    def test_one_card_per_tool(self):
        self.assertEqual(len(self.reg), len(CATALOG))
        self.assertIn("library_stats", self.reg.by_name)

    def test_card_carries_required_arguments_and_server(self):
        card = self.reg.by_name["playlist_add_tracks"]
        self.assertEqual(card.required, ("playlist_id", "uris"))
        self.assertEqual(self.reg.by_name["library_stats"].server, "respotit")
        self.assertIn("playlist_id", card.render())

    def test_summary_is_shortened(self):
        long = "x" * 400
        reg = tool_registry.build([_tool("x_tool", long)])
        card = reg.by_name["x_tool"]
        self.assertLessEqual(len(card.summary), tool_registry.SUMMARY_CHARS)
        self.assertTrue(card.summary.endswith("\u2026"))

    def test_cards_are_far_smaller_than_the_schemas(self):
        schema_chars = len(json.dumps(CATALOG))
        card_chars = sum(len(c.render()) for c in self.reg.cards)
        self.assertLess(card_chars, schema_chars / 2)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.reg = tool_registry.build(CATALOG)

    def test_queue_request_finds_queue_tools(self):
        names = [c.name for c in tool_registry.search(self.reg, "queue a song")]
        self.assertTrue({"add_to_queue", "get_queue"} & set(names))

    def test_playlist_request_finds_playlist_tools(self):
        names = [c.name for c in tool_registry.search(self.reg, "build me a calm playlist")]
        self.assertTrue(
            {"playlist_create", "playlist_add_tracks", "playlist_list"} & set(names)
        )

    def test_genre_request_finds_the_genre_tool(self):
        names = [c.name for c in tool_registry.search(self.reg, "top genres")]
        self.assertEqual(names[0], "library_genres")

    def test_unknown_request_returns_nothing(self):
        self.assertEqual(tool_registry.search(self.reg, "zzzz qqqq"), [])

    def test_limit_is_clamped(self):
        self.assertLessEqual(len(tool_registry.search(self.reg, "library", 99)), 12)
        self.assertEqual(len(tool_registry.search(self.reg, "library", 1)), 1)

    def test_rank_keeps_the_score_search_drops(self):
        ranked = tool_registry.rank(self.reg, "how many albums in my library")
        self.assertTrue(ranked)
        scores = [score for score, _ in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 0)
        first = tool_registry.search(self.reg, "how many albums in my library")[0]
        self.assertEqual(ranked[0][1].name, first.name)

    def test_render_search_mentions_the_next_step(self):
        cards = tool_registry.search(self.reg, "queue")
        text = tool_registry.render_search("queue", cards, len(self.reg))
        self.assertIn("tool_call", text)
        self.assertIn(str(len(self.reg)), text)

    # -- ranking regressions against the real 169-tool surface ---------------

    def test_create_intent_prefers_create_over_list_and_delete(self):
        names = [c.name for c in tool_registry.search(self.reg, "build me a calm playlist")]
        self.assertIn("playlist_create", names[:3])
        self.assertNotIn("playlist_delete", names[:3])

    def test_delete_intent_surfaces_the_delete_tool(self):
        names = [c.name for c in tool_registry.search(self.reg, "delete a playlist")]
        self.assertEqual(names[0], "playlist_delete")

    def test_list_verb_finds_the_list_tool(self):
        names = [c.name for c in tool_registry.search(self.reg, "list my playlists")]
        self.assertIn("playlist_list", names[:2])

    def test_show_verb_does_not_drag_in_shows(self):
        names = [c.name for c in tool_registry.search(self.reg, "show my saved albums")]
        self.assertIn("get_saved_albums", names[:2])
        self.assertNotIn("get_saved_shows", names[:2])

    def test_description_only_match_is_demoted(self):
        names = [c.name for c in tool_registry.search(self.reg, "make a playlist")]
        self.assertNotEqual(names[0], "call_tool")
        self.assertIn("playlist_create", names[:3])

    def test_count_question_prefers_stats(self):
        names = [c.name for c in tool_registry.search(self.reg, "how many albums do i have")]
        self.assertIn("library_stats", names[:2])

    # -- the saved library is not the Spotify catalog ------------------------

    def test_library_search_prefers_library_tools_over_catalog_search(self):
        reg = tool_registry.build(LIBRARY_CATALOG)
        names = [c.name for c in tool_registry.search(reg, "search the library")]
        self.assertIn(
            names[0],
            {"library_list_albums", "library_list_artists", "search_saved_albums",
             "search_saved_tracks"},
        )
        self.assertNotIn("search", names[:2])
        self.assertNotIn("catalog_search", names[:2])

    def test_library_search_keeps_the_requested_media_type_relevant(self):
        reg = tool_registry.build(LIBRARY_CATALOG)
        for query, expected in (
            ("search my saved artists", "library_list_artists"),
            ("search my saved albums", "search_saved_albums"),
            ("search my saved tracks", "search_saved_tracks"),
        ):
            names = [c.name for c in tool_registry.search(reg, query)]
            self.assertEqual(names[0], expected, query)

    def test_generic_library_search_defaults_to_album_listing(self):
        reg = tool_registry.build(LIBRARY_CATALOG)
        names = [c.name for c in tool_registry.search(reg, "search the library")]
        self.assertEqual(names[0], "library_list_albums")

    def test_catalog_search_still_wins_without_a_library_word(self):
        reg = tool_registry.build(LIBRARY_CATALOG)
        names = [c.name for c in tool_registry.search(reg, "search spotify for miles davis")]
        self.assertEqual(names[0], "search")

    def test_library_terms_are_detected(self):
        for text in (
            "search the library",
            "search my saved albums",
            "find albums in my library",
            "what albums do I have",
        ):
            self.assertTrue(tool_registry.wants_library(text), text)
        for text in ("search spotify", "play something", "list my playlists"):
            self.assertFalse(tool_registry.wants_library(text), text)

    def test_catalog_search_schemas_warn_about_the_saved_library(self):
        reg = tool_registry.build(LIBRARY_CATALOG)
        for name in ("search", "catalog_search", "play_from_search"):
            description = reg.schemas[name]["function"]["description"]
            self.assertIn("not the user's saved library", description, name)
        # The note belongs to the advertised copy, not the shared source list.
        self.assertNotIn(
            "not the user's saved library",
            LIBRARY_CATALOG[0]["function"]["description"],
        )


class MetaToolTests(unittest.TestCase):
    def test_three_meta_tools(self):
        names = [t["function"]["name"] for t in tool_registry.META_TOOLS]
        self.assertEqual(names, ["tool_search", "tool_describe", "tool_call"])
        self.assertEqual(set(names), set(tool_registry.META_NAMES))

    def test_hot_tools_only_includes_what_exists(self):
        reg = tool_registry.build([_tool("search"), _tool("pause")])
        names = [t["function"]["name"] for t in tool_registry.hot_tools(reg)]
        self.assertEqual(names, ["search", "pause"])

    def test_hot_tools_matches_a_prefixed_name(self):
        reg = tool_registry.build([_tool("spotify_search")])
        names = [t["function"]["name"] for t in tool_registry.hot_tools(reg)]
        self.assertEqual(names, ["spotify_search"])

    def test_prefixed_catalog_search_is_demoted_and_warned_for_library_intent(self):
        reg = tool_registry.build(
            [
                _tool("spotify_search", "Search Spotify's global catalog.", ["q"]),
                _tool("library_list_albums", "List albums in the local library.", ["search"]),
            ]
        )
        self.assertEqual(
            tool_registry.search(reg, "search the library")[0].name,
            "library_list_albums",
        )
        self.assertIn(
            "not the user's saved library",
            reg.schemas["spotify_search"]["function"]["description"],
        )

    def test_describe_returns_the_full_schema(self):
        reg = tool_registry.build(CATALOG)
        schema = tool_registry.describe(reg, "play")
        self.assertEqual(schema["function"]["name"], "play")
        self.assertIsNone(tool_registry.describe(reg, "nope"))


class RouterChatTests(unittest.TestCase):
    """chat() must advertise a small surface and run meta-tools locally."""

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENWEBUI_URL": "http://webui.test",
                "OPENWEBUI_API_KEY": "sk-test",
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

    def test_auto_mode_advertises_only_hot_tools_and_meta_tools(self):
        _, calls = self._run(CATALOG, [self._answer()])
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertLessEqual(len(sent), len(tool_registry.HOT_TOOLS) + 3)
        self.assertIn("tool_search", sent)
        self.assertNotIn("playlist_create", sent)

    def test_pre_reveal_advertises_the_tool_the_words_ask_for(self):
        result, calls = self._run(
            CATALOG,
            [self._answer()],
            text="How many albums are in my library?",
        )
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertGreaterEqual(result["tool_route"]["pre_revealed"], 1)
        self.assertTrue({"library_stats", "library_list_albums"} & set(sent))
        # L1 adds schemas but keeps the whole meta surface until a real search.
        self.assertIn("tool_describe", sent)
        self.assertIn("tool_search", sent)

    def test_pre_reveal_is_silent_for_small_talk(self):
        result, calls = self._run(CATALOG, [self._answer()], text="hello there")
        self.assertEqual(result["tool_route"]["pre_revealed"], 0)
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertNotIn("library_stats", sent)
        self.assertIn("tool_describe", sent)

    def test_pre_reveal_can_be_disabled(self):
        with patch.dict(os.environ, {"LLM_TOOL_PREREVEAL": "0"}):
            result, calls = self._run(
                CATALOG,
                [self._answer()],
                text="How many albums are in my library?",
            )
        self.assertEqual(result["tool_route"]["pre_revealed"], 0)
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertNotIn("library_stats", sent)

    def test_pre_reveal_respects_its_token_budget(self):
        with patch.dict(os.environ, {"LLM_TOOL_PREREVEAL_TOKENS": "0"}):
            result, _ = self._run(
                CATALOG,
                [self._answer()],
                text="How many albums are in my library?",
            )
        self.assertEqual(result["tool_route"]["pre_revealed"], 0)

    def test_library_turn_hides_catalog_search_and_advertises_library_tools(self):
        result, calls = self._run(
            LIBRARY_CATALOG,
            [self._answer()],
            text="search the library",
        )
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertIn("library_list_albums", sent)
        self.assertNotIn("search", sent)
        self.assertTrue(result["tool_route"]["library_intent"])
        self.assertTrue(result["tool_route"]["catalog_search_hidden"])

    def test_catalog_turn_keeps_the_catalog_search_tool(self):
        result, calls = self._run(
            LIBRARY_CATALOG,
            [self._answer()],
            text="search spotify for miles davis",
        )
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertIn("search", sent)
        self.assertFalse(result["tool_route"]["library_intent"])
        self.assertFalse(result["tool_route"]["catalog_search_hidden"])

    def test_disabled_pre_reveal_keeps_catalog_search_available(self):
        """Without the library schema the turn must keep a usable search tool."""
        with patch.dict(os.environ, {"LLM_TOOL_PREREVEAL": "0"}):
            result, calls = self._run(
                LIBRARY_CATALOG,
                [self._answer()],
                text="search the library",
            )
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertIn("search", sent)
        self.assertTrue(result["tool_route"]["library_intent"])
        self.assertFalse(result["tool_route"]["catalog_search_hidden"])

    def test_library_turn_drops_sticky_catalog_search(self):
        llm_service._store_reveals("library-chat", ["search", "play"], True)
        patches = self._patches(LIBRARY_CATALOG)
        for item in patches:
            item.start()
        try:
            llm_service._store_reveals("library-chat", ["search", "play"], True)
            with patch.object(llm_service, "_request", return_value=self._answer()) as request:
                result = llm_service.chat(
                    [{"role": "user", "content": "search the library"}],
                    model="model-a",
                    chat_id="library-chat",
                )
            calls = [request.call_args.kwargs["json_body"]]
        finally:
            for item in reversed(patches):
                item.stop()
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertNotIn("search", sent)
        self.assertIn("play", sent)
        self.assertTrue(result["tool_route"]["catalog_search_hidden"])

    def test_all_mode_keeps_the_legacy_full_surface(self):
        _, calls = self._run(CATALOG, [self._answer()], tool_router="all")
        sent = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertEqual(len(sent), len(CATALOG))

    def test_off_mode_sends_no_tools(self):
        _, calls = self._run(CATALOG, [self._answer()], tool_router="off")
        self.assertNotIn("tools", calls[0])

    def test_tool_search_is_answered_locally(self):
        search_call = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": json.dumps({"query": "queue a song"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result, calls = self._run(CATALOG, [search_call, self._answer("queued")])

        tool_message = calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn("add_to_queue", tool_message["content"])
        self.assertEqual(result["content"], "queued")
        self.assertEqual(result["tool_route"]["searches"], 1)
        self.assertEqual(result["tool_route"]["mode"], "auto")

    def test_search_reveals_real_schemas_and_drops_indirection(self):
        result, calls = self._run(
            CATALOG,
            [_call("tool_search", {"query": "queue a song"}), self._answer("queued")],
        )
        sent = [t["function"]["name"] for t in calls[1]["tools"]]
        self.assertIn("add_to_queue", sent)
        self.assertIn("tool_search", sent)
        self.assertNotIn("tool_call", sent)
        self.assertNotIn("tool_describe", sent)
        self.assertGreaterEqual(result["tool_route"]["revealed"], 1)

    def test_reveal_limit_caps_injected_schemas(self):
        with patch.dict(os.environ, {"LLM_TOOL_REVEAL": "2"}):
            result, calls = self._run(
                CATALOG,
                [_call("tool_search", {"query": "playlist"}), self._answer()],
            )
        sent = [t["function"]["name"] for t in calls[1]["tools"]]
        self.assertEqual(result["tool_route"]["revealed"], 2)
        self.assertIn("tool_search", sent)
        self.assertNotIn("tool_call", sent)

    def test_tool_describe_reveals_the_schema(self):
        _, calls = self._run(
            CATALOG,
            [_call("tool_describe", {"name": "library_stats"}), self._answer()],
        )
        sent = [t["function"]["name"] for t in calls[1]["tools"]]
        self.assertIn("library_stats", sent)

    def test_zero_reveal_keeps_the_meta_surface(self):
        with patch.dict(os.environ, {"LLM_TOOL_REVEAL": "0"}):
            _, calls = self._run(
                CATALOG,
                [_call("tool_search", {"query": "build me a calm playlist"}), self._answer()],
            )
        sent = [t["function"]["name"] for t in calls[1]["tools"]]
        self.assertIn("tool_call", sent)
        self.assertNotIn("playlist_create", sent)

    def test_tool_call_indirection_reaches_the_mcp_server(self):
        call = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "tool_call",
                                    "arguments": json.dumps(
                                        {"name": "library_stats", "arguments": {"limit": 1}}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        with patch.object(mcp_service, "call_tool", return_value="42 albums") as called:
            result, calls = self._run(CATALOG, [call, self._answer("42 albums")])

        called.assert_called_once_with("library_stats", {"limit": 1})
        self.assertEqual(calls[1]["messages"][-1]["content"], "42 albums")
        self.assertEqual(result["content"], "42 albums")

    def test_tool_call_cannot_call_the_router_itself(self):
        call = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "tool_call",
                                    "arguments": json.dumps({"name": "tool_search"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        _, calls = self._run(CATALOG, [call, self._answer()])
        self.assertIn("cannot call the router", calls[1]["messages"][-1]["content"])

    def test_the_route_counts_the_registered_tools(self):
        result, _ = self._run(CATALOG, [self._answer()])
        self.assertEqual(result["tool_route"]["registered"], len(CATALOG))


class PrefixCacheTests(unittest.TestCase):
    """The tool block must stay the same across the turns of a chat.

    llama.cpp renders the tool schemas before the messages, so a block that
    changes between turns invalidates the whole prompt cache: the same prompt
    reuses 4,542 tokens, a changed tool array reuses none.
    """

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENWEBUI_URL": "http://webui.test",
                "OPENWEBUI_API_KEY": "sk-test",
            },
        )
        self.env.start()
        os.environ.pop("OPENWEBUI_MODEL", None)
        os.environ.pop("LLM_TOOL_ROUTER", None)
        os.environ.pop("LLM_TOOL_REVEAL", None)
        os.environ.pop("LLM_TOOL_PREREVEAL", None)
        os.environ.pop("LLM_TOOL_PREREVEAL_TOKENS", None)
        llm_service._reveal_cache.clear()

    def tearDown(self):
        llm_service._reveal_cache.clear()
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

    @staticmethod
    def _names(body):
        return [t["function"]["name"] for t in body.get("tools") or []]

    def test_the_second_turn_of_a_chat_sends_the_same_tool_block(self):
        text = "How many albums are in my library?"
        _, first = self._run(
            CATALOG, [self._answer()], text=text, chat_id="chat-1",
            user_message_id="u-1", assistant_message_id="a-1",
        )
        _, second = self._run(
            CATALOG, [self._answer()], text=text, chat_id="chat-1",
            user_message_id="u-2", assistant_message_id="a-2",
        )
        self.assertEqual(self._names(first[0]), self._names(second[0]))

    def test_a_revealed_tool_stays_in_the_block(self):
        call = {
            "model": "model-a",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": json.dumps({"query": "queue a song"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result, calls = self._run(
            CATALOG,
            [call, self._answer("queued")],
            chat_id="chat-2",
            user_message_id="u-1",
            assistant_message_id="a-1",
        )
        self.assertIn("add_to_queue", self._names(calls[1]))
        self.assertEqual(result["tool_route"]["searches"], 1)
        # The next turn keeps the revealed tool, so the block does not change.
        _, second = self._run(
            CATALOG,
            [self._answer()],
            text="queue another song",
            chat_id="chat-2",
            user_message_id="u-2",
            assistant_message_id="a-2",
        )
        self.assertIn("add_to_queue", self._names(second[0]))

    def test_a_new_chat_starts_with_an_empty_revealed_set(self):
        result, _ = self._run(
            CATALOG, [self._answer()], text="queue a song", chat_id="chat-fresh",
            user_message_id="u-1", assistant_message_id="a-1",
        )
        self.assertEqual(result["tool_route"]["carried"], 0)

    def test_the_carried_set_is_capped(self):
        names = [f"tool_{i}" for i in range(30)]
        catalog = CATALOG + [_tool(n, f"Tool {n}.") for n in names]
        for i, name in enumerate(names):
            self._run(
                catalog,
                [self._answer()],
                text=f"use {name}",
                chat_id="chat-3",
                user_message_id=f"u-{i}",
                assistant_message_id=f"a-{i}",
            )
        carried = llm_service._cached_reveals("chat-3")[0]
        self.assertLessEqual(len(carried), llm_service.MAX_STICKY_REVEALS)


if __name__ == "__main__":
    unittest.main()
