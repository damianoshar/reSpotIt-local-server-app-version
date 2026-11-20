"""Tests for Phase 5/6 chat history: current-chat pointer + Open WebUI chats.

Mocked only: no Open WebUI, llama-server or Keycloak is contacted. The SQL
tests use the throwaway SQLite database configured in ``conftest.py``.

Run from the server directory:

    python -m pytest tests/test_llm_chats.py -q
"""

import os
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_auth.db")
os.environ.setdefault("JWT_SECRET", "test-secret")

from api.database import engine, get_db
from api.main import app
from api.models import Base, LLMChatSession, User
from api.services import (
    chat_session_service,
    llm_service,
    openwebui_service,
    user_auth_service,
)


def _response(status: int = 200, payload=None, text: str = "") -> Mock:
    resp = Mock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = text
    resp.content = b"x" if payload is not None else b""
    return resp


@pytest.fixture()
def Session():
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.query(LLMChatSession).delete()
        db.query(User).delete()
        db.commit()
    return factory


@pytest.fixture()
def user(Session):
    with Session() as db:
        row = User(
            email="chats@test.com",
            password_hash=user_auth_service.hash_password("secret123"),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return str(row.id), row.email


@pytest.fixture()
def client(Session, monkeypatch):
    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(openwebui_service, "get_user_token", lambda u: "owui-token")
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def _auth(user):
    user_id, email = user
    token = user_auth_service.create_access_token(user_id, email)
    return {"Authorization": f"Bearer {token}"}


# ── current-chat pointer ───────────────────────────────────────────────────


class TestChatSessionService:
    def test_set_then_get(self, Session, user):
        user_id, _ = user
        with Session() as db:
            assert chat_session_service.get_current_chat_id(db, user_id) is None
            chat_session_service.set_current_chat_id(db, user_id, "chat-1")
            assert chat_session_service.get_current_chat_id(db, user_id) == "chat-1"

    def test_set_reuses_one_row(self, Session, user):
        user_id, _ = user
        with Session() as db:
            chat_session_service.set_current_chat_id(db, user_id, "chat-1")
            chat_session_service.set_current_chat_id(db, user_id, "chat-2")
            assert db.query(LLMChatSession).count() == 1
            assert chat_session_service.get_current_chat_id(db, user_id) == "chat-2"

    def test_clear_matching_only(self, Session, user):
        user_id, _ = user
        with Session() as db:
            chat_session_service.set_current_chat_id(db, user_id, "chat-1")
            chat_session_service.clear_current_chat_id(db, user_id, "other")
            assert chat_session_service.get_current_chat_id(db, user_id) == "chat-1"
            chat_session_service.clear_current_chat_id(db, user_id, "chat-1")
            assert chat_session_service.get_current_chat_id(db, user_id) is None

    def test_clear_unknown_user_is_noop(self, Session):
        with Session() as db:
            chat_session_service.clear_current_chat_id(db, "missing", "chat-1")
            assert chat_session_service.get_current_chat_id(db, "missing") is None


# ── Open WebUI chat helpers ────────────────────────────────────────────────


class TestChatHelpers:
    def test_requires_token(self):
        with pytest.raises(openwebui_service.OpenWebUIError):
            openwebui_service.list_chats(None)

    def test_list_chats_uses_bearer_and_limit(self, monkeypatch):
        calls = []

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return _response(200, [{"id": "a"}, {"id": "b"}, {"id": "c"}])

        monkeypatch.setattr(openwebui_service.requests, "request", fake_request)
        rows = openwebui_service.list_chats("tok", limit=2)
        assert [r["id"] for r in rows] == ["a", "b"]
        method, url, kwargs = calls[0]
        assert method == "GET"
        assert url.endswith("/api/v1/chats/")
        assert kwargs["headers"]["Authorization"] == "Bearer tok"

    def test_get_chat_404_maps_to_not_found(self, monkeypatch):
        monkeypatch.setattr(
            openwebui_service.requests, "request",
            lambda *a, **k: _response(404, {"detail": "Not found"}),
        )
        with pytest.raises(openwebui_service.OpenWebUIChatNotFound):
            openwebui_service.get_chat("tok", "missing")

    def test_get_chat_401_maps_to_credential_rejected(self, monkeypatch):
        monkeypatch.setattr(
            openwebui_service.requests, "request",
            lambda *a, **k: _response(401, {"detail": "bad token"}),
        )
        with pytest.raises(openwebui_service.OpenWebUICredentialRejected):
            openwebui_service.get_chat("tok", "chat-1")

    def test_create_chat_posts_chat_shape(self, monkeypatch):
        seen = {}

        def fake_request(method, url, **kwargs):
            seen.update({"method": method, "url": url, "json": kwargs["json"]})
            return _response(200, {"id": "new-chat", "chat": kwargs["json"]["chat"]})

        monkeypatch.setattr(openwebui_service.requests, "request", fake_request)
        row = openwebui_service.create_chat("tok", title="Hi", model="m")
        assert row["id"] == "new-chat"
        assert seen["method"] == "POST"
        assert seen["url"].endswith("/api/v1/chats/new")
        chat = seen["json"]["chat"]
        assert chat["title"] == "Hi"
        assert chat["models"] == ["m"]
        assert chat["history"] == {"messages": {}, "currentId": None}

    def test_history_messages_keeps_last_turns(self):
        row = {
            "chat": {
                "messages": [
                    {"role": "system", "content": "ignore"},
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "   "},
                    {"role": "user", "content": "three"},
                ]
            }
        }
        assert openwebui_service.history_messages(row, 2) == [
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        assert openwebui_service.history_messages(row, 0) == []

    def test_history_messages_keeps_message_ids(self):
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "one"},
                    {"id": "m2", "role": "assistant", "content": "two"},
                    {"role": "user", "content": "no id"},
                ]
            }
        }
        assert openwebui_service.history_messages(row, 10) == [
            {"role": "user", "content": "one", "id": "m1"},
            {"role": "assistant", "content": "two", "id": "m2"},
            {"role": "user", "content": "no id"},
        ]

    def test_history_messages_carries_the_compaction_checkpoint(self):
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "one"},
                    {"id": "m2", "role": "assistant", "content": "two"},
                ],
                "history": {
                    "messages": {
                        "m1": {"id": "m1", "role": "user", "contextSummary": "old talk"},
                        "m2": {"id": "m2", "role": "assistant"},
                    }
                },
            }
        }
        rows = openwebui_service.history_messages(row, 10)
        assert rows[0]["contextSummary"] == "old talk"
        assert "contextSummary" not in rows[1]

    def test_history_messages_carries_a_checkpoint_outside_the_window(self):
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "one"},
                    {"id": "m2", "role": "assistant", "content": "two"},
                    {"id": "m3", "role": "user", "content": "three"},
                ],
                "history": {
                    "messages": {
                        "m1": {"id": "m1", "role": "user", "contextSummary": "ancient talk"},
                        "m2": {"id": "m2", "role": "assistant"},
                        "m3": {"id": "m3", "role": "user"},
                    }
                },
            }
        }
        rows = openwebui_service.history_messages(row, 2)
        assert rows[0]["contextSummary"] == "ancient talk"
        assert rows[1].get("contentSummary") is None

    def test_compacted_chain_drops_what_the_checkpoint_replaced(self):
        """The API has to see the chain Open WebUI would replay."""
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "old question"},
                    {"id": "m2", "role": "assistant", "content": "old answer"},
                    {"id": "m3", "role": "user", "content": "new question"},
                ],
                "history": {
                    "messages": {
                        "m1": {"id": "m1", "role": "user"},
                        "m2": {"id": "m2", "role": "assistant", "contextSummary": "summary text"},
                        "m3": {"id": "m3", "role": "user"},
                    }
                },
            }
        }
        rows = openwebui_service.compacted_chain(row)
        assert rows[0]["role"] == "system"
        assert "summary text" in rows[0]["content"]
        # Everything before the checkpoint is gone, the checkpoint stays.
        assert [r["content"] for r in rows[1:]] == ["old answer", "new question"]

    def test_compacted_chain_rebuilds_the_tool_exchange(self):
        """The stored output items have to come back as tool calls and results.

        Without them the model sees its own call requests and no answers, so a
        multi-step request looks unfinished and it asks the user to prompt the
        next step.
        """
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "make the playlist"},
                    {
                        "id": "m2",
                        "role": "assistant",
                        "content": "All tracks located. Now let me build it.",
                        "output": [
                            {"type": "function_call", "call_id": "call-1",
                             "status": "completed", "name": "playlist_create",
                             "arguments": '{"name": "Gloomy Winter"}'},
                            {"type": "function_call_output", "call_id": "call-1",
                             "output": [{"type": "input_text",
                                         "text": "Created playlist 1DhX9l"}]},
                        ],
                    },
                    {"id": "m3", "role": "assistant", "content": "Built and playing."},
                ],
                "history": {"messages": {}},
            }
        }
        rows = openwebui_service.compacted_chain(row)
        assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant"]
        call = rows[1]["tool_calls"][0]
        assert call["id"] == "call-1"
        assert call["function"]["name"] == "playlist_create"
        assert call["function"]["arguments"] == '{"name": "Gloomy Winter"}'
        assert rows[2]["tool_call_id"] == "call-1"
        assert rows[2]["content"] == "Created playlist 1DhX9l"
        assert rows[3]["content"] == "Built and playing."

    def test_compacted_chain_drops_an_unanswered_tool_call(self):
        """An output without a result is not replayed, the way Open WebUI does."""
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "play it"},
                    {
                        "id": "m2",
                        "role": "assistant",
                        "content": "",
                        "output": [
                            {"type": "function_call", "call_id": "call-1",
                             "status": "completed", "name": "play",
                             "arguments": '{"uris": ["spotify:track:1"]}'},
                            {"type": "function_call", "call_id": "call-2",
                             "status": "in_progress", "name": "play",
                             "arguments": "{}"},
                        ],
                    },
                ],
                "history": {"messages": {}},
            }
        }
        rows = openwebui_service.compacted_chain(row)
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert [c["id"] for c in rows[1]["tool_calls"]] == ["call-1"]

    def test_compacted_chain_anchors_the_checkpoint_on_a_tool_turn(self):
        """A summary on a message that expanded into tool rows still cuts there."""
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "ancient"},
                    {
                        "id": "m2",
                        "role": "assistant",
                        "content": "",
                        "output": [
                            {"type": "function_call", "call_id": "call-1",
                             "status": "completed", "name": "search",
                             "arguments": '{"q": "winter"}'},
                            {"type": "function_call_output", "call_id": "call-1",
                             "output": [{"type": "input_text", "text": "tracks"}]},
                        ],
                    },
                    {"id": "m3", "role": "user", "content": "next"},
                ],
                "history": {
                    "messages": {
                        "m2": {"id": "m2", "role": "assistant",
                               "contextSummary": "compacted talk"},
                    }
                },
            }
        }
        rows = openwebui_service.compacted_chain(row)
        assert rows[0]["role"] == "system"
        assert "compacted talk" in rows[0]["content"]
        assert [r["role"] for r in rows[1:]] == ["assistant", "tool", "user"]

    def test_compacted_chain_without_a_checkpoint_is_the_whole_chain(self):
        row = {
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "one"},
                    {"id": "m2", "role": "assistant", "content": "two"},
                ],
                "history": {"messages": {}},
            }
        }
        rows = openwebui_service.compacted_chain(row)
        assert [r["content"] for r in rows] == ["one", "two"]

    def test_append_exchange_keeps_flat_and_tree_in_sync(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "Existing",
            "chat": {
                "title": "Existing",
                "messages": [
                    {"id": "m1", "parentId": None, "childrenIds": [],
                     "role": "user", "content": "old"},
                ],
                "history": {
                    "messages": {
                        "m1": {"id": "m1", "parentId": None,
                               "childrenIds": [], "role": "user",
                               "content": "old"},
                    },
                    "currentId": "m1",
                },
            },
        }
        captured = {}

        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_exchange(
            "tok", "chat-1", "new question", "new answer", model="m"
        )

        chat = captured["chat"]
        assert len(chat["messages"]) == 3
        user_msg, assistant_msg = chat["messages"][1:]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "new question"
        assert user_msg["parentId"] == "m1"
        assert assistant_msg["parentId"] == user_msg["id"]
        assert assistant_msg["done"] is True

        tree = chat["history"]["messages"]
        assert tree["m1"]["childrenIds"] == [user_msg["id"]]
        assert tree[user_msg["id"]]["childrenIds"] == [assistant_msg["id"]]
        assert chat["history"]["currentId"] == assistant_msg["id"]
        # A chat that already has a title keeps it.
        assert chat["title"] == "Existing"

    def test_append_exchange_titles_new_chat(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "New Chat",
            "chat": {"title": "New Chat", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_exchange("tok", "chat-1", "q", "a", title="My title")
        assert captured["chat"]["title"] == "My title"

    def test_append_exchange_honours_supplied_user_message_id(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {"title": "Chat", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_exchange(
            "tok", "chat-1", "q", "a", user_message_id="turn-1"
        )
        user_msg = captured["chat"]["messages"][0]
        assert user_msg["id"] == "turn-1"
        assert captured["chat"]["history"]["messages"]["turn-1"] is user_msg

    def test_append_exchange_keeps_a_checkpoint_written_while_generating(self, monkeypatch):
        """Open WebUI can checkpoint the turn before we persist it."""
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {
                "title": "Chat",
                "messages": [],
                "history": {
                    "messages": {
                        "turn-1": {"id": "turn-1", "role": "user",
                                   "contextSummary": "compacted earlier talk"},
                    },
                    "currentId": None,
                },
            },
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_exchange(
            "tok", "chat-1", "q", "a", user_message_id="turn-1"
        )
        user_msg = captured["chat"]["messages"][0]
        assert user_msg["contextSummary"] == "compacted earlier talk"

    def test_complete_turn_stores_the_tool_exchange(self, monkeypatch):
        """The stored chain keeps the tool call and its result."""
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {
                "title": "Chat",
                "messages": [],
                "history": {"messages": {}, "currentId": None},
            },
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.complete_turn(
            "tok", "chat-1", "u1", "a1", "question", "answer",
            model="m", parent_id=None,
            tool_calls=[
                {"id": "call-1", "name": "tool_search", "arguments": {"query": "x"},
                 "result": "8 tools"}
            ],
        )
        output = captured["chat"]["history"]["messages"]["a1"]["output"]
        assert output[0]["type"] == "function_call"
        assert output[0]["call_id"] == "call-1"
        assert output[0]["name"] == "tool_search"
        assert output[0]["status"] == "completed"
        assert output[1]["type"] == "function_call_output"
        assert output[1]["call_id"] == "call-1"
        assert output[1]["output"][0]["text"] == "8 tools"

    def test_complete_turn_stores_the_full_tool_answer(self, monkeypatch):
        """The chain keeps the whole answer; the response only keeps a slice."""
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {"title": "Chat", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.complete_turn(
            "tok", "chat-1", "u1", "a1", "question", "answer",
            model="m", parent_id=None,
            tool_calls=[
                {"id": "call-1", "name": "playlist_create", "arguments": {"name": "x"},
                 "result": "short", "result_full": "playlist_id 1DhX9l474TRN0aHjdqCBxv"},
            ],
        )
        output = captured["chat"]["history"]["messages"]["a1"]["output"]
        assert output[1]["output"][0]["text"] == "playlist_id 1DhX9l474TRN0aHjdqCBxv"

    def test_append_exchange_keeps_the_tool_exchange(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {"title": "Chat", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_exchange(
            "tok", "chat-1", "q", "a",
            tool_calls=[{"id": "call-1", "name": "tool_search",
                         "arguments": {"query": "x"}, "result": "8 tools"}],
        )
        output = captured["chat"]["messages"][1]["output"]
        assert output[0]["type"] == "function_call"
        assert output[1]["type"] == "function_call_output"

    def test_current_message_id_reads_the_history_pointer(self):
        row = {
            "chat": {
                "messages": [{"id": "m1"}, {"id": "m2"}],
                "history": {"messages": {}, "currentId": "m2"},
            }
        }
        assert openwebui_service.current_message_id(row) == "m2"

    def test_current_message_id_falls_back_to_the_last_message(self):
        row = {"chat": {"messages": [{"id": "m1"}, {"id": "m2"}]}}
        assert openwebui_service.current_message_id(row) == "m2"

    def test_complete_turn_fills_the_assistant_placeholder(self, monkeypatch):
        """Open WebUI inserts the turn; the API only adds the answer."""
        stored = {
            "id": "chat-1",
            "title": "Chat",
            "chat": {
                "title": "Chat",
                "messages": [{"id": "m1", "role": "user", "content": "older"}],
                "history": {
                    "messages": {
                        "m1": {"id": "m1", "role": "user", "childrenIds": ["m2"]},
                        "m2": {"id": "m2", "role": "assistant", "parentId": "m1",
                               "content": "", "done": False},
                    },
                    "currentId": "m2",
                },
            },
        }
        captured = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.complete_turn(
            "tok", "chat-1", "u1", "a1", "question", "answer",
            model="m", parent_id="m2",
        )
        chat = captured["chat"]
        tree = chat["history"]["messages"]
        assert tree["a1"]["content"] == "answer"
        assert tree["a1"]["done"] is True
        assert tree["u1"]["content"] == "question"
        assert tree["u1"]["parentId"] == "m2"
        assert chat["history"]["currentId"] == "a1"
        # The turn is mirrored into the flat list the UI renders.
        assert [m["id"] for m in chat["messages"]] == ["m1", "u1", "a1"]

    def test_append_exchange_names_one_model_on_the_user_message(self, monkeypatch):
        """Open WebUI hides an answer when the user message has no model."""
        stored = {
            "id": "chat-1",
            "title": "Scan",
            "chat": {"title": "Scan", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        captured = {}

        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )
        monkeypatch.setattr(
            openwebui_service, "resolve_model_id", lambda token=None: "model-x",
        )

        openwebui_service.append_exchange("tok", "chat-1", "question", "answer")

        chat = captured["chat"]
        user_msg, assistant_msg = chat["messages"]
        assert user_msg["models"] == ["model-x"]
        assert assistant_msg["model"] == "model-x"
        assert assistant_msg["modelName"] == "model-x"

    def test_append_assistant_text_extends_the_last_assistant_message(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "Scan",
            "chat": {
                "title": "Scan",
                "messages": [
                    {"id": "u1", "parentId": None, "childrenIds": ["a1"],
                     "role": "user", "content": "question", "models": ["m"]},
                    {"id": "a1", "parentId": "u1", "childrenIds": [],
                     "role": "assistant", "content": "first", "model": "m",
                     "done": True},
                ],
                "history": {
                    "messages": {
                        "u1": {"id": "u1", "parentId": None, "childrenIds": ["a1"],
                               "role": "user", "content": "question", "models": ["m"]},
                        "a1": {"id": "a1", "parentId": "u1", "childrenIds": [],
                               "role": "assistant", "content": "first", "model": "m",
                               "done": True},
                    },
                    "currentId": "a1",
                },
            },
        }
        captured = {}

        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            openwebui_service, "update_chat",
            lambda token, cid, chat: captured.update(chat=chat) or {"id": cid},
        )

        openwebui_service.append_assistant_text("tok", "chat-1", "second", model="m")

        chat = captured["chat"]
        assert len(chat["messages"]) == 2
        assert chat["messages"][1]["content"] == "first\n\nsecond"
        assert chat["history"]["messages"]["a1"]["content"] == "first\n\nsecond"
        assert chat["history"]["currentId"] == "a1"

    def test_append_assistant_text_without_assistant_message_fails(self, monkeypatch):
        stored = {
            "id": "chat-1",
            "title": "Scan",
            "chat": {"title": "Scan", "messages": [],
                     "history": {"messages": {}, "currentId": None}},
        }
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        with pytest.raises(openwebui_service.OpenWebUIChatNotFound):
            openwebui_service.append_assistant_text("tok", "chat-1", "text")


# ── routes ─────────────────────────────────────────────────────────────────


class TestChatRoutes:
    def test_requires_auth(self, client):
        assert client.get("/api/v1/llm/chats").status_code == 401
        assert client.get("/api/v1/llm/chats/current").status_code == 401

    def test_list_chats(self, client, user, monkeypatch):
        monkeypatch.setattr(
            openwebui_service, "list_chats",
            lambda token, limit=50, chat_type=None: [{"id": "chat-1", "title": "One"}],
        )
        r = client.get("/api/v1/llm/chats", headers=_auth(user))
        assert r.status_code == 200
        assert r.json()["data"] == [{"id": "chat-1", "title": "One"}]

    def test_create_sets_current_pointer(self, client, user, monkeypatch):
        monkeypatch.setattr(
            openwebui_service, "create_chat",
            lambda token, title="New Chat", model=None, chat_type=None: {
                "id": "chat-new", "title": title,
            },
        )
        r = client.post(
            "/api/v1/llm/chats", json={"title": "Plan"}, headers=_auth(user)
        )
        assert r.status_code == 200
        assert r.json()["data"]["id"] == "chat-new"

        monkeypatch.setattr(
            openwebui_service, "get_chat",
            lambda token, chat_id: {"id": chat_id, "title": "Plan"},
        )
        r = client.get("/api/v1/llm/chats/current", headers=_auth(user))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == "chat-new"

    def test_current_chat_clears_stale_pointer(self, client, user, Session, monkeypatch):
        user_id, _ = user
        with Session() as db:
            chat_session_service.set_current_chat_id(db, user_id, "gone")

        def raise_missing(token, chat_id):
            raise openwebui_service.OpenWebUIChatNotFound("Chat not found.")

        monkeypatch.setattr(openwebui_service, "get_chat", raise_missing)
        r = client.get("/api/v1/llm/chats/current", headers=_auth(user))
        assert r.status_code == 200
        assert r.json()["data"] is None
        with Session() as db:
            assert chat_session_service.get_current_chat_id(db, user_id) is None

    def test_clear_current_pointer(self, client, user, Session, monkeypatch):
        user_id, _ = user
        with Session() as db:
            chat_session_service.set_current_chat_id(db, user_id, "chat-1")

        r = client.delete("/api/v1/llm/chats/current", headers=_auth(user))
        assert r.status_code == 200
        assert r.json()["data"] == {"cleared": True}
        with Session() as db:
            assert chat_session_service.get_current_chat_id(db, user_id) is None

    def test_delete_clears_pointer(self, client, user, Session, monkeypatch):
        user_id, _ = user
        with Session() as db:
            chat_session_service.set_current_chat_id(db, user_id, "chat-1")

        monkeypatch.setattr(
            openwebui_service, "delete_chat", lambda token, chat_id: True
        )
        r = client.delete("/api/v1/llm/chats/chat-1", headers=_auth(user))
        assert r.status_code == 200
        assert r.json()["data"] == {"deleted": True, "chat_id": "chat-1"}
        with Session() as db:
            assert chat_session_service.get_current_chat_id(db, user_id) is None

    def test_missing_chat_returns_404(self, client, user, monkeypatch):
        def raise_missing(token, chat_id):
            raise openwebui_service.OpenWebUIChatNotFound("Chat not found.")

        monkeypatch.setattr(openwebui_service, "get_chat", raise_missing)
        r = client.get("/api/v1/llm/chats/nope", headers=_auth(user))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "llm_chat_not_found"

    def test_chat_prepends_history_and_persists(self, client, user, monkeypatch):
        stored = {
            "id": "chat-1",
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "earlier q"},
                    {"id": "m2", "role": "assistant", "content": "earlier a"},
                ],
                "history": {"messages": {}, "currentId": "m2"},
            },
        }
        seen = {}

        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)

        def fake_llm(messages, **kwargs):
            seen["messages"] = list(messages)
            seen["kwargs"] = kwargs
            return {"content": "fresh answer", "model": "qwen"}

        monkeypatch.setattr(llm_service, "chat", fake_llm)

        def fake_complete(token, chat_id, user_id, assistant_id, user_text, assistant_text, **kwargs):
            seen["complete"] = (chat_id, user_id, assistant_id, user_text, assistant_text, kwargs)
            return {"id": chat_id}

        monkeypatch.setattr(openwebui_service, "complete_turn", fake_complete)

        def boom(*args, **kwargs):
            raise AssertionError("append_exchange must not be called")

        monkeypatch.setattr(openwebui_service, "append_exchange", boom)

        r = client.post(
            "/api/v1/llm/chat",
            json={
                "messages": [{"role": "user", "content": "new q"}],
                "chat_id": "chat-1",
                "use_history": True,
                "history_limit": 2,
                "persist": True,
            },
            headers=_auth(user),
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["content"] == "fresh answer"
        assert data["chat_id"] == "chat-1"
        assert data["persisted"] is True
        assert seen["messages"] == [
            {"role": "user", "content": "earlier q", "id": "m1"},
            {"role": "assistant", "content": "earlier a", "id": "m2"},
            {"role": "user", "content": "new q"},
        ]
        # The compaction checkpoint and the persisted row must share one id.
        turn_id = seen["kwargs"].get("user_message_id")
        assert turn_id
        assert seen["complete"][1] == turn_id
        assert seen["complete"][0] == "chat-1"
        assert seen["complete"][3] == "new q"
        assert seen["complete"][4] == "fresh answer"
        assert seen["kwargs"].get("chat_id") == "chat-1"
        # The turn continues the chat's current node.
        assert seen["kwargs"].get("parent_id") == "m2"
        assert seen["complete"][5].get("parent_id") == "m2"
        assert seen["kwargs"].get("assistant_message_id") == seen["complete"][2]

    def test_chat_without_persist_does_not_write(self, client, user, monkeypatch):
        seen = {}

        def fake_llm(messages, **kwargs):
            seen.update(kwargs)
            return {"content": "answer", "model": "qwen"}

        monkeypatch.setattr(llm_service, "chat", fake_llm)

        def boom(*args, **kwargs):
            raise AssertionError("append_exchange must not be called")

        monkeypatch.setattr(openwebui_service, "append_exchange", boom)
        r = client.post(
            "/api/v1/llm/chat",
            json={"messages": [{"role": "user", "content": "q"}], "persist": False},
            headers=_auth(user),
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["chat_id"] is None
        assert data["persisted"] is False
        assert seen.get("user_message_id") is None

    def test_a_turn_stores_one_user_message(self, client, user, monkeypatch):
        """Open WebUI stores one user message per turn, so the turn is merged."""
        stored = {
            "id": "chat-1",
            "chat": {
                "messages": [],
                "history": {"messages": {}, "currentId": None},
            },
        }
        seen = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            llm_service, "chat",
            lambda messages, **kwargs: (seen.update(kwargs), {"content": "ok", "model": "qwen"})[1],
        )
        monkeypatch.setattr(
            openwebui_service, "complete_turn",
            lambda token, chat_id, user_id, assistant_id, user_text, assistant_text, **kwargs:
                seen.update(text=user_text) or {"id": chat_id},
        )

        r = client.post(
            "/api/v1/llm/chat",
            json={
                "messages": [
                    {"role": "user", "content": "first part"},
                    {"role": "user", "content": "second part"},
                ],
                "chat_id": "chat-1",
                "persist": True,
            },
            headers=_auth(user),
        )
        assert r.status_code == 200, r.text
        assert seen["text"] == "first part\n\nsecond part"
        assert seen["user_text"] == "first part\n\nsecond part"

    def test_the_guard_measures_the_stored_chain(self, client, user, monkeypatch):
        stored = {
            "id": "chat-1",
            "chat": {
                "messages": [
                    {"id": "m1", "role": "user", "content": "x" * 4000},
                ],
                "history": {"messages": {}, "currentId": "m1"},
            },
        }
        seen = {}
        monkeypatch.setattr(openwebui_service, "get_chat", lambda token, cid: stored)
        monkeypatch.setattr(
            llm_service, "chat",
            lambda messages, **kwargs: (seen.update(kwargs), {"content": "ok", "model": "qwen"})[1],
        )
        monkeypatch.setattr(
            openwebui_service, "complete_turn",
            lambda token, chat_id, user_id, assistant_id, user_text, assistant_text, **kwargs: {"id": chat_id},
        )

        r = client.post(
            "/api/v1/llm/chat",
            json={
                "messages": [{"role": "user", "content": "short"}],
                "chat_id": "chat-1",
                "history_limit": 1,
                "persist": True,
            },
            headers=_auth(user),
        )
        assert r.status_code == 200, r.text
        # The window is one message, but the model sees the whole chain.
        assert len(seen["estimate_messages"]) == 1
        assert len(seen["estimate_messages"][0]["content"]) == 4000
