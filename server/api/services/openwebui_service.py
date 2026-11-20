"""services/openwebui_service.py - per-user Open WebUI identity (Phase 4).

Open WebUI is the system of record for chat history, so calls made by the API
must be attributed to the signed-in user instead of the shared service key.

Open WebUI supports trusted-header sign-in: ``POST /api/v1/auths/signin`` with
``WEBUI_AUTH_TRUSTED_EMAIL_HEADER`` set returns a normal Open WebUI JWT for
that email (auto-provisioning the account when needed). This module performs
that exchange, caches the token per local user, and hands it to
``llm_service`` as the bearer token.

Security: the header name is only trusted by Open WebUI and must never be
reachable from the public internet. It is opt-in via
``OPENWEBUI_TRUSTED_AUTH=true``; when disabled, the static service key is used
exactly as before.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import jwt
import requests

from ..config import get_settings
from ..models import User

logger = logging.getLogger("openwebui")

CONNECT_TIMEOUT = 5.0
SIGNIN_TIMEOUT = 15.0
DEFAULT_CACHE_SECONDS = 3600

_lock = threading.Lock()
# user_id -> (token, expires_at_epoch)
_cache: dict[str, tuple[str, float]] = {}


class OpenWebUIError(RuntimeError):
    """Raised when a per-user Open WebUI token cannot be obtained."""


def _base_url() -> str:
    return os.getenv("OPENWEBUI_URL", "http://localhost:3000").rstrip("/")


def trusted_auth_enabled() -> bool:
    return get_settings().openwebui_trusted_auth


def _token_expiry(token: str) -> float:
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
    except jwt.PyJWTError:
        pass
    return time.time() + DEFAULT_CACHE_SECONDS


def _signin(email: str, name: str | None = None) -> str:
    s = get_settings()
    headers = {
        s.openwebui_trusted_header: email,
        "Content-Type": "application/json",
    }
    if name:
        headers[s.openwebui_trusted_name_header] = name
    try:
        resp = requests.post(
            f"{_base_url()}/api/v1/auths/signin",
            json={"email": email, "password": ""},
            headers=headers,
            timeout=(CONNECT_TIMEOUT, SIGNIN_TIMEOUT),
        )
    except requests.RequestException as exc:
        raise OpenWebUIError(f"Cannot reach Open WebUI: {exc}") from exc

    if resp.status_code >= 400:
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error") or ""
        except ValueError:
            detail = resp.text[:200]
        raise OpenWebUIError(
            f"Open WebUI trusted sign-in failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise OpenWebUIError("Open WebUI returned non-JSON sign-in response.") from exc
    token = data.get("token") if isinstance(data, dict) else None
    if not token:
        raise OpenWebUIError("Open WebUI returned no token.")
    return str(token)


def get_user_token(user: User) -> str | None:
    """Return a cached Open WebUI JWT for this user, or None when disabled.

    The caller falls back to the shared service key on None. Failures are
    logged without the token or header value.
    """
    if not trusted_auth_enabled():
        return None

    user_id = str(user.id)
    now = time.time()
    with _lock:
        cached = _cache.get(user_id)
        if cached and cached[1] - 30 > now:
            return cached[0]

    try:
        token = _signin(user.email, name=None)
    except OpenWebUIError as exc:
        logger.warning("Per-user Open WebUI token unavailable: %s", exc)
        return None

    with _lock:
        _cache[user_id] = (token, _token_expiry(token))
    return token


def invalidate(user_id: str) -> None:
    with _lock:
        _cache.pop(str(user_id), None)


def status() -> dict[str, Any]:
    s = get_settings()
    return {
        "trusted_auth": s.openwebui_trusted_auth,
        "trusted_header": s.openwebui_trusted_header if s.openwebui_trusted_auth else None,
        "cached_users": len(_cache),
    }


def generate_api_key(service_email: str = "mcp-service@respotit.local") -> str | None:
    """Auto-generate an Open WebUI API key via trusted-header sign-in.

    Returns the new key (sk-...) or None if trusted auth is disabled or fails.
    The key is NOT persisted to .env; callers should set OPENWEBUI_API_KEY
    or cache it in process memory.
    """
    if not trusted_auth_enabled():
        return None

    try:
        token = _signin(service_email, name="reSpotIt Service")
    except OpenWebUIError as exc:
        logger.warning("Auto API key generation: sign-in failed: %s", exc)
        return None

    try:
        resp = requests.post(
            f"{_base_url()}/api/v1/auths/api_key",
            headers={"Authorization": f"Bearer {token}"},
            timeout=(CONNECT_TIMEOUT, SIGNIN_TIMEOUT),
        )
    except requests.RequestException as exc:
        logger.warning("Auto API key generation: request failed: %s", exc)
        return None

    if resp.status_code >= 400:
        logger.warning(
            "Auto API key generation: Open WebUI returned %s: %s",
            resp.status_code, resp.text[:200],
        )
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    key = data.get("api_key") or data.get("key") if isinstance(data, dict) else None
    if not key:
        logger.warning("Auto API key generation: no key in response: %s", data)
        return None

    logger.info("Auto-generated Open WebUI API key for %s", service_email)
    return str(key)


# ── Chat history (Phase 5/6) ────────────────────────────────────────────────
#
# Open WebUI owns the chats. These helpers use the per-user token from
# ``get_user_token`` so every read/write lands in that user's own history.

CHAT_TIMEOUT = 20.0


class OpenWebUIChatNotFound(OpenWebUIError):
    """The chat does not exist or belongs to another Open WebUI user."""


class OpenWebUICredentialRejected(OpenWebUIError):
    """The cached per-user Open WebUI token is no longer accepted."""


def _require_token(token: str | None) -> str:
    if not token:
        raise OpenWebUIError(
            "Chat history requires a per-user Open WebUI token "
            "(set OPENWEBUI_TRUSTED_AUTH=true)."
        )
    return token


def _chat_request(
    method: str,
    path: str,
    token: str | None,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float = CHAT_TIMEOUT,
) -> Any:
    key = _require_token(token)
    try:
        resp = requests.request(
            method,
            f"{_base_url()}{path}",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=json_body,
            timeout=(CONNECT_TIMEOUT, timeout),
        )
    except requests.RequestException as exc:
        raise OpenWebUIError(f"Cannot reach Open WebUI chats: {exc}") from exc

    if resp.status_code == 404:
        raise OpenWebUIChatNotFound("Chat not found.")
    if resp.status_code == 401:
        # Open WebUI returns 401 (not 404) when a chat exists but belongs to
        # another user, to avoid leaking its existence. Treat that specific
        # response as not-found; other 401s mean the token is stale and the
        # caller should refresh it and retry.
        detail_text = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail_text = str(body.get("detail") or body.get("error") or "")
        except ValueError:
            detail_text = resp.text[:200]
        if "could not find what you're looking for" in detail_text.lower():
            raise OpenWebUIChatNotFound("Chat not found.")
        raise OpenWebUICredentialRejected("Open WebUI rejected the chat credential.")
    if resp.status_code >= 400:
        detail: Any = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error") or ""
        except ValueError:
            detail = resp.text[:200]
        raise OpenWebUIError(
            f"Open WebUI chat request failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )
    if resp.status_code == 204 or not resp.content:
        return None
    try:
        return resp.json()
    except ValueError as exc:
        raise OpenWebUIError("Open WebUI returned non-JSON chat data.") from exc


# Open WebUI renders an assistant reply only when the parent user message
# names exactly one model: the reply is grouped by that model and a user
# message with an empty "models" list leaves the answer invisible in the UI.
# Every chat this service writes therefore carries a concrete model id.
FALLBACK_MODEL_ID = "open-webui"

_model_cache: dict[str, str] = {}


def resolve_model_id(token: str | None = None) -> str:
    """Return a model id Open WebUI knows, for chats written by this service.

    Uses OPENWEBUI_MODEL when set, otherwise the first model Open WebUI
    lists, otherwise a placeholder id so the chat still renders.
    """
    override = os.getenv("OPENWEBUI_MODEL", "").strip()
    if override:
        return override

    key = token or ""
    with _lock:
        cached = _model_cache.get(key)
    if cached:
        return cached

    model_id = FALLBACK_MODEL_ID
    try:
        from . import llm_service

        model_id = llm_service.resolve_model(token=token)
    except Exception as exc:  # noqa: BLE001 - a chat must never fail to write
        logger.info("Using the placeholder model id for the chat: %s", exc)

    with _lock:
        _model_cache[key] = model_id
    return model_id


def list_chats(token: str | None, limit: int = 50, chat_type: str | None = None) -> list[dict[str, Any]]:
    path = "/api/v1/chats/"
    if chat_type:
        path += f"?chat_type={chat_type}"
    data = _chat_request("GET", path, token)
    rows = data if isinstance(data, list) else []
    return [row for row in rows if isinstance(row, dict)][: max(1, limit)]


def get_chat(token: str | None, chat_id: str) -> dict[str, Any]:
    data = _chat_request("GET", f"/api/v1/chats/{chat_id}", token)
    if not isinstance(data, dict):
        raise OpenWebUIChatNotFound("Chat not found.")
    return data


def create_chat(
    token: str | None,
    *,
    title: str = "New Chat",
    model: str | None = None,
    chat_type: str | None = None,
) -> dict[str, Any]:
    chat = {
        "title": title,
        "models": [(model or "").strip() or resolve_model_id(token)],
        "params": {},
        "messages": [],
        "history": {"messages": {}, "currentId": None},
        "tags": [],
        "files": [],
    }
    if chat_type:
        chat["chat_type"] = chat_type
    data = _chat_request("POST", "/api/v1/chats/new", token, json_body={"chat": chat})
    if not isinstance(data, dict) or not data.get("id"):
        raise OpenWebUIError("Open WebUI did not return the created chat.")
    return data


def update_chat(token: str | None, chat_id: str, chat: dict[str, Any]) -> dict[str, Any]:
    data = _chat_request(
        "POST", f"/api/v1/chats/{chat_id}", token, json_body={"chat": chat}
    )
    if not isinstance(data, dict):
        raise OpenWebUIChatNotFound("Chat not found.")
    return data


def delete_chat(token: str | None, chat_id: str) -> bool:
    _chat_request("DELETE", f"/api/v1/chats/{chat_id}", token)
    return True


def _context_summaries(chat: dict[str, Any]) -> dict[str, str]:
    """Map message id -> compaction summary from Open WebUI's history tree.

    Open WebUI writes a compaction checkpoint onto the history node, not onto
    the flat message list the UI renders, so the summary has to be read from
    ``chat.history.messages``.
    """
    history = chat.get("history")
    tree = history.get("messages") if isinstance(history, dict) else None
    summaries: dict[str, str] = {}
    if not isinstance(tree, dict):
        return summaries
    for message_id, node in tree.items():
        if not isinstance(node, dict):
            continue
        value = node.get("contextSummary") or node.get("context_summary")
        if isinstance(value, str) and value.strip():
            summaries[str(message_id)] = value
    return summaries


def history_messages(chat_row: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    """Return the last ``limit`` user/assistant turns as chat-completion rows.

    Each row keeps its Open WebUI ``id`` when present: context compaction
    anchors its persisted checkpoint on the first kept message, so dropping the
    id would make every turn re-summarise the same history.

    A row also keeps the compaction summary stored on that message. Open WebUI
    re-applies the newest summary checkpoint and drops everything before it, so
    a row without it would silently lose the compacted history.
    """
    if limit <= 0:
        return []
    chat = chat_row.get("chat") if isinstance(chat_row, dict) else None
    if not isinstance(chat, dict):
        return []
    messages = chat.get("messages")
    if not isinstance(messages, list):
        return []
    summaries = _context_summaries(chat)
    rows: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            row: dict[str, Any] = {"role": role, "content": content}
            message_id = message.get("id")
            if isinstance(message_id, str) and message_id:
                row["id"] = message_id
                summary = summaries.get(message_id)
                if summary:
                    row["contextSummary"] = summary
            rows.append(row)
    window = rows[-limit:]

    # The newest checkpoint can sit outside the window. Carry its summary on the
    # first row instead: Open WebUI then drops the rows before it (none) and
    # keeps the compacted history.
    if window:
        in_window = {row.get("id") for row in window if row.get("id")}
        outside = [
            (row.get("id"), summaries.get(str(row.get("id"))))
            for row in rows
            if row.get("id") not in in_window and summaries.get(str(row.get("id")))
        ]
        if outside and not any(row.get("contextSummary") for row in window):
            _latest_id, latest_summary = outside[-1]
            window[0]["contextSummary"] = latest_summary
    return window


def _safe_arguments(raw: Any) -> str:
    """Return a tool-call arguments string that Open WebUI can always parse.

    A saved chat can hold a truncated string from a model that hit its token
    limit (a URI cut before its closing quote). Open WebUI parses every
    replayed call, so an invalid value has to become ``{}``; otherwise the
    whole request fails with 500 "Failed to parse tool call arguments as JSON".
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "{}"
        return raw if isinstance(parsed, dict) else "{}"
    if raw is None:
        return "{}"
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _output_rows(output: list[Any]) -> list[dict[str, Any]]:
    """Rebuild a stored ``output`` item list into chat-completion rows.

    Mirrors Open WebUI's ``convert_output_to_messages`` for the two items this
    server writes: a ``function_call`` becomes an assistant ``tool_calls``
    entry, and its ``function_call_output`` becomes a ``tool`` row. Open WebUI
    replays exactly that for a saved chat, so the API has to carry the rows too.
    Without them the model sees its own call requests and no results, cannot
    tell whether a step ran, and asks the user to prompt it again.
    """
    calls: list[dict[str, Any]] = []
    results: dict[str, str] = {}
    for item in output:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        call_id = str(item.get("call_id") or "")
        if not call_id:
            continue
        if kind == "function_call":
            # Open WebUI only replays a call that reached a terminal status.
            if item.get("status") not in {"completed", "failed", "rejected"}:
                continue
            arguments = item.get("arguments")
            arguments = _safe_arguments(arguments)
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": arguments,
                    },
                }
            )
        elif kind == "function_call_output":
            parts = item.get("output")
            text = ""
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        value = part.get("text")
                        if isinstance(value, str):
                            text += value
            elif isinstance(parts, str):
                text = parts
            results[call_id] = text
    if not calls:
        return []
    rows: list[dict[str, Any]] = [{"role": "assistant", "content": "", "tool_calls": calls}]
    for call in calls:
        # An unmatched output is dropped, the way Open WebUI's
        # ``get_paired_tool_call_ids`` drops it.
        if call["id"] in results:
            rows.append(
                {"role": "tool", "tool_call_id": call["id"], "content": results[call["id"]]}
            )
    return rows


def compacted_chain(chat_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the chain Open WebUI would replay for a saved chat.

    Open WebUI rebuilds an assistant node's stored ``output`` items into
    assistant ``tool_calls`` plus ``tool`` rows, and drops every message before
    the newest compaction checkpoint in favour of that checkpoint's summary.
    The API needs the same view for two reasons:

    * its own request messages must match what the model saw, so a tool-loop
      follow-up (which cannot use the saved-chat replay, because Open WebUI
      would drop the tool result) keeps the same context as the first request;
    * the budget guard has to measure what the model actually receives, not
      the messages that were compacted away.
    """
    chat = chat_row.get("chat") if isinstance(chat_row, dict) else None
    if not isinstance(chat, dict):
        return []
    messages = chat.get("messages")
    if not isinstance(messages, list):
        return []
    summaries = _context_summaries(chat)
    rows: list[dict[str, Any]] = []
    checkpoint: tuple[int, str] | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        output = message.get("output")
        message_id = message.get("id")

        expanded: list[dict[str, Any]] = []
        if role == "assistant" and isinstance(output, list) and output:
            # An assistant node with output is replayed from its items only;
            # Open WebUI ignores that node's prose, because the items already
            # describe the turn. Keeping the shapes identical keeps the prompt
            # prefix identical between the first request and a tool follow-up.
            expanded = _output_rows(output)
        if not expanded:
            if role not in {"user", "assistant"} or not content.strip():
                continue
            row: dict[str, Any] = {"role": role, "content": content}
            if isinstance(message_id, str) and message_id:
                row["id"] = message_id
            expanded = [row]

        if isinstance(message_id, str) and message_id:
            # Both the flat list and the history tree can carry the checkpoint;
            # Open WebUI anchors it on the message id, so only an id that is
            # actually in the flat chain starts a new view.
            summary = summaries.get(message_id)
            flat_value = message.get("contextSummary") or message.get("context_summary")
            if isinstance(flat_value, str) and flat_value.strip():
                summary = flat_value
            if summary:
                checkpoint = (len(rows), summary)
        rows.extend(expanded)
    if checkpoint is None:
        return rows
    index, summary = checkpoint
    # Open WebUI keeps the checkpoint message itself and prepends the summary
    # as a system message.
    view: list[dict[str, Any]] = [
        {"role": "system", "content": f"[CONVERSATION SUMMARY]\n{summary}"}
    ]
    view.extend(rows[index:])
    return view


def current_message_id(chat_row: dict[str, Any]) -> str | None:
    """Return the id of the chat's current (last) message node.

    Open WebUI attaches the next turn to this node, so the API has to send it
    back as the ``parentId`` of the new user message.
    """
    chat = chat_row.get("chat") if isinstance(chat_row, dict) else None
    if not isinstance(chat, dict):
        return None
    history = chat.get("history")
    current = history.get("currentId") if isinstance(history, dict) else None
    if isinstance(current, str) and current:
        return current
    messages = chat.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict):
                message_id = message.get("id")
                if isinstance(message_id, str) and message_id:
                    return message_id
    return None


def _tool_output_items(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Build Open WebUI's output items for a tool exchange.

    Open WebUI renders ``output`` items and rebuilds them into OpenAI-style
    ``tool_calls`` / ``tool`` messages when it replays the chat, so storing
    the exchange keeps the tool result visible to the next turn.
    """
    items: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or "")
        name = str(call.get("name") or "")
        if not call_id or not name:
            continue
        arguments = call.get("arguments")
        arguments = _safe_arguments(arguments)
        # The response carries a short ``result`` for the phone; the stored
        # chain should keep the whole tool answer so the next turn can still
        # read the ids and counts it needs.
        text = call.get("result_full")
        if not isinstance(text, str) or not text:
            text = str(call.get("result") or "")
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "status": "completed",
                "name": name,
                "arguments": arguments,
            }
        )
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": [
                    {"type": "input_text", "text": text}
                ],
            }
        )
    return items


def complete_turn(
    token: str | None,
    chat_id: str,
    user_message_id: str,
    assistant_message_id: str,
    user_text: str,
    assistant_text: str,
    *,
    model: str | None = None,
    parent_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Finish a turn that Open WebUI already inserted into the chat.

    When a request carries a saved ``chat_id``, Open WebUI writes the turn
    itself: the user message and an empty assistant placeholder go into
    ``history.messages`` (this is what lets native context compaction anchor
    its checkpoint on a real message). This function fills in the answer and
    mirrors the turn into the flat ``messages`` list the UI renders.
    """
    row = get_chat(token, chat_id)
    chat = dict(row.get("chat") or {})
    messages = list(chat.get("messages") or [])
    history = dict(chat.get("history") or {})
    tree = dict(history.get("messages") or {})

    now = int(time.time())
    model_id = (model or "").strip() or resolve_model_id(token)

    user_node = tree.get(user_message_id)
    user_node = dict(user_node) if isinstance(user_node, dict) else {}
    user_node.update(
        {
            "id": user_message_id,
            "parentId": parent_id if parent_id is not None else user_node.get("parentId"),
            "role": "user",
            "content": user_text,
            "models": [model_id],
        }
    )
    user_node.setdefault("childrenIds", [assistant_message_id])
    user_node.setdefault("timestamp", now)

    assistant_node = tree.get(assistant_message_id)
    assistant_node = dict(assistant_node) if isinstance(assistant_node, dict) else {}
    assistant_node.update(
        {
            "id": assistant_message_id,
            "parentId": user_message_id,
            "role": "assistant",
            "content": assistant_text,
            "model": model_id,
            "modelName": model_id,
            "done": True,
        }
    )
    assistant_node.setdefault("childrenIds", [])
    assistant_node.setdefault("timestamp", now)

    # Keep the tool exchange: Open WebUI stored the first (tool-calling)
    # answer of the turn, and the tool results belong to the stored chain so
    # the next turn can replay them.
    exchange = _tool_output_items(tool_calls)
    if exchange:
        assistant_node["output"] = exchange

    if parent_id and isinstance(tree.get(parent_id), dict):
        siblings = tree[parent_id].get("childrenIds")
        children = list(siblings) if isinstance(siblings, list) else []
        if user_message_id not in children:
            children.append(user_message_id)
        tree[parent_id]["childrenIds"] = children

    tree[user_message_id] = user_node
    tree[assistant_message_id] = assistant_node
    history["messages"] = tree
    history["currentId"] = assistant_message_id

    flat_ids = {m.get("id") for m in messages if isinstance(m, dict)}
    if user_message_id not in flat_ids:
        messages.append(user_node)
    if assistant_message_id not in flat_ids:
        messages.append(assistant_node)

    chat["messages"] = messages
    chat["history"] = history

    return update_chat(token, chat_id, chat)


def append_exchange(
    token: str | None,
    chat_id: str,
    user_text: str,
    assistant_text: str,
    *,
    model: str | None = None,
    title: str | None = None,
    user_message_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append one user/assistant turn to an existing Open WebUI chat.

    Both the flat ``messages`` list (what the UI renders) and the
    ``history.messages`` tree (what the UI navigates) are kept in sync, which
    is the same shape the Open WebUI frontend writes.
    """
    import uuid

    row = get_chat(token, chat_id)
    chat = dict(row.get("chat") or {})
    messages = list(chat.get("messages") or [])
    history = dict(chat.get("history") or {})
    tree = dict(history.get("messages") or {})

    parent_id = history.get("currentId")
    if not parent_id and messages:
        last = messages[-1]
        if isinstance(last, dict):
            parent_id = last.get("id")

    now = int(time.time())
    user_id = (user_message_id or "").strip() or str(uuid.uuid4())
    assistant_id = str(uuid.uuid4())
    model_id = (model or "").strip() or resolve_model_id(token)

    user_message = {
        "id": user_id,
        "parentId": parent_id,
        "childrenIds": [assistant_id],
        "role": "user",
        "content": user_text,
        "timestamp": now,
        "models": [model_id],
    }
    assistant_message = {
        "id": assistant_id,
        "parentId": user_id,
        "childrenIds": [],
        "role": "assistant",
        "content": assistant_text,
        "model": model_id,
        "modelName": model_id,
        "timestamp": now,
        "done": True,
    }
    exchange = _tool_output_items(tool_calls)
    if exchange:
        assistant_message["output"] = exchange

    if parent_id and isinstance(tree.get(parent_id), dict):
        siblings = tree[parent_id].get("childrenIds")
        children = list(siblings) if isinstance(siblings, list) else []
        children.append(user_id)
        tree[parent_id]["childrenIds"] = children

    # Open WebUI can write a compaction checkpoint onto this turn while the
    # model call is running. Keep it when the stored row replaces the stub.
    existing = tree.get(user_id)
    if isinstance(existing, dict):
        summary = existing.get("contextSummary") or existing.get("context_summary")
        if isinstance(summary, str) and summary.strip():
            user_message["contextSummary"] = summary

    tree[user_id] = user_message
    tree[assistant_id] = assistant_message
    history["messages"] = tree
    history["currentId"] = assistant_id

    chat["messages"] = messages + [user_message, assistant_message]
    chat["history"] = history

    if title and str(row.get("title") or "").strip() in {"", "New Chat"}:
        chat["title"] = title

    return update_chat(token, chat_id, chat)


def append_assistant_text(
    token: str | None,
    chat_id: str,
    text: str,
    *,
    model: str | None = None,
    separator: str = "\n\n",
) -> dict[str, Any]:
    """Append ``text`` to the last assistant message of a chat.

    Open WebUI shows one assistant bubble per turn, so streaming progress
    into that bubble keeps a chat readable instead of adding a new turn for
    every update. The flat ``messages`` list and the ``history.messages``
    tree are kept in sync, like :func:`append_exchange` does.
    """
    row = get_chat(token, chat_id)
    chat = dict(row.get("chat") or {})
    messages = list(chat.get("messages") or [])
    history = dict(chat.get("history") or {})
    tree = dict(history.get("messages") or {})

    last: dict[str, Any] | None = None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            last = message
            break
    if last is None:
        raise OpenWebUIChatNotFound("Chat has no assistant message to extend.")

    model_id = (model or "").strip() or str(last.get("model") or "") or resolve_model_id(token)
    content = str(last.get("content") or "")
    updated = {**last, "content": f"{content}{separator}{text}" if content else text}
    if model_id:
        updated["model"] = model_id
        updated["modelName"] = model_id

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("id") == updated.get("id"):
            messages[index] = updated
            break
    if updated.get("id"):
        tree[updated["id"]] = updated

    chat["messages"] = messages
    chat["history"] = {**history, "messages": tree}

    return update_chat(token, chat_id, chat)
