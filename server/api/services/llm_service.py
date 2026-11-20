"""llm_service.py - Open WebUI proxy for the local llama-server model.

Open WebUI sits in front of llama-server (OpenAI-compatible) and adds user
accounts, conversation history and model management. Keeping the Open WebUI
personal access token on the backend means the phone never holds it.

Environment:
    OPENWEBUI_URL      Base URL (default http://localhost:3000)
    OPENWEBUI_API_KEY  Personal access token (sk-...). Unset = feature disabled.
    OPENWEBUI_MODEL    Optional model id. Default: first model Open WebUI lists.

The endpoints are additive: the frozen /api/v1 library contract is untouched.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import date
from typing import Any, Iterable

from . import mcp_service, tool_registry

import requests

logger = logging.getLogger("llm")

CONNECT_TIMEOUT = 5.0
MODELS_TIMEOUT = 15.0
DEFAULT_READ_TIMEOUT = 30.0

# Diagnostic tracing of every request/response that leaves this module.
# Off by default; set LLM_TRACE=1 on the api service to log full prompts and
# completions (may contain user data -- use only while debugging).
TRACE_CHARS = 20000


def _trace_enabled() -> bool:
    return os.getenv("LLM_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def _trace(label: str, payload: Any) -> None:
    if not _trace_enabled():
        return
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(payload)
    if len(text) > TRACE_CHARS:
        text = text[:TRACE_CHARS] + f"... (+{len(text) - TRACE_CHARS} chars)"
    logger.info("LLM TRACE %s: %s", label, text)


class LLMError(RuntimeError):
    """Base class for errors the router turns into a contract envelope."""

    status_code = 502
    code = "llm_error"


class LLMNotConfigured(LLMError):
    status_code = 503
    code = "llm_not_configured"


class LLMUnavailable(LLMError):
    status_code = 502
    code = "llm_unavailable"


class LLMUpstreamError(LLMError):
    status_code = 502
    code = "llm_upstream_error"


class LLMCredentialRejected(LLMUpstreamError):
    """The Open WebUI credential was rejected (stale per-user token)."""
    code = "llm_unauthorized"


class LLMTimeout(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMContextExceeded(LLMError):
    """The request cannot fit the local model's context window.

    Raised by the budget guard before the request leaves this module, so the
    caller gets an actionable message instead of llama-server's raw
    "exceeds the available context size" error.
    """

    status_code = 413
    code = "llm_context_exceeded"


ROUTER_MODES = ("auto", "all", "off")
DEFAULT_CONTEXT_TOKENS = 16384
DEFAULT_BUDGET_FRACTION = 0.8
# L3: after a search hit, how many real schemas to inject for native calling.
DEFAULT_REVEAL = 5
MAX_REVEAL = 8

# Hot tools hidden for one turn when the user names their saved library. The
# catalog `search` is otherwise always advertised and a small model picks it
# for "search the library" on the strength of its name alone. It stays
# reachable through tool_search, and the next non-library turn advertises it
# again.
HIDDEN_ON_LIBRARY_TURN = frozenset({"search", "spotify_search"})

# Per-chat revealed tools. llama.cpp renders the tool schemas before the
# messages, so a tool block that changes between the turns of a conversation
# invalidates the whole prompt cache: the same prompt reuses 4,542 tokens, a
# changed tool array reuses none (~1.7 s per turn on the 9B). The revealed set
# is therefore kept per chat, and the block only changes when the
# conversation needs a tool it has not seen.
MAX_STICKY_REVEALS = 8
REVEAL_CACHE_MAX_CHATS = 256

_reveal_cache: "OrderedDict[str, tuple[list[str], bool]]" = OrderedDict()
_reveal_lock = threading.Lock()


def _cached_reveals(chat_id: str | None) -> tuple[list[str], bool]:
    """The revealed tools a chat carries over, and whether it has searched.

    A chat without an id (the first turn of a conversation) starts empty.
    """
    if not chat_id:
        return [], False
    with _reveal_lock:
        entry = _reveal_cache.get(chat_id)
        if entry is None:
            return [], False
        _reveal_cache.move_to_end(chat_id)
        return list(entry[0]), entry[1]


def _store_reveals(chat_id: str | None, names: list[str], searched: bool) -> None:
    """Keep the revealed set for the next turn of this chat.

    Only the most recent tools are kept: a conversation that needs many
    different tools would otherwise grow the block without bound.
    """
    if not chat_id:
        return
    with _reveal_lock:
        _reveal_cache[chat_id] = (list(names)[-MAX_STICKY_REVEALS:], searched)
        _reveal_cache.move_to_end(chat_id)
        while len(_reveal_cache) > REVEAL_CACHE_MAX_CHATS:
            _reveal_cache.popitem(last=False)


def _router_mode(explicit: str | None = None) -> str:
    """Tool-routing mode: ``auto`` (L0 ladder), ``all`` (legacy) or ``off``."""
    raw = (explicit or os.getenv("LLM_TOOL_ROUTER", "") or "auto").strip().lower()
    return raw if raw in ROUTER_MODES else "auto"


def _context_limit(explicit: int | None = None) -> int:
    """Context window to budget against (llama-server ``-c``)."""
    if explicit and explicit > 0:
        return int(explicit)
    try:
        value = int(os.getenv("LLM_CONTEXT_TOKENS", "") or DEFAULT_CONTEXT_TOKENS)
    except ValueError:
        value = DEFAULT_CONTEXT_TOKENS
    return value if value > 0 else DEFAULT_CONTEXT_TOKENS


def _budget_fraction() -> float:
    try:
        value = float(os.getenv("LLM_CONTEXT_BUDGET", "") or DEFAULT_BUDGET_FRACTION)
    except ValueError:
        value = DEFAULT_BUDGET_FRACTION
    return min(0.95, max(0.3, value))


def _reveal_limit() -> int:
    """How many real schemas a search may inject (0 disables L3)."""
    try:
        value = int(os.getenv("LLM_TOOL_REVEAL", "") or DEFAULT_REVEAL)
    except ValueError:
        value = DEFAULT_REVEAL
    return min(MAX_REVEAL, max(0, value))


# L1: schemas revealed from the user's own words before the model asks for them.
# Measured need: a 9B model calls a plausible-but-wrong hot tool (the Spotify
# catalog `search` for "how many albums do I have") or no tool at all instead of
# calling tool_search. When the router's own ranking is unambiguous, hand it the
# right schema up front; the extra schema costs ~300 prompt tokens.
DEFAULT_PREREVEAL = 2
DEFAULT_PREREVEAL_TOKENS = 800
MAX_PREREVEAL = 4
MAX_PREREVEAL_TOKENS = 2000
PREREVEAL_MIN_SCORE = 4.0


def _prereveal_limit() -> int:
    """How many schemas the pre-reveal may add (0 disables L1)."""
    try:
        value = int(os.getenv("LLM_TOOL_PREREVEAL", "") or DEFAULT_PREREVEAL)
    except ValueError:
        value = DEFAULT_PREREVEAL
    return min(MAX_PREREVEAL, max(0, value))


def _prereveal_token_budget() -> int:
    """Prompt tokens the pre-reveal may spend on schemas.

    Bounds the worst case (hot tools ~3.0k + this + meta tools) so a long chat
    still fits the guard's budget. A schema that does not fit is skipped, not
    truncated.
    """
    try:
        value = int(os.getenv("LLM_TOOL_PREREVEAL_TOKENS", "") or DEFAULT_PREREVEAL_TOKENS)
    except ValueError:
        value = DEFAULT_PREREVEAL_TOKENS
    return min(MAX_PREREVEAL_TOKENS, max(0, value))


def _prereveal_min_score() -> float:
    """Below this BM25 score the match is a stray keyword, not an intent."""
    try:
        value = float(os.getenv("LLM_TOOL_PREREVEAL_MIN", "") or PREREVEAL_MIN_SCORE)
    except ValueError:
        value = PREREVEAL_MIN_SCORE
    return max(0.0, value)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Text of the newest user message (string or content-part list)."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    return ""


def _pre_reveal(
    registry: tool_registry.Registry | None,
    messages: list[dict[str, Any]],
) -> list[str]:
    """Names of non-hot tools to advertise natively for this request.

    Deterministic: the router's BM25 ranking over the newest user message, never
    a model call. Hot tools need no help and are skipped; a query with no strong
    match (greetings, small talk) reveals nothing.
    """
    limit = _prereveal_limit()
    if registry is None or limit <= 0:
        return []
    text = _last_user_text(messages)
    if not text.strip():
        return []

    hot_names = {
        (tool.get("function") or {}).get("name")
        for tool in tool_registry.hot_tools(registry)
        if isinstance(tool, dict)
    }
    threshold = _prereveal_min_score()
    budget = _prereveal_token_budget()
    used = 0
    picked: list[str] = []
    for score, card in tool_registry.rank(registry, text, limit + len(hot_names)):
        if score < threshold:
            break
        if card.name in hot_names or card.name in picked:
            continue
        schema = registry.schemas.get(card.name)
        if not schema:
            continue
        size = _estimate_tokens(schema)
        if used + size > budget:
            continue
        picked.append(card.name)
        used += size
        if len(picked) >= limit:
            break
    return picked


# ---------------------------------------------------------------------------
# Textual tool-call markup
#
# A local model sometimes emits its chat template's textual tool-call markup in
# ``content`` instead of a native ``tool_calls`` array: llama.cpp only parses the
# exact special-token form, and a model that writes the markup as plain text gets
# it back as prose. The loop used to treat that as the final answer, so the call
# was never executed and the assistant appeared to stop and wait for the user to
# say "continue". These helpers execute the markup like a native call.
# ---------------------------------------------------------------------------

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S | re.I)
_FUNCTION_TAG = re.compile(r"<function\s*=\s*([A-Za-z0-9_.:\-]+)\s*>", re.I)
_PARAMETER_TAG = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:\-]+)\s*>(.*?)"
    r"(?:</parameter>|(?=</function>)|(?=<parameter\s*=)|$)",
    re.S | re.I,
)
_BARE_FUNCTION = re.compile(r"<function\s*=[^>]*>.*?(?:</function>|$)", re.S | re.I)


def _decode_parameter(raw: str) -> Any:
    """A parameter value as JSON when it parses, else its literal text.

    The template writes lists and objects as JSON and plain strings bare, so a
    string that fails to parse (``spotify:track:...``) has to stay a string.
    """
    text = raw.strip()
    if text.endswith("</function>"):
        text = text[: -len("</function>")].strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _parse_text_tool_calls(content: Any) -> list[dict[str, Any]]:
    """Turn ``<tool_call>`` markup into OpenAI-style tool-call dicts.

    Handles the template shape used by the local model::

        <tool_call>
        <function=play>
        <parameter=uris>
        ["spotify:track:..."]
        </parameter>
        </function>
        </tool_call>

    A bare ``<function=...>`` without a ``<tool_call>`` wrapper is accepted too,
    because a truncated reply drops the outer tag.
    """
    if not isinstance(content, str) or "<function" not in content.lower():
        return []
    blocks = [match.group(1) for match in _TOOL_CALL_BLOCK.finditer(content)]
    if not blocks:
        blocks = [content]
    calls: list[dict[str, Any]] = []
    for block in blocks:
        tags = list(_FUNCTION_TAG.finditer(block))
        for position, tag in enumerate(tags):
            name = tag.group(1).strip()
            if not name:
                continue
            body_end = tags[position + 1].start() if position + 1 < len(tags) else len(block)
            body = block[tag.end():body_end]
            arguments: dict[str, Any] = {}
            for param in _PARAMETER_TAG.finditer(body):
                key = param.group(1).strip()
                if key:
                    arguments[key] = _decode_parameter(param.group(2))
            calls.append(
                {
                    "id": f"call_text_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    return calls


def _strip_text_tool_calls(content: Any) -> str:
    """Remove tool-call markup so the user never sees it in an answer."""
    if not isinstance(content, str):
        return ""
    text = _TOOL_CALL_BLOCK.sub("", content)
    text = _BARE_FUNCTION.sub("", text)
    return text.strip()


def _normalise_tool_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Return ``(arguments, valid)`` for one model tool call.

    A model that runs out of tokens mid-call returns an arguments string that
    is not valid JSON — typically a URI cut before its closing quote. Open WebUI
    parses every replayed tool call, so forwarding the raw string makes the
    *next* request fail with ``Failed to parse tool call arguments as JSON``
    (HTTP 500). Callers must send ``json.dumps`` of the returned dict and treat
    ``valid=False`` as "do not run this half-parsed call".
    """
    if isinstance(raw, dict):
        return raw, True
    if raw is None or raw == "":
        return {}, True
    if not isinstance(raw, str):
        return {}, False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}, False
    if isinstance(parsed, dict):
        return parsed, True
    return {}, False


def _chat_fields(
    chat_id: str | None,
    messages: list[dict[str, Any]],
    user_message_id: str | None = None,
    assistant_message_id: str | None = None,
    parent_id: str | None = None,
    model: str | None = None,
    user_text: str | None = None,
) -> dict[str, Any]:
    """Top-level Open WebUI fields that let native context compaction run.

    ``main.py`` pops ``chat_id`` / ``user_message`` / ``parent_id`` /
    ``message_ids`` from the request body and builds its own ``metadata``,
    overwriting anything the caller sends there. It derives
    ``user_message_id`` from ``user_message["id"]``, and compaction only runs
    for a saved ``chat_id`` with a ``user_message_id``. So the fields have to be
    top-level.

    A saved chat also needs ``message_ids``: Open WebUI writes the turn into
    the chat itself (``Chats.insert_chat_turn``) and rejects a request without
    an assistant message id with ``409 Message already exists or has an
    invalid ID``. Writing the turn upstream is what lets the compaction
    checkpoint land on a real message, and it keeps the chat tree in the shape
    the Open WebUI UI expects.

    For a saved chat Open WebUI replays the stored chain instead of the
    messages in the request, so ``user_text`` is what the turn is stored with.
    """
    if not chat_id:
        return {}
    fields: dict[str, Any] = {"chat_id": str(chat_id)}
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    message_id = user_message_id or (last_user.get("id") if last_user else None)
    if not message_id:
        return fields
    content = user_text if user_text is not None else (last_user or {}).get("content", "")
    user_message: dict[str, Any] = {
        "id": str(message_id),
        "role": "user",
        "content": content,
    }
    if parent_id:
        user_message["parentId"] = str(parent_id)
    if model:
        user_message["models"] = [str(model)]
    fields["user_message"] = user_message
    if assistant_message_id:
        fields["message_ids"] = [
            {
                "model_id": str(model or ""),
                "message_id": str(assistant_message_id),
            }
        ]
    return fields


def _estimate_tokens(payload: Any) -> int:
    """Conservative token estimate for anything JSON-serialisable.

    Deliberately not exact: the point is to catch a 46k-token request before it
    reaches the model, and 4 chars/token over-estimates JSON and code, which is
    the safe direction.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    return max(1, len(text) // 4)


def _guard_context(
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    context_tokens: int,
    baseline_tokens: int = 0,
) -> int:
    """Return the estimated prompt size, or raise before the model sees it.

    ``baseline_tokens`` is the size of the history the upstream model will
    actually see. For a saved chat Open WebUI replays the whole stored chain,
    not the window the caller sent, so the guard has to use the larger number.
    """
    tool_tokens = _estimate_tokens(tools) if tools else 0
    message_tokens = _estimate_tokens(messages)
    total = tool_tokens + max(message_tokens, baseline_tokens)
    limit = int(context_tokens * _budget_fraction())
    if total > limit:
        raise LLMContextExceeded(
            f"Prompt is ~{total} tokens (tools {tool_tokens}, messages "
            f"{message_tokens}) but the local model accepts ~{limit} of "
            f"{context_tokens}. Raise -c on llama-server, start a new chat, "
            f"or lower the history limit."
        )
    return total


def _base_url() -> str:
    return os.getenv("OPENWEBUI_URL", "http://localhost:3000").rstrip("/")


_API_KEY_FILE = "/data/.owui_api_key"


def _api_key() -> str:
    # 1. Environment variable (explicit config).
    env_key = os.getenv("OPENWEBUI_API_KEY", "").strip()
    if env_key:
        return env_key
    # 2. Auto-generated key persisted to file.
    try:
        with open(_API_KEY_FILE) as f:
            return f.read().strip()
    except (OSError, IOError):
        return ""


def is_configured(token: str | None = None) -> bool:
    return bool(token or _api_key())


def _headers(token: str | None = None) -> dict[str, str]:
    key = (token or "").strip() or _api_key()
    if not key:
        raise LLMNotConfigured(
            "LLM is not configured. Set OPENWEBUI_API_KEY on the server "
            "(Open WebUI: Settings -> Account -> API Keys)."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: tuple[float, float] | float | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    url = _base_url() + path
    _trace("-> {} {}".format(method, path), json_body)
    try:
        resp = requests.request(
            method,
            url,
            headers=_headers(token),
            json=json_body,
            timeout=timeout if timeout is not None else (CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        )
    except requests.Timeout as exc:
        raise LLMTimeout("The LLM did not answer in time.") from exc
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Cannot reach Open WebUI at {_base_url()}.") from exc

    if resp.status_code >= 400:
        _trace("<- {} {} (error)".format(resp.status_code, path), resp.text[:4000])
        detail: Any = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error") or ""
            else:
                detail = body
        except ValueError:
            detail = resp.text[:300]
        message = f"Open WebUI returned {resp.status_code}"
        if detail:
            message += f": {detail}"
        if resp.status_code == 401:
            raise LLMCredentialRejected(message)
        raise LLMUpstreamError(message)

    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMUpstreamError("Open WebUI returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise LLMUpstreamError("Open WebUI returned an unexpected response.")
    _trace("<- {} {}".format(resp.status_code, path), data)
    return data


def list_models(token: str | None = None) -> list[dict[str, Any]]:
    """Return the models Open WebUI currently exposes."""
    data = _request("GET", "/api/models", timeout=(CONNECT_TIMEOUT, MODELS_TIMEOUT),
                    token=token)
    rows = data.get("data")
    if not isinstance(rows, list):
        raise LLMUpstreamError("Open WebUI returned an unexpected model list.")

    models: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not model_id:
            continue
        models.append(
            {
                "id": str(model_id),
                "name": str(row.get("name") or model_id),
                "owned_by": str(row.get("owned_by") or ""),
            }
        )
    return models


def resolve_model(models: list[dict[str, Any]] | None = None,
                  token: str | None = None) -> str:
    """Pick OPENWEBUI_MODEL when set, else the first model Open WebUI lists."""
    override = os.getenv("OPENWEBUI_MODEL", "").strip()
    if override:
        return override
    available = list_models(token=token) if models is None else models
    if not available:
        raise LLMUnavailable(
            "Open WebUI has no model connected. "
            "Check Admin -> Settings -> Connections."
        )
    return str(available[0]["id"])


def _run_tool(
    name: str,
    arguments: dict[str, Any],
    registry: tool_registry.Registry | None,
    route: dict[str, Any],
    state: dict[str, Any],
    reveal_limit: int = 0,
) -> str:
    """Execute one tool call: router meta-tools locally, everything else via MCP.

    ``state`` carries the revealed schemas (``revealed``), the names a search
    revealed this turn (``search_hits``) and whether the chat has ever
    revealed anything by searching (``searched``). A revealed schema is
    advertised on the next round, so the model can call it natively instead of
    going through ``tool_call``.
    """
    if registry is not None and name in tool_registry.META_NAMES:
        if name == "tool_search":
            query = str(arguments.get("query") or "").strip()
            limit = arguments.get("limit") or tool_registry.DEFAULT_RESULTS
            cards = tool_registry.search(registry, query, limit)
            route["searches"] += 1
            for card in cards[: max(0, reveal_limit)]:
                if card.name not in state["revealed"]:
                    state["revealed"].append(card.name)
                    state["search_hits"].append(card.name)
                    state["searched"] = True
            return tool_registry.render_search(query, cards, len(registry))
        if name == "tool_describe":
            wanted = str(arguments.get("name") or "").strip()
            schema = tool_registry.describe(registry, wanted)
            if schema is None:
                return f"Unknown tool: {wanted}. Use tool_search first."
            if wanted not in state["revealed"]:
                state["revealed"].append(wanted)
                state["search_hits"].append(wanted)
                state["searched"] = True
            # The function body is what the model needs; the outer
            # {"type": "function"} wrapper is meaningless in a tool result.
            body = schema.get("function") if isinstance(schema.get("function"), dict) else schema
            return json.dumps(body, ensure_ascii=False)
        # tool_call: indirect execution of a tool found with tool_search.
        target = str(arguments.get("name") or "").strip()
        inner = arguments.get("arguments")
        if not isinstance(inner, dict):
            inner = {}
        if not target:
            return "tool_call needs a tool name."
        if target in tool_registry.META_NAMES:
            return "tool_call cannot call the router tools."
        return mcp_service.call_tool(target, inner)
    if not name:
        return "Unknown tool."
    return mcp_service.call_tool(name, arguments)


def chat(
    messages: Iterable[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    use_tools: bool = True,
    tool_rounds: int = 12,
    token: str | None = None,
    timeout: tuple[float, float] | float | None = None,
    tool_router: str | None = None,
    context_tokens: int | None = None,
    chat_id: str | None = None,
    user_message_id: str | None = None,
    assistant_message_id: str | None = None,
    parent_id: str | None = None,
    user_text: str | None = None,
    estimate_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Send a non-streaming chat completion through Open WebUI.

    Tool routing (``tool_router``, default from ``LLM_TOOL_ROUTER``):

    ``auto``  L0: advertise ten hot playback tools plus ``tool_search`` /
              ``tool_describe`` / ``tool_call``. Before the first request the
              router also reveals the best non-hot tools for the user's own
              words (L1, ``LLM_TOOL_PREREVEAL``), because a small model reaches
              for a plausible hot tool instead of searching. A search hit
              reveals the top-k real schemas on the next round (L3) and drops
              the describe/call indirection, so the actual call is native
              function calling. Every other MCP tool stays callable but costs no
              prompt tokens until it is searched for. This is what keeps a
              169-tool surface inside a 16k context.
    ``all``   Legacy behaviour: send every merged tool schema on every turn
              (~42k tokens for 169 tools). Only sane with a large context.
    ``off``   No tools.

    A budget guard runs before each upstream call and raises
    ``LLMContextExceeded`` instead of letting llama-server reject the request.
    """
    chosen = (model or "").strip() or resolve_model(token=token)
    msgs: list[dict[str, Any]] = [dict(m) for m in messages]
    chat_fields = _chat_fields(
        chat_id,
        msgs,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        parent_id=parent_id,
        model=chosen,
        user_text=user_text,
    )
    baseline = _estimate_tokens(estimate_messages) if estimate_messages else 0

    mode = _router_mode(tool_router) if use_tools else "off"
    ctx = _context_limit(context_tokens)

    configured = mode != "off" and mcp_service.is_configured()
    all_tools: list[dict[str, Any]] = mcp_service.list_tools() if configured else []
    registry: tool_registry.Registry | None = None
    if configured and mode != "all":
        registry = tool_registry.build(all_tools, mcp_service.tool_servers())

    # L1: tools revealed from the user's own words. L3: tools the model asked
    # for with tool_search/tool_describe. Tracked apart because only a search
    # justifies dropping tool_describe/tool_call from the surface.
    #
    # The revealed set is kept per chat: llama.cpp renders the tool schemas
    # before the messages, so a block that changes between turns invalidates
    # the whole prompt cache. What the chat has already revealed stays in the
    # block; only a tool it has not seen yet changes it.
    sticky, searched_ever = _cached_reveals(chat_id)
    state: dict[str, Any] = {
        # A prior catalog search may be sticky within the chat. Do not carry
        # that schema into a new turn explicitly asking about the local library.
        "revealed": [
            name
            for name in sticky
            if not (
                tool_registry.wants_library(_last_user_text(msgs))
                and name in HIDDEN_ON_LIBRARY_TURN
            )
        ],
        "search_hits": [],
        "searched": searched_ever,
    }
    pre_revealed = _pre_reveal(registry, msgs)
    for name in pre_revealed:
        if name not in state["revealed"]:
            state["revealed"].append(name)
    reveal_limit = _reveal_limit()

    # A request that names the saved library must not be answered by the
    # all-Spotify catalog `search`. Hide that hot tool for this turn, but only
    # when a library schema is already in front of the model (so the turn still
    # has a tool to call if the pre-reveal is disabled).
    library_turn = tool_registry.wants_library(_last_user_text(msgs))
    library_revealed = any(
        tool_registry.is_library_tool(name) for name in state["revealed"]
    )
    hide_catalog_search = library_turn and library_revealed

    def _advertised() -> list[dict[str, Any]]:
        """The tool surface for the next request (L0, L1, L3)."""
        if not configured:
            return []
        if mode == "all":
            return all_tools
        tools = tool_registry.hot_tools(registry) if registry is not None else []
        if hide_catalog_search:
            tools = [
                tool
                for tool in tools
                if (tool.get("function") or {}).get("name")
                not in HIDDEN_ON_LIBRARY_TURN
            ]
        present = {
            (t.get("function") or {}).get("name")
            for t in tools
            if isinstance(t, dict)
        }
        for tool_name in state["revealed"]:
            if hide_catalog_search and tool_name in HIDDEN_ON_LIBRARY_TURN:
                continue
            if tool_name in present:
                continue
            schema = registry.schemas.get(tool_name) if registry is not None else None
            if schema is not None:
                tools.append(schema)
                present.add(tool_name)
        if state["searched"]:
            # L3: the searched schemas are in front of the model, so keep only
            # tool_search (to refine) and drop describe/call indirection.
            tools.append(tool_registry.META_TOOLS[0])
        else:
            # L0/L1: the model may still need to look a tool up, so keep the
            # whole meta surface.
            tools.extend(tool_registry.META_TOOLS)
        return tools

    route: dict[str, Any] = {
        "mode": mode,
        "registered": len(registry) if registry is not None else len(all_tools),
        "advertised": 0,
        "searches": 0,
        "revealed": 0,
        "pre_revealed": len(pre_revealed),
        "carried": len(sticky),
        "reveal_limit": reveal_limit,
        "library_intent": library_turn,
        "catalog_search_hidden": hide_catalog_search,
        "estimated_tokens": 0,
        "text_calls": 0,
    }

    tool_calls_log: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    finish_reason = None
    data: dict[str, Any] = {}
    for round_index in range(max(1, tool_rounds)):
        tools = _advertised()
        route["advertised"] = len(tools)
        route["revealed"] = len(state["search_hits"])
        route["estimated_tokens"] = _guard_context(tools, msgs, ctx, baseline)
        body: dict[str, Any] = {
            "model": chosen,
            "messages": msgs,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        # Only the first request owns the turn. A follow-up request (the model
        # asked for a tool) must not repeat the chat fields: Open WebUI would
        # try to insert the same turn again and reject the assistant reply it
        # already wrote with "409 Message already exists or has an invalid ID".
        # A follow-up also has to carry its own messages, because a saved chat
        # makes Open WebUI replay the stored chain and drop the tool result.
        if chat_fields and round_index == 0:
            body.update(chat_fields)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = _request("POST", "/api/chat/completions", json_body=body, token=token,
                        timeout=timeout)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMUpstreamError("Open WebUI returned no choices.")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish_reason = choice.get("finish_reason")

        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        answer_text = message.get("content") or ""
        if not tool_calls:
            # The model may have written the template's markup as ordinary
            # text. Run it like a native call so the pipeline continues in this
            # turn instead of ending with the markup on screen.
            parsed = _parse_text_tool_calls(answer_text)
            if parsed:
                tool_calls = parsed
                answer_text = _strip_text_tool_calls(answer_text)
                route["text_calls"] = int(route.get("text_calls") or 0) + len(parsed)
                logger.info(
                    "Executing %d textual tool call(s): %s",
                    len(parsed),
                    ", ".join(
                        str((c.get("function") or {}).get("name") or "") for c in parsed
                    ),
                )
        if not tool_calls:
            # Final answer
            _store_reveals(chat_id, state["revealed"], state["searched"])
            return {
                "content": answer_text,
                "reasoning_content": message.get("reasoning_content") or "",
                "finish_reason": finish_reason,
                "model": data.get("model") or chosen,
                "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
                "tool_calls": tool_calls_log,
                "tool_route": route,
            }

        # Record the assistant turn with tool_calls. Arguments are re-serialised
        # from the parsed object: a model that hit its token limit returns a
        # string that is not valid JSON (a URI cut before its closing quote),
        # and Open WebUI parses every replayed call. Forwarding the raw string
        # makes the next request fail with "Failed to parse tool call arguments
        # as JSON" (500), which ended the whole turn.
        prepared: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            call_id = tc.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call_{round_index}_{len(prepared)}"
            args, valid = _normalise_tool_arguments(fn.get("arguments"))
            prepared.append(
                (
                    {
                        "id": call_id,
                        "type": tc.get("type") or "function",
                        "function": {
                            "name": fn.get("name") or "",
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    },
                    args,
                    valid,
                )
            )

        if not prepared:
            # Every entry was unusable; do not loop on an empty call list.
            _store_reveals(chat_id, state["revealed"], state["searched"])
            return {
                "content": answer_text,
                "reasoning_content": message.get("reasoning_content") or "",
                "finish_reason": finish_reason,
                "model": data.get("model") or chosen,
                "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
                "tool_calls": tool_calls_log,
                "tool_route": route,
            }

        msgs.append(
            {
                "role": "assistant",
                "content": answer_text,
                "tool_calls": [call for call, _args, _valid in prepared],
            }
        )
        # Execute each tool
        for call, args, valid in prepared:
            tc_id = call["id"]
            name = call["function"]["name"]
            if not valid:
                # Never run a half-parsed call: a truncated playlist payload
                # would act on the wrong items. Ask for a smaller retry instead.
                result_text = (
                    f"The call to {name} had incomplete JSON arguments (the "
                    "reply was cut off before the closing quote). Nothing was "
                    "run. Retry with fewer items per call, then continue."
                )
                tool_calls_log.append(
                    {
                        "id": tc_id,
                        "name": name,
                        "arguments": {},
                        "result": result_text[:500],
                        "result_full": result_text,
                        "error": "invalid_arguments",
                    }
                )
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_text,
                    }
                )
                continue
            # A model that repeats the exact same call burns rounds without
            # learning anything. Answer from the first result instead, which
            # also keeps the tool/result pairing valid.
            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if signature in seen_calls:
                result_text = (
                    f"Already called {name} with these exact arguments in this "
                    "turn; the result is above. Use a different tool or "
                    "different arguments, or summarise what you have."
                )
            else:
                seen_calls.add(signature)
                result_text = _run_tool(
                    name,
                    args,
                    registry,
                    route,
                    state,
                    reveal_limit,
                )
            tool_calls_log.append(
                {
                    "id": tc_id,
                    "name": name,
                    "arguments": args,
                    # Short form for the phone; the whole answer is stored in
                    # the chat's output items so the next turn keeps it.
                    "result": result_text[:500],
                    "result_full": result_text[:20000],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_text,
                }
            )

    # Exhausted rounds: do one final call without tools to get a summary.
    # The turn was already inserted by the first request, so this follow-up
    # must not repeat the chat fields.
    #
    # The instruction is explicit because a bare "answer now" call made the
    # model narrate what it was *about* to do, or apologise about a step, with
    # no account of the work that had actually finished.
    if tool_calls_log:
        msgs.append(
            {
                "role": "user",
                "content": (
                    "Stop calling tools. Using only the tool results above, "
                    "state what actually happened: each step that ran and what "
                    "it returned, and anything still unfinished. Do not claim a "
                    "step succeeded unless its result says so."
                ),
            }
        )
    route["estimated_tokens"] = _guard_context([], msgs, ctx, baseline)
    body = {
        "model": chosen,
        "messages": msgs,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }
    data = _request("POST", "/api/chat/completions", json_body=body, token=token,
                    timeout=timeout)
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    _store_reveals(chat_id, state["revealed"], state["searched"])
    return {
        "content": _strip_text_tool_calls(message.get("content") or ""),
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "model": data.get("model") or chosen,
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
        "tool_calls": tool_calls_log,
        "tool_route": route,
    }


def status(token: str | None = None) -> dict[str, Any]:
    """Configuration + reachability snapshot; never raises."""
    result: dict[str, Any] = {
        "configured": is_configured(token),
        "available": False,
        "base_url": _base_url(),
        "model": os.getenv("OPENWEBUI_MODEL", "").strip() or None,
        "models": 0,
        "error": None,
        "mcp_configured": mcp_service.is_configured(),
        "mcp_tools": len(mcp_service.list_tools()) if mcp_service.is_configured() else 0,
        "mcp_servers": mcp_service.server_status(),
        "tool_router": _router_mode(),
        "context_tokens": _context_limit(),
        "per_user_auth": bool(token),
    }
    if not result["configured"]:
        result["error"] = "OPENWEBUI_API_KEY is not set."
        return result

    try:
        models = list_models(token=token)
        result["models"] = len(models)
        result["model"] = result["model"] or (models[0]["id"] if models else None)
        result["available"] = bool(models)
        if not models:
            result["error"] = "No model is connected in Open WebUI."
    except LLMCredentialRejected:
        raise
    except LLMError as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Phase 0 helpers: URL fetch + structured JSON extraction for AI URL scan
# ---------------------------------------------------------------------------

import json as _json
import re as _re


def html_to_text(html: str) -> str:
    """Strip scripts/styles/tags and collapse whitespace to plain text."""
    text = html or ""
    text = _re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=_re.S | _re.I)
    text = _re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=_re.S | _re.I)
    text = _re.sub(r"<[^>]+>", " ", text)
    return _re.sub(r"\s+", " ", text).strip()


def fetch_page_text(url: str, *, max_chars: int = 12000) -> str:
    """Fetch a URL and return cleaned plain text (tags stripped, truncated)."""
    if not url or not str(url).strip():
        raise LLMError("url is required")
    try:
        resp = requests.get(
            str(url).strip(),
            headers={"User-Agent": "reSpotIt/1.0 (+https://github.com/damianoshar/reSpotIt)"},
            timeout=(CONNECT_TIMEOUT, 30.0),
        )
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Cannot fetch page: {exc}") from exc
    if resp.status_code >= 400:
        raise LLMUpstreamError(f"Page returned HTTP {resp.status_code}")
    text = html_to_text(resp.text or "")
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _parse_json_object(text: str) -> dict:
    if not text:
        raise LLMUpstreamError("LLM returned empty content.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = _re.sub(r"\n?```$", "", cleaned).strip()
    match = _re.search(r"\{.*\}", cleaned, _re.S)
    if not match:
        match = _re.search(r"\[.*\]", cleaned, _re.S)
    if not match:
        raise LLMUpstreamError("LLM did not return JSON.")
    try:
        data = _json.loads(match.group(0))
    except (ValueError, TypeError) as exc:
        raise LLMUpstreamError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        data = {"items": data}
    return data


_ALBUMS_SYSTEM = (
    "You are a data extractor. Given the text of a web page, extract every album "
    "mentioned (artist + album title). Return ONLY valid JSON with this exact shape: "
    '{"albums": [{"artist": "...", "album": "..."}]}. No prose, no markdown fences. '
    "Use only the album title and its artist: ignore chart ranks, position changes "
    "(\"up 13 from 57th\"), review scores, prices and other table text around them. "
    "Keep names in the page's original language and script. "
    "If none found, return {\"albums\": []}."
)

_TRACKS_SYSTEM = (
    "You are a data extractor. Given the text of a web page, extract every song "
    "title mentioned, with the artist who performs it. Return ONLY valid JSON with "
    'this exact shape: {"tracks": [{"artist": "...", "title": "..."}]}. '
    "No prose, no markdown fences. "
    "Use the song title and the performing artist only: ignore chart ranks, "
    "position changes, review scores, prices, durations and other table text "
    "around them. "
    "Keep names in the page's original language and script. "
    "If none found, return {\"tracks\": []}."
)

_CONCERTS_SYSTEM = (
    "You are a data extractor. Given the text of a web page about concerts or "
    "events, extract every concert. Return ONLY valid JSON with this exact shape: "
    '{"concerts": [["artist", "YYYY-MM-DD or empty", "venue"], ...]}. '
    "Each concert is a 3-item ARRAY, not an object: artist first, then date, "
    "then venue. Never add other keys or a url. Keep the JSON compact. "
    "Rules: 'artist' is the specific performer or band only - never an event "
    "title, festival name, tribute, recital or benefit concert. When an entry "
    "has no named performer (for example 'Concert of Baroque Music' or 'A "
    "concert for Children'), skip that entry entirely: do not copy the event "
    "title into 'artist' and do not cut a name down to a single generic word. "
    "Keep every name in the page's original language and script: do not "
    "translate, transliterate or invent names. 'venue' is the place only, "
    "without the city. Infer the year from the page when it is clear; "
    "otherwise leave the date empty. "
    "No prose, no markdown fences. "
    "If none found, return {\"concerts\": []}."
)


_STRICTER_SUFFIX = (
    "\n\nIMPORTANT: Your previous answer was not valid JSON. "
    "Reply with ONLY a single JSON object matching the requested shape "
    '(for concerts: {"concerts": [["artist", "date", "venue"], ...]}). '
    "No prose, no markdown fences, no extra text."
)

# Nudges for degenerate answers from a small local model. They are appended to
# the *user* message, never to the system prompt: llama.cpp reuses the prompt
# cache by prefix, so rewriting the system prompt would re-evaluate the whole
# page (seconds -> minutes) on every retry.
_CONCERT_NUDGE_EMPTY = (
    "\n\n---\n\nIMPORTANT: this text lists real events with dates. Extract every "
    "one of them as a separate [artist, date, venue] array. Never return an "
    "empty list when dated events are present."
)

_CONCERT_NUDGE_DATES = (
    "\n\n---\n\nIMPORTANT: every event in this text has a date next to it. Put "
    "the date in slot 2 of every array as YYYY-MM-DD. When the year is missing, "
    "infer the nearest upcoming year from the page context. Never leave the "
    "date empty when the text shows one."
)

# The local model sometimes answers {"tracks": []} for a page that clearly
# lists songs (reproduced with short lists such as "Radiohead Weird Fishes.
# Portishead Roads. Daft Punk Around the World."). One retry with an explicit
# nudge recovers them. The nudge must not push the model into inventing songs,
# so it asks the model to *check* the text instead of asserting that it lists
# tracks, and the retried rows are grounded in the page text before they are
# kept (see :func:`_grounded_in`).
#
# Like the concert nudges, it is appended to the *user* message, never to the
# system prompt: llama.cpp reuses the prompt cache by prefix, so rewriting the
# system prompt would re-evaluate the whole page on every retry.
_TRACK_NUDGE_EMPTY = (
    "\n\n---\n\nIMPORTANT: check the text above for song titles. If it lists any, "
    'extract each one as a separate {"artist", "title"} object, one object per '
    "song. Only answer with an empty list when the text really contains no "
    "song titles."
)

# Event pages keep a stale year in the header ("... στη Θεσσαλονίκη 2025") long
# after the listed events rolled over, and a local model copies it. Telling the
# model today's date keeps the year right in the first place; the scan service
# also rolls a slipped year forward after extraction.
def _today_nudge(today: date | None = None) -> str:
    today = today or date.today()
    return (
        f"\n\n---\n\nToday is {today.isoformat()}. A year printed in the page "
        "header is often out of date. When an event's year is missing or has "
        "already passed, use the nearest year that puts the date in the future "
        "(15 November on a page titled 2025, read in 2026, is 2026-11-15). "
        "Never drop an event because its printed year looks past."
    )


# Separators that unambiguously mean "several performers on one bill".
# "&", "and" and "," are deliberately absent: they are part of single artist
# names too (Simon & Garfunkel, Earth, Wind & Fire), and a wrong split would
# resolve each half separately.
_LINEUP_SEPARATORS = (" + ", " / ", " - ", " – ", " | ")


def _split_lineup(artist: str) -> list[str]:
    """Split "A + B" / "A - B" lineups into individual performers."""
    parts = [artist]
    for sep in _LINEUP_SEPARATORS:
        parts = [chunk for part in parts for chunk in part.split(sep)]
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for name in _split_comma_and_lineup(part):
            name = name.strip(" ,;·•")
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            cleaned.append(name)
    return cleaned or [artist]


# A fragment that starts with one of these words continues the previous name
# instead of naming a new performer ("Florence and the Machine").
_CONTINUATION_WORDS = ("the", "a", "an")


def _split_top_level(text: str) -> list[tuple[str, str]]:
    """Split on top-level commas and " and ".

    Returns ``(fragment, separator)`` pairs; the separator is "," or "and"
    for every fragment except the last one. Parenthesised text is never split.
    """
    out: list[tuple[str, str]] = []
    current: list[str] = []
    depth = 0
    index = 0
    lowered = text.lower()
    pending = ""
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if depth == 0 and char == ",":
            out.append(("".join(current), pending))
            current = []
            pending = ","
            index += 1
            continue
        if depth == 0 and lowered.startswith(" and ", index):
            if "".join(current).strip():
                out.append(("".join(current), pending))
                current = []
                pending = "and"
            # An oxford list ("A, B, and C") leaves the "and" in front of the
            # last name: drop it instead of emitting an empty fragment.
            index += 5
            continue
        current.append(char)
        index += 1
    out.append(("".join(current), pending))
    return out


def _split_comma_and_lineup(artist: str) -> list[str]:
    """Split "A, B, and C" / "A and B" into individual performers.

    Conservative on purpose: an ampersand, a lowercase article and a
    "Tyler, The Creator"-style comma both mean the whole string is one
    artist name, so the split is refused and the name is kept intact.
    """
    if "&" in artist:
        # "Simon & Garfunkel", "Earth, Wind & Fire" are single artist names.
        return [artist]
    parts = _split_top_level(artist)
    if len(parts) < 2:
        return [artist]
    cleaned: list[str] = []
    for index, (fragment, separator) in enumerate(parts):
        name = fragment.strip(" ,;·•")
        if not name or len(name) < 2:
            return [artist]
        if index > 0:
            first = name.split()[0]
            word = first.strip(".,").lower()
            if first.islower() and word in _CONTINUATION_WORDS:
                # "Florence and the Machine" continues the previous name.
                return [artist]
            if separator == "," and word in _CONTINUATION_WORDS:
                # "Tyler, The Creator" is one artist.
                return [artist]
        cleaned.append(name)
    return cleaned


def _strip_venue_suffix(artist: str, venue: str) -> str:
    """Drop a venue the model glued onto the artist ("Kiya Tabassian Westerkerk")."""
    if not artist or not venue or len(venue) < 4:
        return artist
    if artist.casefold().endswith(venue.casefold()):
        trimmed = artist[: len(artist) - len(venue)].strip(" ,;·-")
        if trimmed:
            return trimmed
    return artist


def extract_structured(system: str, user_text: str, *, token=None, max_tokens: int = 1024,
                       timeout: tuple[float, float] | float | None = None) -> dict:
    """Run a chat with tools disabled and parse the reply as JSON.

    Retries once with a stricter prompt if the first attempt does not yield
    parseable JSON (per the plan's edge case). On the second failure the raw
    content is included in the error so the caller can surface it.

    ``timeout`` overrides the request read timeout; background callers (the
    concert scan) pass a long one because a local model can take minutes.
    """
    result = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        temperature=0.1,
        max_tokens=max_tokens,
        use_tools=False,
        token=token,
        timeout=timeout,
    )
    content = result.get("content") or ""
    try:
        return _parse_json_object(content)
    except LLMUpstreamError:
        pass

    # Retry once with a stricter prompt.
    retry_result = chat(
        [{"role": "system", "content": system + _STRICTER_SUFFIX},
         {"role": "user", "content": user_text}],
        temperature=0.0,
        max_tokens=max_tokens,
        use_tools=False,
        token=token,
        timeout=timeout,
    )
    retry_content = retry_result.get("content") or ""
    try:
        return _parse_json_object(retry_content)
    except LLMUpstreamError as exc:
        raise LLMUpstreamError(
            f"LLM did not return valid JSON: {exc}. "
            f"Raw response: {retry_content[:500]!r}"
        ) from exc


# Album extraction runs over long WebView snapshots, so it is chunked like the
# concert scan: overlapping windows keep the model inside its context limit.
ALBUM_CHUNK_CHARS = int(os.getenv("ALBUM_SCAN_CHUNK_CHARS", "24000"))
ALBUM_CHUNK_OVERLAP = int(os.getenv("ALBUM_SCAN_CHUNK_OVERLAP", "1000"))


def _split_for_extraction(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping windows so a long page never exceeds the model context."""
    text = text or ""
    if size <= 0 or len(text) <= size:
        return [text] if text else []
    step = max(1, size - max(0, overlap))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def extract_albums(text: str, *, token=None) -> dict:
    """Extract (artist, album) pairs, chunking long pages like the concert scan.

    A WebView snapshot of an album list can be tens of thousands of characters;
    one call would silently truncate it at the model's context limit, so long
    text is extracted chunk by chunk and merged.
    """
    chunks = _split_for_extraction(text, ALBUM_CHUNK_CHARS, ALBUM_CHUNK_OVERLAP)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, chunk in enumerate(chunks):
        try:
            data = extract_structured(_ALBUMS_SYSTEM, chunk, token=token, max_tokens=4096)
        except LLMError:
            if len(chunks) == 1:
                raise
            logger.warning("Album scan chunk %d failed; skipping it", index)
            continue
        albums = data.get("albums") if isinstance(data.get("albums"), list) else []
        for a in albums:
            if not isinstance(a, dict):
                continue
            artist = str(a.get("artist") or "").strip()
            album = str(a.get("album") or "").strip()
            key = (artist.casefold(), album.casefold())
            if not (artist or album) or key in seen:
                continue
            seen.add(key)
            out.append({"artist": artist, "album": album})
    return {"albums": out}


def _norm_words(text: str) -> str:
    """Lowercase text with everything but letters and digits turned into spaces."""
    return _re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).strip()


def _grounded_in(text: str, title: str) -> bool:
    """True when the title really comes from the page text.

    Guards the nudge retry: the retry must not be able to put a song on the
    user's playlist that the page never mentioned. The check is deliberately
    loose - the title itself, or its longest word, has to appear in the page -
    so a slightly reworded title ("Weird Fishes / Arpeggi" -> "Weird Fishes")
    still passes.
    """
    norm_text = _norm_words(text)
    norm_title = _norm_words(title)
    if not norm_title or not norm_text:
        return False
    if norm_title in norm_text:
        return True
    words = [w for w in norm_title.split() if len(w) >= 5]
    return bool(words) and any(w in norm_text for w in words)


def _track_rows(data: dict) -> list[dict]:
    """The (artist, title) rows of an extraction reply, without empty ones."""
    rows = data.get("tracks") if isinstance(data, dict) else None
    out: list[dict] = []
    for t in rows if isinstance(rows, list) else []:
        if not isinstance(t, dict):
            continue
        artist = str(t.get("artist") or "").strip()
        title = str(t.get("title") or "").strip()
        if not (artist or title):
            continue
        out.append({"artist": artist, "title": title})
    return out


def extract_tracks(text: str, *, token=None) -> dict:
    """Extract (artist, title) track pairs, chunking long pages like the album scan.

    The track counterpart of :func:`extract_albums`. A rendered tracklist page
    can be tens of thousands of characters, so long text is extracted chunk by
    chunk and merged.

    A chunk that comes back empty is asked once more with a nudge: the local
    model does sometimes answer ``{"tracks": []}`` for a page that clearly
    lists songs, and a silent empty list would look like "no tracks found".
    """
    chunks = _split_for_extraction(text, ALBUM_CHUNK_CHARS, ALBUM_CHUNK_OVERLAP)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, chunk in enumerate(chunks):
        try:
            data = extract_structured(_TRACKS_SYSTEM, chunk, token=token, max_tokens=4096)
        except LLMError:
            if len(chunks) == 1:
                raise
            logger.warning("Track scan chunk %d failed; skipping it", index)
            continue
        rows = _track_rows(data)
        if not rows:
            # The local model occasionally answers {"tracks": []} for a page
            # full of song titles (seen finishing in ~1 second). Ask once more
            # before reporting "no tracks".
            logger.warning("Track scan chunk %d returned no tracks; retrying", index)
            try:
                data = extract_structured(
                    _TRACKS_SYSTEM,
                    chunk + _TRACK_NUDGE_EMPTY,
                    token=token,
                    max_tokens=4096,
                )
            except LLMError:
                data = {}
            # The retry may only keep songs the page really mentions.
            rows = [r for r in _track_rows(data) if _grounded_in(chunk, r["title"])]
        for row in rows:
            artist = row["artist"]
            title = row["title"]
            key = (artist.casefold(), title.casefold())
            if key in seen:
                continue
            seen.add(key)
            out.append({"artist": artist, "title": title})
    return {"tracks": out}


def extract_concerts(text: str, *, token=None, timeout=None, mode: str | None = None) -> dict:
    """Extract concerts from page text.

    ``mode`` is a retry hint used by the background scan: ``"empty"`` when the
    model returned no rows for a page that clearly lists events, ``"dates"``
    when it returned rows without any date.
    """
    system = _CONCERTS_SYSTEM
    # The nudges go on the user message so the cached system+page prefix
    # survives the retry (llama.cpp reuses the prompt cache by prefix). The
    # page text stays first, so the prefix is still cacheable.
    prompt = text + _today_nudge()
    if mode == "empty":
        prompt += _CONCERT_NUDGE_EMPTY
    elif mode == "dates":
        prompt += _CONCERT_NUDGE_DATES
    data = extract_structured(system, prompt, token=token, max_tokens=4096,
                              timeout=timeout)
    concerts = data.get("concerts") if isinstance(data.get("concerts"), list) else []
    out = []
    for c in concerts:
        if isinstance(c, dict):
            # Legacy object shape (and anything the model improvises).
            row = {
                "artist": str(c.get("artist") or "").strip(),
                "date": str(c.get("date") or "").strip(),
                "venue": str(c.get("venue") or "").strip(),
                "url": str(c.get("url") or "").strip(),
            }
        elif isinstance(c, (list, tuple)):
            # Compact [artist, date, venue] row: far fewer generated tokens
            # (~12 vs ~30 per concert), which is the scan's real bottleneck.
            vals = [str(v).strip() if v is not None else "" for v in c]
            vals += [""] * (3 - len(vals))
            row = {
                "artist": vals[0],
                "date": vals[1],
                "venue": vals[2],
                "url": "",
            }
        else:
            continue
        # A lineup ("A + B", "A - B") is one event with several performers;
        # each performer becomes its own row so it can correlate on its own.
        artist = _strip_venue_suffix(row["artist"], row["venue"])
        for artist in _split_lineup(artist):
            out.append({**row, "artist": artist})
    return {"concerts": out}
