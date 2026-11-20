"""routers/llm.py - local LLM access through Open WebUI.

  GET    /api/v1/llm/status          - configuration + reachability
  GET    /api/v1/llm/models          - models Open WebUI exposes
  POST   /api/v1/llm/chat            - non-streaming chat completion
  GET    /api/v1/llm/chats           - the user's Open WebUI chats
  GET    /api/v1/llm/chats/current   - the chat this user is currently in
  POST   /api/v1/llm/chats           - create a chat
  GET    /api/v1/llm/chats/{id}      - one chat with its messages
  DELETE /api/v1/llm/chats/{id}      - delete a chat
  POST   /api/v1/llm/extract-albums  - album scan (also logs the scan chat)
  POST   /api/v1/llm/extract-tracks  - track scan (also logs the scan chat)

Additive: the frozen library contract is untouched. Every route needs the
same server JWT (or a Keycloak access token) as the rest of /api/v1; the
Open WebUI credential stays server-side.

Phase 4: when OPENWEBUI_TRUSTED_AUTH is enabled, the API exchanges the
signed-in user's email for a per-user Open WebUI JWT, so chat activity is
attributed to that user instead of the shared service key.

Phase 5/6: chats are stored in Open WebUI under that same per-user identity.
``POST /chat`` can continue a chat (``chat_id``), prepends its last N turns
(``use_history`` / ``history_limit``) and appends the new exchange
(``persist``). The pointer to the current chat is kept locally in
``llm_chat_sessions``.

Scans are logged: ``extract-albums`` and ``extract-tracks`` write one turn
into a fresh Open WebUI chat (``chat_type=extract_albums`` / ``extract_tracks``)
and answer with the ``chat_id`` next to the extracted rows, so the scan shows
up in the user's chat history instead of vanishing.
"""
import logging
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..contract import ContractError
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..services import chat_session_service, llm_service, openwebui_service, scan_chat_service

logger = logging.getLogger("llm")

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

MAX_MESSAGE_CHARS = 20_000
MAX_MESSAGES = 40
MAX_CHAT_ID_CHARS = 128


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class LLMChatBody(BaseModel):
    messages: list[LLMMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    model: str | None = Field(default=None, max_length=512)
    system: str | None = Field(default=None, max_length=MAX_MESSAGE_CHARS)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    use_tools: bool = Field(default=True)
    # Tool routing: "auto" advertises a small searchable surface (default),
    # "all" sends every MCP schema (large prompt), "off" sends none.
    tool_router: Literal["auto", "all", "off"] | None = Field(default=None)
    # Phase 5/6: continue a stored chat instead of starting from scratch.
    chat_id: str | None = Field(default=None, max_length=MAX_CHAT_ID_CHARS)
    use_history: bool = Field(default=True)
    history_limit: int = Field(default=20, ge=0, le=MAX_MESSAGES)
    persist: bool = Field(default=True)
    # Chat type for filtering (assistant, extract_albums, extract_concerts, etc.)
    chat_type: str | None = Field(default=None, max_length=64)


class LLMChatCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=512)
    # Chat type for filtering (assistant, extract_albums, extract_concerts, etc.)
    chat_type: str | None = Field(default=None, max_length=64)


class ExtractBody(BaseModel):
    url: str | None = Field(default=None, max_length=2048)
    text: str | None = Field(default=None, max_length=20000)


def _contract_error(exc: llm_service.LLMError) -> ContractError:
    return ContractError(exc.status_code, exc.code, str(exc))


def _chat_error(exc: openwebui_service.OpenWebUIError) -> ContractError:
    if isinstance(exc, openwebui_service.OpenWebUIChatNotFound):
        return ContractError(404, "llm_chat_not_found", str(exc))
    return ContractError(502, "llm_chat_error", str(exc))


def _status_for_user(user: User) -> dict:
    token = openwebui_service.get_user_token(user)
    try:
        return llm_service.status(token=token)
    except llm_service.LLMCredentialRejected:
        if not token:
            raise
        openwebui_service.invalidate(str(user.id))
        token = openwebui_service.get_user_token(user)
        return llm_service.status(token=token)


async def _with_user_token(user: User, fn, *args, **kwargs):
    """Run an Open WebUI LLM call as the signed-in user, retrying once on 401.

    A recreated Open WebUI container can invalidate cached JWTs; the retry
    performs a fresh trusted-header sign-in before giving up.
    """
    token = await run_in_threadpool(openwebui_service.get_user_token, user)
    try:
        return await run_in_threadpool(fn, *args, token=token, **kwargs)
    except llm_service.LLMCredentialRejected:
        if not token:
            raise
        openwebui_service.invalidate(str(user.id))
        token = await run_in_threadpool(openwebui_service.get_user_token, user)
        return await run_in_threadpool(fn, *args, token=token, **kwargs)


async def _chat_token(user: User) -> str:
    token = await run_in_threadpool(openwebui_service.get_user_token, user)
    if not token:
        raise ContractError(
            503,
            "llm_chat_unavailable",
            "Chat history requires OPENWEBUI_TRUSTED_AUTH=true and a "
            "reachable Open WebUI.",
        )
    return token


async def _with_chat_token(user: User, fn, *args, **kwargs):
    """Run a chat-history call with the per-user token (token is first arg)."""
    token = await _chat_token(user)
    try:
        return await run_in_threadpool(fn, token, *args, **kwargs)
    except openwebui_service.OpenWebUICredentialRejected:
        openwebui_service.invalidate(str(user.id))
        token = await _chat_token(user)
        return await run_in_threadpool(fn, token, *args, **kwargs)


@router.get("/status", summary="LLM configuration and reachability")
def status(user: User = Depends(get_current_user)):
    data = _status_for_user(user)
    data["chat_history"] = openwebui_service.trusted_auth_enabled()
    return {
        "data": data,
        "identity": openwebui_service.status(),
    }


@router.get("/models", summary="Models exposed by Open WebUI")
async def models(user: User = Depends(get_current_user)):
    try:
        return {"data": await _with_user_token(user, llm_service.list_models)}
    except llm_service.LLMError as exc:
        raise _contract_error(exc) from exc


@router.get("/chats", summary="The signed-in user's Open WebUI chats")
async def list_chats(
    user: User = Depends(get_current_user),
    chat_type: str | None = None,
):
    try:
        rows = await _with_chat_token(user, openwebui_service.list_chats, chat_type=chat_type)
        return {"data": rows}
    except openwebui_service.OpenWebUIError as exc:
        raise _chat_error(exc) from exc


@router.get("/chats/current", summary="The chat this user is currently in")
async def current_chat(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat_id = chat_session_service.get_current_chat_id(db, str(user.id))
    if not chat_id:
        return {"data": None}
    try:
        row = await _with_chat_token(user, openwebui_service.get_chat, chat_id)
        return {"data": row}
    except openwebui_service.OpenWebUIChatNotFound:
        chat_session_service.clear_current_chat_id(db, str(user.id), chat_id)
        return {"data": None}
    except openwebui_service.OpenWebUIError as exc:
        raise _chat_error(exc) from exc


@router.delete("/chats/current", summary="Forget the current chat without deleting it")
async def clear_current_chat(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat_session_service.clear_current_chat_id(db, str(user.id))
    return {"data": {"cleared": True}}


@router.post("/chats", summary="Create a new Open WebUI chat")
async def create_chat(
    body: LLMChatCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    title = (body.title or "").strip() or "New Chat"
    try:
        row = await _with_chat_token(
            user, openwebui_service.create_chat, title=title, model=body.model, chat_type=body.chat_type
        )
    except openwebui_service.OpenWebUIError as exc:
        raise _chat_error(exc) from exc
    chat_session_service.set_current_chat_id(db, str(user.id), row["id"])
    return {"data": row}


@router.get("/chats/{chat_id}", summary="One Open WebUI chat with messages")
async def get_chat(chat_id: str, user: User = Depends(get_current_user)):
    try:
        row = await _with_chat_token(user, openwebui_service.get_chat, chat_id)
        return {"data": row}
    except openwebui_service.OpenWebUIError as exc:
        raise _chat_error(exc) from exc


@router.delete("/chats/{chat_id}", summary="Delete an Open WebUI chat")
async def delete_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        await _with_chat_token(user, openwebui_service.delete_chat, chat_id)
    except openwebui_service.OpenWebUIError as exc:
        raise _chat_error(exc) from exc
    chat_session_service.clear_current_chat_id(db, str(user.id), chat_id)
    return {"data": {"deleted": True, "chat_id": chat_id}}


@router.post("/extract-albums", summary="Extract albums from a page using AI")
async def extract_albums(body: ExtractBody, user: User = Depends(get_current_user)):
    text = ""
    if body.url:
        try:
            text = await run_in_threadpool(llm_service.fetch_page_text, body.url, max_chars=6000)
        except llm_service.LLMError as exc:
            raise _contract_error(exc) from exc
    if body.text:
        text = (text + "\n" + body.text).strip() if text else body.text.strip()
    if not text:
        raise ContractError(422, "llm_empty_input", "Provide url or text.")
    try:
        data = await _with_user_token(user, llm_service.extract_albums, text)
    except llm_service.LLMError as exc:
        raise _contract_error(exc) from exc
    # The scan is logged in the user's Open WebUI chats so the request and the
    # answer stay visible (best-effort: a missing chat never fails the scan).
    chat_id = scan_chat_service.record_scan(
        user,
        kind=scan_chat_service.ALBUM_KIND,
        source=(body.url or "").strip() or None,
        rows=data.get("albums") or [],
    )
    return {"data": data, "chat_id": chat_id}


@router.post("/extract-tracks", summary="Extract tracks from a page using AI")
async def extract_tracks(body: ExtractBody, user: User = Depends(get_current_user)):
    """The track counterpart of /extract-albums.

    Additive route: the frozen library contract is untouched. The route only
    reads the page and asks the model for the track list; it never writes
    anything to Spotify (UI policy P9).
    """
    text = ""
    if body.url:
        try:
            text = await run_in_threadpool(llm_service.fetch_page_text, body.url, max_chars=6000)
        except llm_service.LLMError as exc:
            raise _contract_error(exc) from exc
    if body.text:
        text = (text + "\n" + body.text).strip() if text else body.text.strip()
    if not text:
        raise ContractError(422, "llm_empty_input", "Provide url or text.")
    try:
        data = await _with_user_token(user, llm_service.extract_tracks, text)
    except llm_service.LLMError as exc:
        raise _contract_error(exc) from exc
    chat_id = scan_chat_service.record_scan(
        user,
        kind=scan_chat_service.TRACK_KIND,
        source=(body.url or "").strip() or None,
        rows=data.get("tracks") or [],
    )
    return {"data": data, "chat_id": chat_id}


@router.post("/extract-concerts", summary="Extract concerts from a page using AI")
async def extract_concerts(body: ExtractBody, user: User = Depends(get_current_user)):
    text = ""
    if body.url:
        try:
            text = await run_in_threadpool(llm_service.fetch_page_text, body.url, max_chars=6000)
        except llm_service.LLMError as exc:
            raise _contract_error(exc) from exc
    if body.text:
        text = (text + "\n" + body.text).strip() if text else body.text.strip()
    if not text:
        raise ContractError(422, "llm_empty_input", "Provide url or text.")
    try:
        data = await _with_user_token(user, llm_service.extract_concerts, text)
    except llm_service.LLMError as exc:
        raise _contract_error(exc) from exc
    return {"data": data}


@router.post("/chat", summary="Chat with the local model")
async def chat(
    body: LLMChatBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    messages: list[dict[str, str]] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})

    chat_id = body.chat_id
    parent_id: str | None = None
    stored_chain: list[dict[str, Any]] = []

    # A request that names a saved chat is a turn of that chat: Open WebUI
    # writes the turn itself and needs both message ids up front. The same ids
    # are used to finish the turn, and the user message id is what native
    # context compaction anchors its checkpoint on.
    managed_turn = bool(body.persist and chat_id)
    user_message_id = str(uuid.uuid4()) if managed_turn else None
    assistant_message_id = str(uuid.uuid4()) if managed_turn else None

    # Phase 6: load the stored conversation before the new turn.
    if chat_id:
        try:
            row = await _with_chat_token(user, openwebui_service.get_chat, chat_id)
        except openwebui_service.OpenWebUIError as exc:
            raise _chat_error(exc) from exc
        # The chain Open WebUI replays for this chat, with its compaction
        # checkpoint applied. The API sends the same view, so a tool-loop
        # follow-up (which cannot use the saved-chat replay) keeps the context
        # the first request had.
        stored_chain = openwebui_service.compacted_chain(row)
        if body.use_history and body.history_limit > 0:
            messages.extend(
                stored_chain if managed_turn else stored_chain[-body.history_limit:]
            )
        parent_id = openwebui_service.current_message_id(row)

    messages.extend(message.model_dump() for message in body.messages)

    user_text = ""
    if body.persist:
        # A saved chat stores one user message per turn (that is what Open WebUI
        # replays), so the turn's user messages are stored as one.
        user_text = "\n\n".join(
            m.content.strip() for m in body.messages if m.role == "user" and m.content.strip()
        )

    try:
        result = await _with_user_token(
            user,
            llm_service.chat,
            messages,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            use_tools=body.use_tools,
            tool_router=body.tool_router,
            chat_id=chat_id if managed_turn else None,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            parent_id=parent_id,
            user_text=user_text or None,
            estimate_messages=stored_chain or None,
        )
    except llm_service.LLMError as exc:
        raise _contract_error(exc) from exc

    persisted = False
    if body.persist:
        try:
            if not managed_turn:
                row = await _with_chat_token(
                    user,
                    openwebui_service.create_chat,
                    title=user_text[:80] or "New Chat",
                    model=result.get("model"),
                    chat_type=body.chat_type,
                )
                chat_id = row["id"]
                await _with_chat_token(
                    user,
                    openwebui_service.append_exchange,
                    chat_id,
                    user_text,
                    str(result.get("content") or ""),
                    model=result.get("model"),
                    title=user_text[:80] or None,
                    tool_calls=result.get("tool_calls") or None,
                )
            else:
                await _with_chat_token(
                    user,
                    openwebui_service.complete_turn,
                    chat_id,
                    str(user_message_id),
                    str(assistant_message_id),
                    user_text,
                    str(result.get("content") or ""),
                    model=result.get("model"),
                    parent_id=parent_id,
                    tool_calls=result.get("tool_calls") or None,
                )
            persisted = True
            chat_session_service.set_current_chat_id(db, str(user.id), chat_id)
        except openwebui_service.OpenWebUIError as exc:
            # The answer is still useful; report that it was not stored.
            logger.warning("Could not persist chat turn: %s", exc)

    # The full tool answers exist only for the stored chain: they are already
    # in the chat's output items above. The response keeps the short ``result``
    # so a turn does not return tens of kilobytes to the phone.
    for entry in result.get("tool_calls") or []:
        if isinstance(entry, dict):
            entry.pop("result_full", None)

    payload = dict(result)
    payload["chat_id"] = chat_id
    payload["persisted"] = persisted
    return {"data": payload}
