"""
services/auth_service.py
-------------------------
Single responsibility: Spotify OAuth token lifecycle.
Dialect-agnostic: works on both Postgres and SQLite.
"""

import base64
import hashlib
import json
import os
import secrets
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy.orm import Session

from ..database import SessionLocal, is_sqlite
from ..models import SpotifyToken, OAuthState, User
from ..scoping import as_uuid

CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
REDIRECT_URI = (
    (f"{PUBLIC_BASE_URL}/api/v1/auth/callback" if PUBLIC_BASE_URL else "")
    or os.getenv("REDIRECT_URI")
    or "http://127.0.0.1:8888/api/v1/auth/callback"
)
DATABASE_URL  = os.environ.get("DATABASE_URL", "")
MIGRATIONS_DIR = os.getenv("MIGRATIONS_DIR", "/app/migrations")
LEGACY_TOKEN_FILE = "/data/.spotify_token.json"

SCOPE = (
    "user-library-read "
    "user-library-modify "
    "user-follow-read "
    "user-follow-modify "
    "user-read-private "
    "user-read-email "
    "playlist-modify-public "
    "playlist-modify-private "
    "playlist-read-private "
    "playlist-read-collaborative "
    "user-modify-playback-state "
    "user-read-playback-state "
    "user-read-currently-playing "
    "streaming"
)


def _pkce_verifier() -> str:
    """A PKCE code verifier (RFC 7636 wants 43-128 unreserved characters)."""
    return secrets.token_urlsafe(64)


def _pkce_challenge(verifier: str) -> str:
    """The S256 code challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class OAuthGrant:
    """What one authorization request carries to the callback."""

    user_id: str
    code_verifier: str | None = None


def _store_oauth_state(user_id: str) -> tuple[str, str]:
    """Store a state and its PKCE verifier; return both."""
    _ensure_schema()
    owner = str(as_uuid(user_id))
    state = secrets.token_urlsafe(24)
    verifier = _pkce_verifier()
    with SessionLocal() as db:
        db.add(OAuthState(state=state, user_id=owner, code_verifier=verifier))
        # purge old states
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
        db.query(OAuthState).filter(OAuthState.created_at < cutoff).delete()
        db.commit()
    return state, verifier


def create_oauth_state(user_id: str) -> str:
    """The CSRF state for one authorization request."""
    state, _verifier = _store_oauth_state(user_id)
    return state


def _stored_verifier(state: str) -> str | None:
    """The PKCE verifier stored with a state, if it is still valid."""
    if not state:
        return None
    _ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
    with SessionLocal() as db:
        row = db.query(OAuthState).filter(
            OAuthState.state == state,
            OAuthState.created_at > cutoff,
        ).first()
        if not row or not row.code_verifier:
            return None
        return str(row.code_verifier)


def build_auth_url(user_id: str, state: str | None = None) -> str:
    """The URL the browser is sent to.

    A deployment with a client secret is a confidential client: it proves the
    exchange with the secret and the authorize URL stays plain. A deployment
    without one is a public client and proves the exchange with PKCE, the same
    way the phone does. Spotify rejects an empty client_secret with
    "invalid_client", so the secret is only ever sent when it is set.
    """
    if state is None:
        state, verifier = _store_oauth_state(user_id)
    else:
        verifier = _stored_verifier(state) or ""
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPE,
        "state":         state,
    }
    if not CLIENT_SECRET and verifier:
        params["code_challenge"] = _pkce_challenge(verifier)
        params["code_challenge_method"] = "S256"
    return "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)


def _ensure_schema():
    if is_sqlite():
        from .db_service import _ensure_sqlite_schema
        _ensure_sqlite_schema()
        return
    from .db_service import run_migrations, wait_for_db
    import psycopg2
    wait_for_db(DATABASE_URL)
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    try:
        run_migrations(conn, MIGRATIONS_DIR)
    finally:
        conn.close()


def validate_oauth_state(state: str | None) -> OAuthGrant | None:
    """Consume a state and return what the callback needs.

    The row is deleted, so a state can be used once. The PKCE verifier comes
    back with it: the callback needs it to exchange the code.
    """
    if not state:
        return None
    _ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
    with SessionLocal() as db:
        row = db.query(OAuthState).filter(
            OAuthState.state == state,
            OAuthState.created_at > cutoff,
        ).first()
        if not row:
            return None
        user_id = row.user_id
        verifier = row.code_verifier
        db.delete(row)
        db.commit()
    return OAuthGrant(
        user_id=str(user_id),
        code_verifier=str(verifier) if verifier else None,
    )


def exchange_code(code: str, code_verifier: str | None = None) -> dict:
    """Trade the authorization code for tokens.

    A deployment with a client secret proves itself with the secret; one
    without proves itself with the PKCE verifier. The secret is never sent
    empty: Spotify answers that with "invalid_client", which is what broke
    the link when the deployment had no secret of its own.
    """
    data = {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     CLIENT_ID,
    }
    if CLIENT_SECRET:
        data["client_secret"] = CLIENT_SECRET
    elif code_verifier:
        data["code_verifier"] = code_verifier
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data=data,
    )
    resp.raise_for_status()
    return resp.json()


def save_token(tok: dict, user_id: str) -> None:
    _ensure_schema()
    owner = str(as_uuid(user_id))
    expires_at = _expires_at(tok)
    with SessionLocal() as db:
        existing = db.query(SpotifyToken).filter(SpotifyToken.user_id == owner).first()
        if existing:
            existing.access_token = tok["access_token"]
            existing.refresh_token = tok.get("refresh_token") or existing.refresh_token
            existing.expires_at = expires_at
            existing.updated_at = datetime.now(timezone.utc)
        else:
            db.add(SpotifyToken(
                user_id=owner,
                access_token=tok["access_token"],
                refresh_token=tok.get("refresh_token"),
                expires_at=expires_at,
            ))
        db.commit()


def load_token(user_id: str) -> dict | None:
    _ensure_schema()
    owner = str(as_uuid(user_id))
    with SessionLocal() as db:
        row = db.query(SpotifyToken).filter(SpotifyToken.user_id == owner).first()
    if not row:
        return None
    tok = {"access_token": row.access_token, "refresh_token": row.refresh_token}
    if row.expires_at:
        tok["expires_at"] = row.expires_at.isoformat() if hasattr(row.expires_at, "isoformat") else str(row.expires_at)
    return tok


def delete_token(user_id: str) -> bool:
    _ensure_schema()
    owner = str(as_uuid(user_id))
    with SessionLocal() as db:
        n = db.query(SpotifyToken).filter(SpotifyToken.user_id == owner).delete()
        db.commit()
    return n > 0


def _refresh(refresh_tok: str) -> tuple[str, str, int | None]:
    # PKCE (public-client) tokens refresh with client_id only. Sending an
    # empty client_secret is rejected by Spotify, so include it only when set.
    data = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_tok,
        "client_id":     CLIENT_ID,
    }
    if CLIENT_SECRET:
        data["client_secret"] = CLIENT_SECRET
    resp = requests.post("https://accounts.spotify.com/api/token", data=data)
    resp.raise_for_status()
    data = resp.json()
    return (
        data["access_token"],
        data.get("refresh_token", refresh_tok),
        data.get("expires_in"),
    )


def get_valid_token(user_id: str) -> str:
    _migrate_legacy_file(user_id)
    tok = load_token(user_id)
    if not tok:
        raise RuntimeError("Spotify is not linked for this account.")
    if not _is_expired(tok):
        return tok["access_token"]
    access, new_refresh, expires_in = _refresh(tok["refresh_token"])
    tok["access_token"]  = access
    tok["refresh_token"] = new_refresh
    if expires_in is not None:
        tok["expires_in"] = expires_in
    tok.pop("expires_at", None)
    save_token(tok, user_id)
    return access


def is_authenticated(user_id: str) -> bool:
    try:
        get_valid_token(user_id)
        return True
    except Exception:
        return False


_schema_ready = False


def _ensure_sqlite_schema():
    global _schema_ready
    if _schema_ready:
        return
    from .db_service import _ensure_sqlite_schema as _ess
    _ess()
    _schema_ready = True


def _migrate_legacy_file(user_id: str):
    import pathlib
    legacy = pathlib.Path(LEGACY_TOKEN_FILE)
    if not legacy.exists():
        return
    try:
        tok = json.loads(legacy.read_text())
        if tok.get("access_token") and tok.get("refresh_token"):
            if load_token(user_id) is None:
                save_token(tok, user_id)
        legacy.unlink()
    except Exception:
        pass


def _expires_at(tok: dict):
    expires_in = tok.get("expires_in")
    if expires_in is None:
        existing = tok.get("expires_at")
        if not existing:
            return None
        try:
            return datetime.fromisoformat(str(existing).replace("Z", "+00:00"))
        except ValueError:
            return None
    try:
        return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    except (ValueError, TypeError):
        return None


def _is_expired(tok: dict) -> bool:
    exp = _expires_at(tok)
    if exp is None:
        return True
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp <= datetime.now(timezone.utc) + timedelta(seconds=60)