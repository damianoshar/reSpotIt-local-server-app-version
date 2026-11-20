"""routers/auth.py — Spotify OAuth flow."""
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user, get_optional_user
from ..models import User
from ..schemas import UserRegister, UserLogin, TokenRefresh, SpotifyTokenImport
from ..services.auth_service import (
    build_auth_url, exchange_code, save_token, delete_token,
    get_valid_token, is_authenticated, validate_oauth_state,
)
from ..services import mcp_grant_service, user_auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_SUCCESS_HTML = """<!DOCTYPE html>
<html>
<head><title>Spotify Auth</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0c0b09; color: #ede8df;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: #1b1916; border: 1px solid #2e2a22; border-radius: 10px;
          padding: 2.5rem 3rem; text-align: center; }
  h2 { margin: 1rem 0 0.5rem; } p { color: #9a9080; font-size: 0.9rem; }
</style></head>
<body><div class="card">
  <div style="font-size:2.5rem">✓</div>
  <h2>Signed in to Spotify</h2>
  <p>You can close this window.</p>
</div>
<script>
  if (window.opener) {
    window.opener.postMessage({ type: 'SPOTIFY_AUTH_SUCCESS' }, '*');
    setTimeout(() => window.close(), 1200);
  }
</script></body></html>"""

_ERROR_HTML = """<!DOCTYPE html>
<html>
<head><title>Auth Error</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0c0b09; color: #ede8df;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
  .card {{ background: #1b1916; border: 1px solid #5a2a2a; border-radius: 10px;
          padding: 2.5rem 3rem; text-align: center; }}
  h2 {{ color: #c85a4a; }} p {{ color: #9a9080; font-size: 0.85rem; }}
</style></head>
<body><div class="card">
  <div style="font-size:2.5rem">✕</div>
  <h2>Authorization failed</h2>
  <p>{reason}</p>
</div></body></html>"""


@router.post("/register", summary="Register a user account")
def register(body: UserRegister, db: Session = Depends(get_db)):
    try:
        user = user_auth_service.register_user(db, body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return user_auth_service.issue_tokens(user)


@router.post("/login", summary="Log in with email and password")
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = user_auth_service.authenticate_user(db, body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return user_auth_service.issue_tokens(user)


@router.post("/refresh", summary="Refresh an access token")
def refresh(body: TokenRefresh):
    try:
        payload = user_auth_service.decode_token(body.refresh_token, "refresh")
    except Exception:
        raise HTTPException(status_code=401, detail="invalid refresh token")
    access = user_auth_service.create_access_token(payload["sub"], payload["email"])
    return {"access_token": access, "token_type": "bearer"}


@router.get("/status", summary="Check user authentication status")
def status(user: User | None = Depends(get_optional_user)):
    if user is None:
        return {"authenticated": False, "user_id": None, "email": None}
    return {"authenticated": True, "user_id": str(user.id), "email": user.email}


@router.get("/me", summary="Current user")
def me(user: User = Depends(get_current_user)):
    return {"id": str(user.id), "email": user.email, "created_at": user.created_at.isoformat()}


@router.get("/callback", response_class=HTMLResponse, include_in_schema=False)
def callback(code: str = None, error: str = None, state: str = None):
    if error or not code:
        return HTMLResponse(_ERROR_HTML.format(reason=error or "no code received"), status_code=400)
    grant = validate_oauth_state(state)
    if not grant:
        return HTMLResponse(_ERROR_HTML.format(reason="invalid or expired state"), status_code=400)
    try:
        save_token(exchange_code(code, grant.code_verifier), grant.user_id)
    except Exception as e:
        return HTMLResponse(_ERROR_HTML.format(reason=str(e)), status_code=500)
    _sync_mcp_grant(grant.user_id)
    return HTMLResponse(_SUCCESS_HTML)


def _sync_mcp_grant(owner_id: str) -> None:
    """Keep the assistant's MCP grant in step with a (re)link.

    The MCP re-reads its token file per call, so no restart is needed. Only the
    user the assistant already follows may move the file: the MCP runs on one
    Spotify account, and seeding from someone else would switch accounts
    silently. Never raises - a link must not fail because of the assistant.
    """
    try:
        target = mcp_grant_service.target_user_id()
        if target is None or target == owner_id:
            mcp_grant_service.seed(force=True)
    except Exception as e:
        logger.warning("MCP grant seed after Spotify link failed: %s", e)


def _unlink_spotify(user: User) -> dict:
    deleted = delete_token(str(user.id))
    try:
        mcp_grant_service.revoke(str(user.id))
    except Exception as e:
        logger.warning("MCP grant revoke after unlink failed: %s", e)
    return {"message": "Spotify unlinked." if deleted else "Spotify was not linked."}


@router.delete("/token", summary="Unlink Spotify")
def sign_out(user: User = Depends(get_current_user)):
    return _unlink_spotify(user)


@router.post("/logout", summary="Log out (client discards JWT)")
def logout(user: User = Depends(get_current_user)):
    # JWT logout is client-side; do not unlink this user's Spotify account.
    return {"message": "Logged out."}


@router.get("/login", summary="Get Spotify OAuth URL (legacy alias)")
def login_spotify_legacy(user: User = Depends(get_current_user)):
    # The old Python web UI used GET /auth/login for Spotify OAuth. Keep it
    # working while POST /auth/login is the multi-user JWT login.
    return {"auth_url": build_auth_url(str(user.id))}


@router.get("/spotify/login", summary="Get Spotify OAuth URL (client alias)")
def spotify_login(user: User = Depends(get_current_user)):
    return {"auth_url": build_auth_url(str(user.id))}


@router.post("/spotify/token", summary="Import Spotify tokens from a mobile client")
def spotify_token_import(body: SpotifyTokenImport,
                         user: User = Depends(get_current_user)):
    """Store tokens produced by the app's own PKCE flow.

    A phone on the LAN cannot complete the server-side OAuth: Spotify only
    allows HTTPS redirects except for 127.0.0.1/[::1] loopback, which on the
    phone resolves to the phone itself. The app therefore authorises with its
    custom-scheme redirect and hands the resulting tokens here.
    """
    if not body.access_token or not body.refresh_token:
        raise HTTPException(status_code=422, detail="access_token and refresh_token are required")
    save_token({
        "access_token":  body.access_token,
        "refresh_token": body.refresh_token,
        "expires_in":    body.expires_in,
    }, str(user.id))
    _sync_mcp_grant(str(user.id))
    return {"connected": True}


@router.get("/spotify/mcp-grant", summary="Assistant Spotify grant status")
def mcp_grant_status(user: User = Depends(get_current_user)):
    """Whether the Spotify MCP (the assistant's tools) has a usable grant."""
    path = mcp_grant_service.token_file()
    target = mcp_grant_service.target_user_id()
    return {
        "enabled": mcp_grant_service.enabled(),
        "path": str(path),
        "file_present": path.exists(),
        "user_id": target,
        "follows_caller": target == str(user.id),
    }


@router.post("/spotify/mcp-grant", summary="Seed the assistant's Spotify grant")
def mcp_grant_seed(user: User = Depends(get_current_user)):
    """Write the caller's grant into the MCP token file.

    Only the user the assistant follows may do this: the MCP runs on a single
    Spotify account, so seeding from another user would switch accounts.
    """
    target = mcp_grant_service.target_user_id()
    if target != str(user.id):
        raise HTTPException(
            status_code=403,
            detail="The assistant follows a different Spotify account; "
                   "only that user can refresh its grant.",
        )
    return mcp_grant_service.seed(force=True, user_id=target)


@router.delete("/spotify", summary="Unlink Spotify (client alias)")
def spotify_unlink(user: User = Depends(get_current_user)):
    return _unlink_spotify(user)