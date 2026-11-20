"""Keep the Spotify MCP's token file in step with the app's Spotify link.

`spotify-mcp` (third-party) keeps its own token file and refreshes it itself.
The app already owns a Spotify grant per user, and Spotify keeps that grant's
refresh token stable across refreshes (measured 2026-11-16), so the MCP can run
on the app's grant instead of a second consent. This module writes the file; the
MCP does the refreshing from then on.

    SPOTIFY_MCP_DATA_DIR    directory the MCP reads ``tokens.json`` from
                            (default ``/data/spotify-mcp``, the bind mount)
    SPOTIFY_MCP_GRANT_SEED  "0" disables seeding (default enabled)
    SPOTIFY_MCP_GRANT_USER  the user whose grant the MCP follows; when unset the
                            user recorded in the file is kept, else the earliest
                            linked user

The file carries an extra ``user_id`` field so a relink or unlink by that user
can be recognised. The MCP's own refresh spreads the old object, so the field
survives its write-back. The running MCP re-reads the file on demand, so a seed
takes effect on the next tool call without a restart.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..database import SessionLocal
from ..models import SpotifyToken
from . import auth_service

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/data/spotify-mcp"
GRANT_FILE = "tokens.json"
#: Matches the lifetime Spotify hands out; the MCP refreshes on its own after.
ACCESS_TOKEN_TTL_SECONDS = 3500


def enabled() -> bool:
    return os.getenv("SPOTIFY_MCP_GRANT_SEED", "1").strip().lower() not in {"0", "false", "no", "off"}


def token_file() -> Path:
    return Path(os.getenv("SPOTIFY_MCP_DATA_DIR", DEFAULT_DATA_DIR)) / GRANT_FILE


def _read_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _linked_user_ids() -> list[str]:
    """Users with a Spotify grant, oldest first. Tests patch this."""
    with SessionLocal() as db:
        rows = db.query(SpotifyToken.user_id).order_by(SpotifyToken.created_at).all()
    return [str(row[0]) for row in rows]


def target_user_id(explicit: str | None = None, path: Path | None = None) -> str | None:
    """The user whose grant the MCP runs on.

    Order: the caller's pick, the configured grant user, the user already
    recorded in the file, then the earliest linked user. The recorded user sits
    ahead of the fallback so a second person linking Spotify cannot silently
    move the assistant to another account.
    """
    linked = _linked_user_ids()
    recorded = (_read_file(path or token_file()) or {}).get("user_id")
    configured = os.getenv("SPOTIFY_MCP_GRANT_USER", "").strip()
    for candidate in (explicit, configured, recorded):
        if candidate and candidate in linked:
            return candidate
    return linked[0] if linked else None


def seed(force: bool = False, user_id: str | None = None) -> dict[str, Any]:
    """Write the app's grant to the MCP token file. Never raises."""
    if not enabled():
        return {"status": "disabled"}

    path = token_file()
    try:
        target = target_user_id(user_id, path)
    except Exception as exc:  # database not reachable during startup races
        logger.warning("MCP grant: cannot resolve the linked user: %s", exc)
        return {"status": "error", "detail": str(exc)}
    if not target:
        return {"status": "skipped", "reason": "no linked Spotify user"}

    try:
        row = auth_service.load_token(target) or {}
        refresh = row.get("refresh_token")
        if not refresh:
            return {"status": "skipped", "reason": "the target user has no grant"}
        existing = _read_file(path)
        if (
            not force
            and existing
            and existing.get("refresh_token") == refresh
            and existing.get("user_id") == target
        ):
            # Keep whatever the MCP wrote back: its access token may be fresher.
            return {"status": "unchanged", "user_id": target, "path": str(path)}
        access = auth_service.get_valid_token(target)
    except Exception as exc:
        logger.warning("MCP grant: cannot read the grant: %s", exc)
        return {"status": "error", "detail": str(exc)}

    payload = {
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": int(time.time() * 1000) + ACCESS_TOKEN_TTL_SECONDS * 1000,
        "scope": auth_service.SCOPE,
        "user_id": target,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{GRANT_FILE}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("MCP grant: cannot write %s: %s", path, exc)
        return {"status": "error", "detail": str(exc)}

    logger.info("MCP grant seeded from user %s into %s", target[:8], path)
    return {"status": "written", "user_id": target, "path": str(path)}


def revoke(user_id: str | None = None) -> dict[str, Any]:
    """Drop the MCP token file when its user unlinks Spotify."""
    if not enabled():
        return {"status": "disabled"}
    path = token_file()
    if not path.exists():
        return {"status": "absent"}

    recorded = (_read_file(path) or {}).get("user_id")
    owner = recorded or os.getenv("SPOTIFY_MCP_GRANT_USER", "").strip() or None
    if user_id and owner and user_id != owner:
        return {"status": "ignored", "reason": "not the user the MCP follows"}
    if user_id and not owner:
        return {"status": "ignored", "reason": "the MCP file records no user"}
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("MCP grant: cannot remove %s: %s", path, exc)
        return {"status": "error", "detail": str(exc)}
    logger.info("MCP grant removed (%s)", path)
    return {"status": "removed", "path": str(path)}
