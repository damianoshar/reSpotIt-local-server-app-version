# reSpotIt — Python Project

## About

reSpotIt was built to fill gaps in Spotify's library management. It is an album-centric tool that lets you organize your library and playlists using rich filters, add entire artist discographies to your library or turn them into playlists, import album links from any website, and build queues from albums you are listening to right now. This is a work in progress — expect rough edges and things that may not work yet.

Self-hosted server for the reSpotIt library, with a browser frontend and an optional AI assistant. Runs entirely on your machine.

---

## 🚀 Installation — Solo (reSpotIt only)

### Option A: Docker (recommended)
```bash
# Run from the repo root: docker-compose.yml and .env live there.
cp .env.example .env           # fill in SPOTIFY_CLIENT_ID (the client secret is optional)
docker compose up -d --build   # api :8000, frontend :8888, db :5433
```

**TLS certificates are auto-generated on first run** (self-signed, valid for the LAN IP). No manual cert setup needed.

- **Frontend:** http://192.168.1.11:8888
- **API:** https://192.168.1.11:8000 (health: `/api/v1/health`)
- **API docs:** https://192.168.1.11:8000/docs

> **Note:** Browser will show a cert warning for self-signed certs. Accept it or use `SSL_ENABLED=false` in `.env` for HTTP.
>
> **Exposed URLs always name the PC's LAN IP** (here `192.168.1.11`), never `localhost` — a phone or another PC resolves `localhost` to itself. Update `LAN_IP`, `PUBLIC_ORIGIN` and `PUBLIC_BASE_URL` in `.env` when your PC's address changes.
>
> **LAN exposure:** Docker Desktop publishes the containers on high host ports. Run `scripts\lan-expose.ps1` as Administrator once (and after every Docker restart) to publish them back on 8000/8888 and open the firewall.

### Option B: Desktop app (no Docker)
Self-contained executable (PyInstaller) that runs the FastAPI backend (serving the built frontend) and opens a native window.

```bash
cd desktop
pip install -r requirements.txt
python main.py --no-window   # backend only, for testing
python main.py               # backend + native window
# Build:
pyinstaller spotify-dashboard.spec --noconfirm
# -> desktop/dist/reSpotIt/reSpotIt.exe
```

Notes:
- The desktop app binds to the loopback interface and opens its own window on this PC (`http://127.0.0.1:8888`). It is not reachable from a phone or another PC — use the Docker stack for LAN access.
- Default port 8888. The Spotify OAuth redirect URI is `PUBLIC_BASE_URL` + `/api/v1/auth/callback` (from `.env`; the fallback when nothing is configured is `http://127.0.0.1:8888/api/v1/auth/callback`). Register the exact same URI in your Spotify developer app. If 8888 is busy the app exits with a clear error; use `--port <n>` for testing (OAuth will not work on other ports).
- Requires a reachable Postgres (default `postgresql://spotify:spotify@127.0.0.1:5433/spotify`); override via `DATABASE_URL`. Or use SQLite by setting `DATABASE_URL=sqlite:///path/to/db`.
- Data/logs/images live under `%APPDATA%\reSpotIt\data` (Windows) or `~/.local/share/reSpotIt/data` (Linux).
- Browser playback in the Python frontend uses Spotify's official Web Playback SDK (`server/frontend/src/components/PlayerView.jsx`); it requires Premium and the `streaming` scope.

> **Full stack, production deployment, LAN/phone access and environment variables:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
> **Full setup details:** `server/README.md`.

---

## 🧱 Stack

- **Backend:** Python/FastAPI with SQLAlchemy
- **Database:** PostgreSQL (SQLite supported for the desktop app)
- **Frontend:** React + Vite
- **Auth:** Spotify PKCE OAuth + optional Keycloak OIDC
- **Deployment:** Docker Compose (solo stack) with an optional full-stack overlay
- **AI stack:** Open WebUI + MCP services (full-stack overlay)

---

## reSpotIt Project Structure

reSpotIt is a multi-component project with several variants:

- **Local Server (standalone)** (this repo) — Python/FastAPI backend + React web frontend + PostgreSQL. Runs entirely on your machine. No AI assistant. Full feature set: library management, playlists, sync, stats, CSV import/export.
- **Local Server + AI assistant** — Same server plus OpenWebUI, Keycloak, and MCP services for AI-powered library queries. Available as a full stack overlay in this repo (`docker-compose.full.yml`).
- **AI URL scan** — In media-server mode the Android apps can scan any web page for albums/concerts (even pages without Spotify links) via the local LLM: "Scan page with AI (albums)" and "Scan concerts (AI)" in the Tools tab. Results resolve to Spotify and save to the library or become a playlist.
- **Chat sync across AI features (planned)** — Currently only the AI Assistant has Open WebUI chat history. Planned: each scan (albums/concerts) and future features (duplicate explain, metadata enrich, playlist describe, reports, reorganize) become chat threads with list/resume/refine.
- **Kotlin client** — Native Android app (Kotlin + Jetpack Compose). Full feature parity with media-server mode. See `reSpotIt-kotlin-version`.
- **React clients** — Capacitor/TypeScript Android apps. Offer better compatibility on older Android models and low-RAM devices than the Kotlin client. Two variants:
  - **Standard** — Web Playback SDK for browser-based playback.
  - **Librespot** — Experimental on-device full-track playback via Rust/JNI. Only this variant supports on-device audio.

> **Note:** All Android clients work fully offline without any server. Media-server mode is optional — enable it in Settings to sync, push/pull, and use the AI assistant.

Self-contained project cut from the general reSpotIt. Organised as:

```
server/            Python backend (FastAPI) + React frontend (Vite) + Docker
certs/             mkcert CA + server cert (gitignored; pre-generated for full stack)
data/              Local data (CSV, images) — gitignored
desktop/           Standalone desktop app (PyInstaller)
docs/              Additional documentation
OpenWebUI-stack/   nginx, keycloak realm, MCP sources (gitignored; included for full stack)
scripts/           Utility scripts
.github/           GitHub Actions / CI config
```

**Repository:** https://github.com/damianoshar/reSpotIt-local-server-app-version (branch: `main`)

**Android clients (current):**
- Kotlin/Compose: https://github.com/damianoshar/reSpotIt-kotlin-version
- React/Capacitor: https://github.com/damianoshar/reSpotIt-react-version (branch `main`)
- React/Capacitor + librespot: https://github.com/damianoshar/reSpotIt-react-version (branch `librespot-player`)

---

## 🎮 What the app does

| Area | What it does |
|---|---|
| **Library** | Browse saved albums, search, rich filters (genre, artist, year, decade, popularity), sorting, multiple view modes, album track sheet, stats |
| **Playlists** | Create, edit, delete and reorder Spotify playlists, add/remove albums, visibility, expiry (retention), duplicates inside a playlist |
| **Sync** | Pull the Spotify library to the server, push changes back, background jobs with progress, retention cleanup |
| **Duplicates & hidden albums** | Detect duplicate albums, hide them, or permanently remove them from Spotify |
| **Import / export** | CSV import and export for backup and migration, artwork cache |
| **AI (full stack)** | Open WebUI assistant, AI page scans (albums and concerts), concerts calendar, chat sync across AI features |
| **Auth** | Spotify PKCE OAuth, optional Keycloak OIDC accounts, per-user API tokens |
| **Client API** | Frozen REST contract under `/api/v1` for the Kotlin and React Android clients |

---

## 📚 Documentation

| Document | What it covers |
|---|---|
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Full-stack overlay, production deployment, LAN/phone access, environment variables |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Server details, repository layout, `/api/v1` routers |
| [`docs/TESTING.md`](docs/TESTING.md) | Pytest suite and release CI |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Full-stack and solo-stack failures |
| [`docs/STORAGE-POLICY.md`](docs/STORAGE-POLICY.md) | Canonical server/device data ownership |

---

## 📋 Notes

- `.env` lives at the repo root (referenced by `docker-compose.yml`).
- `data/` stays at the root and is mounted into the api container.
- Build artifacts (`node_modules/`, `dist/`, `build/`, `.gradle/`, `.vs/`) are gitignored.
- The Android clients live in **separate repositories** (see the links at the top): `reSpotIt-kotlin-version`, `reSpotIt-react-version` (branch `main`) and `reSpotIt-react-version` (branch `librespot-player`). This repository contains no client sources.
- Full stack files (`certs/`, `OpenWebUI-stack/`) are now included in this repo (gitignored). The start-full-stack script copies them from a full-stack workspace when they are missing.
- The Open WebUI API key is auto-generated on first deploy and persisted to `/data/.owui_api_key`. No manual key creation needed.
- The one-click launcher preserves an existing `.env`; it only fills missing or placeholder secrets. This is deliberate: changing DB passwords after the Postgres volumes exist locks the stack out.
