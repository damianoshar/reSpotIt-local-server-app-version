# Architecture & API — Python server

Server details, the repository layout and the `/api/v1` routers.

---

## 🔧 Server Details

- **Backend:** Python/FastAPI with SQLAlchemy + PostgreSQL
- **Frontend:** React/Vite (served via nginx in production)
- **Auth:** Spotify PKCE OAuth with token refresh + Keycloak OIDC (optional)
- **API contract:** Frozen `/api/v1` — additive changes only
- **HTTPS:** Auto-generated self-signed certs (solo) or mkcert (full stack)
- **AI assistant:** Open WebUI API key auto-generated on first deploy

---

## 📁 Project Layout

```
server/                ← Python backend (FastAPI) + React frontend (Vite) + Docker
├── api/              ← FastAPI application
│   ├── main.py       ← App factory, router registration, Open WebUI key bootstrap
│   ├── config.py     ← Settings (env-driven)
│   ├── database.py   ← SQLAlchemy engine/session
│   ├── models.py     ← SQLAlchemy models
│   ├── schemas.py    ← Pydantic DTOs
│   ├── contract.py   ← Frozen /api/v1 contract helpers
│   ├── scoping.py    ← Per-user data scoping
│   ├── job_store.py  ← In-memory job registry
│   ├── routers/      ← /api/v1 routes (16 routers, see below)
│   └── services/     ← Business logic (Spotify, sync, CSV, images, LLM, MCP, Keycloak)
├── frontend/         ← React/Vite frontend (PlayerView.jsx = Web Playback SDK)
│   ├── src/          ← React sources
│   └── scripts/      ← verify-local-mode / verify-pipeline / render-check / css-build
├── migrations/       ← 001..015 SQL migrations (concerts, hidden albums, LLM chats, …)
├── tests/            ← pytest suite (16 test files, 216 tests)
├── scripts/          ← smoke-python.ps1
├── Dockerfile.api / Dockerfile.api.dev
├── entrypoint.sh     ← TLS cert generation + uvicorn start
├── requirements.txt / requirements.api.txt / requirements-dev.txt
└── README.md         ← API reference, local LLM, Spotify setup

.github/              ← GitHub Actions / CI config (release.yml)
certs/                ← mkcert CA + server cert (gitignored; pre-generated for full stack)
data/                 ← Local CSV/image cache (gitignored)
desktop/              ← Standalone desktop app (PyInstaller)
docs/                 ← Additional documentation (STORAGE-POLICY.md, DEPLOYMENT.md, …)
OpenWebUI-stack/      ← nginx, keycloak realm, MCP sources (gitignored; included for full stack)
scripts/              ← Utility scripts
├── prepare_full_stack.py  ← idempotent pre-start repair (nginx, CA, .env)
├── bootstrap_full_stack.py ← post-start bootstrap (MCP, sync, verification)
├── local-certs.sh    ← Reissue the leaf cert for a new LAN IP
├── lan-expose.ps1    ← Publish the API on the LAN (portproxy + firewall)
└── backup.ps1 / backup.sh ← Dump ./data + Postgres into ./backups/<timestamp>/

docker-compose.yml         ← Solo stack (db, api, frontend) — repo root
docker-compose.full.yml    ← Full stack overlay (OpenWebUI, Keycloak, MCP, etc.)
.env.example               ← Template for all configs (56 keys)
.gitignore                 ← Ignores .env, certs/, OpenWebUI-stack/, data/, build artifacts
start-full-stack.sh/.bat  ← One-click deploy scripts
```

---

## API routers (`/api/v1`)

`albums` · `artists` · `auth` · `concerts` · `concert-scans` · `exports` · `hidden-albums`
· `images` · `import` · `jobs` · `keycloak-auth` · `llm` · `load` · `meta` · `spotify` · `sync`

The contract is **frozen**: additive changes only. Endpoint-level detail lives in
`server/README.md` ("API" section) and in the interactive docs at `/docs`.
