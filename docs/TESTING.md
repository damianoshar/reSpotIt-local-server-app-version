# Tests & CI — Python server

```bash
# Backend (FastAPI, pytest)
cd server
pip install -r requirements.api.txt -r requirements-dev.txt
python -m pytest tests -q            # 16 test files, 216 tests

# Frontend (Vite + verification scripts)
cd server/frontend
npm ci
npm run build
npm run verify:local                 # local-mode + pipeline + render + CSS-build checks
```

Backend test files cover concerts and hidden albums, concert scans, CSV import (replace mode), Keycloak auth, LLM chats and the chunked LLM extraction service, the MCP service, the phase-0 endpoints, playlist actions, artist resolution, retention, Spotify token import and user auth.

CI (`.github/workflows/release.yml`, triggers on `v*` tags or manual dispatch):

| Step | What it does |
|---|---|
| Build frontend | `npm install` → `npm run build` → `npm run verify:local` in `server/frontend` |
| Install Python deps | `requirements.api.txt` + `requirements-dev.txt` + `desktop/requirements.txt` |
| Run tests | `python -m pytest tests -q` in `server` |
| Build desktop exe | `pyinstaller spotify-dashboard.spec --noconfirm` |
| Package release | `reSpotIt-Windows.zip` + `SHA256SUMS.txt` |
| Publish | GitHub release with the zip and the checksum file |
