# Deployment — Python server

Full-stack overlay, production deployment, LAN/phone access and environment variables.

---

## 🤖 Full Stack — reSpotIt + OpenWebUI + Keycloak + MCP

### One-click (recommended)
```bash
./start-full-stack.sh          # Linux/macOS
start-full-stack.bat           # Windows
```

The launcher is **idempotent** — run it as often as you like. It:

1. creates `.env` on first run, then **preserves existing secrets** (rotating
   DB passwords would break existing Postgres volumes);
2. copies `certs/` and `OpenWebUI-stack/` from the full-stack workspace when
   they are missing;
3. repairs the known full-stack hiccups before starting containers:
   - replaces the legacy nginx `location /api/` with `/respotit/` so Open
     WebUI's own `/api/config` is not shadowed (the "Backend Required" error);
   - keeps `WEBUI_AUTH_TRUSTED_*` off the public `open-webui` instance and
     points the API's trusted-header exchange at the internal `open-webui-2`;
   - makes `OAUTH_CLIENT_SECRET` / `KEYCLOAK_CLIENT_SECRET` match the running
     Keycloak client (or the realm export on first run);
   - appends the local mkcert root CA to Open WebUI's CA bundle so the OIDC
     backchannel can verify the LAN origin;
4. starts the stack, then bootstraps it:
   - creates the reSpotIt MCP service account;
   - copies the owner's Spotify link to that account on a single-user stack;
   - generates the Open WebUI API key used by the AI assistant;
   - registers/refreshes the reSpotIt MCP connection (38 tools);
   - starts a library sync when a Spotify link is available;
   - verifies the public routes, OIDC discovery, Keycloak client secret and
     that public trusted-header auth is disabled.

Pass `--no-sync` to skip the initial library sync, or `--no-verify` to skip
the final checks.

### First run
1. Open **`https://<pc-lan-ip>/`** (printed by the launcher; also stored as
   `PUBLIC_ORIGIN` in `.env`) and click Keycloak's **Register** link to create
   your account (the realm allows self-registration).
2. Open **`http://<pc-lan-ip>:8888`**, sign in to reSpotIt, and connect Spotify.
3. Re-run the one-click launcher (or just
   `python scripts/bootstrap_full_stack.py`) to copy the Spotify link to the
   MCP service account and sync the library.

Every exposed URL names the PC's LAN IP, never `localhost`: a phone or another
PC resolves `localhost` to itself. The origin is detected on every start and
written to `.env` as `LAN_IP` / `PUBLIC_ORIGIN`; `KEYCLOAK_HOSTNAME`,
`OPENID_PROVIDER_URL`, `OPENID_REDIRECT_URI`, `PUBLIC_BASE_URL` and the Keycloak
realm follow it automatically.

### Manual start
```bash
python scripts/prepare_full_stack.py
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d --build
python scripts/bootstrap_full_stack.py
```

> **Note:** The full stack uses pre-generated mkcert certs for nginx/Keycloak/OpenWebUI trust. The reSpotIt API uses the same certs (mounted read-only).

This adds:
- OpenWebUI at `https://<pc-lan-ip>/` (via nginx proxy on 443)
- Keycloak at `https://<pc-lan-ip>/authKeycloak`
- MinIO at :9000/:9001
- Redis + Redis Insight at :5540
- reSpotIt MCP at 127.0.0.1:8091
- Spotify MCP at 127.0.0.1:8090
- reSpotIt API through nginx at `https://<pc-lan-ip>/respotit/`

> **MinIO / Redis:** both are wired to Open WebUI (restored 2026-11-08) — `STORAGE_PROVIDER=s3` with `S3_*` for file uploads and `WEBSOCKET_MANAGER=redis` for websockets, on both `open-webui` and `open-webui-2`. Credentials come from `MINIO_*` (defaults keep existing data readable).

### Option 2: Self-contained full-stack workspace
```bash
cd ../<your-full-stack-workspace>
docker compose up -d --build
```

A self-contained workspace has all required files (certs, OpenWebUI-stack,
.env.example) pre-configured. See the project README for where to obtain it.

---

## 📦 Deployment

### Solo (Docker)
```bash
docker compose build
docker compose up -d
```

### Full Stack (Docker)
```bash
python scripts/prepare_full_stack.py
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d
python scripts/bootstrap_full_stack.py
```

- Data: everything now lives under `./docker-data/` (no named volumes). Back up `./data` and `./docker-data/`.
- Full stack bind-mounts: `./docker-data/postgres`, `./docker-data/minio`, `./docker-data/open-webui`, `./docker-data/redis`, `./docker-data/pipelines`, `./docker-data/keycloak-db`, `./docker-data/openwebui-db`

---

## 📡 LAN / phone access

The Android clients reach the server over HTTPS on the LAN, so the cert must
cover the PC's LAN IP.

```bash
# Regenerate certs/server.pem with the current LAN IPs in the SAN list.
# Replace 192.168.1.11 with the PC's address (ipconfig / Get-NetIPAddress).
docker run --rm -v "$PWD/certs:/certs" alpine/openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout /certs/server-key.pem -out /certs/server.pem -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:api,IP:127.0.0.1,IP:192.168.1.11"
docker compose -f docker-compose.yml -f docker-compose.full.yml restart api
```

- **Cert trust:** the API cert is issued from the local development CA in
  `certs/ca/rootCA.pem` (mkcert or a self-created CA). The Android apps trust
  that CA:
  - *Personal builds* (gitignored `src/personal/res/` overlay) bundle the CA at
    `res/raw/respotit_ca.pem` -> zero-touch install.
  - *Public/CI builds* trust only system + user CAs; the user installs the CA
    once (Settings -> Security -> Install a certificate -> CA certificate, pick
    `rootCA.pem`). Because the CA is stable (10 years), this install survives
    every server cert reissue and every app update.
  - Reissue the leaf for a new LAN IP with `scripts/local-certs.sh <ip>` and
    restart the API container. No app rebuild needed.
- **Port:** use `https://<pc-ip>:8000`. If Docker Desktop on Windows publishes
  port 8000 to loopback only, forward it with `lan_proxy.py`
  (`0.0.0.0:8001 → 127.0.0.1:8000`, kept in
  `%LOCALAPPDATA%\reSpotIt\lan_proxy.py` and started from the user's Startup
  folder) and use `https://<pc-ip>:8001` instead.
- **Firewall:** allow inbound TCP on the chosen port for the Public profile
  (`netsh advfirewall firewall add rule name="reSpotIt API 8001" dir=in action=allow protocol=TCP localport=8001`).

---

## 🔑 Environment variables

`.env` lives at the repo root (`.env.example` documents every key — 56 in total). The most important ones:

| Group | Keys |
|---|---|
| Spotify | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `REDIRECT_URI`, `PUBLIC_BASE_URL` |
| LAN origin | `LAN_IP`, `PUBLIC_ORIGIN` (auto-detected and rewritten on every start) |
| Server | `SSL_ENABLED`, `CORS_ALLOW_ORIGIN`, `FORWARDED_ALLOW_IPS` |
| Database | `DATABASE_URL` (default `postgresql://spotify:spotify@127.0.0.1:5433/spotify`) |
| Keycloak / OIDC | `KEYCLOAK_ENABLED`, `KEYCLOAK_REALM`, `KEYCLOAK_HOSTNAME`, `KEYCLOAK_INTERNAL_URL`, `KEYCLOAK_PUBLIC_URL`, `KEYCLOAK_ISSUER`, `KEYCLOAK_REDIRECT_URI`, `KEYCLOAK_ALLOWED_REDIRECTS`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, `KEYCLOAK_ADMIN`, `KEYCLOAK_ADMIN_PASSWORD`, `KEYCLOAK_DB_USER`, `KEYCLOAK_DB_PASSWORD`, `OPENID_PROVIDER_URL`, `OPENID_REDIRECT_URI`, `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_SCOPES`, `OAUTH_PROVIDER_NAME`, `OAUTH_AUTO_SIGNUP`, `OAUTH_MERGE_ACCOUNTS_BY_EMAIL`, `ENABLE_OAUTH_SIGNUP` |
| Open WebUI | `WEBUI_AUTH`, `WEBUI_SECRET_KEY`, `DEFAULT_USER_ROLE`, `OPENWEBUI_DB_*`, `OPENWEBUI_URL`, `OPENWEBUI_TRUSTED_AUTH`, `OPENWEBUI_TRUSTED_HEADER`, `OPENWEBUI_TRUSTED_NAME_HEADER`, `OPENAI_API_BASE_URL`, `OPENAI_API_KEY` |
| MCP | `MCP_API_KEY`, `MCP_TOOLS`, `RESPOTIT_MCP_*`, `SPOTIFY_MCP_URL`, `SPOTIFY_MCP_TOKEN` |
| MinIO / Redis | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `MINIO_BUCKET_NAME`, `MINIO_REGION` |
| Telemetry | `SCARF_NO_ANALYTICS`, `DO_NOT_TRACK`, `ANONYMIZED_TELEMETRY` |

> **Secrets:** `.env`, `certs/`, `OpenWebUI-stack/` and `data/` are gitignored. Never commit DB passwords, Keycloak client secrets, the Open WebUI API key (`/data/.owui_api_key`) or MCP tokens. The one-click launcher deliberately **preserves** an existing `.env` — changing the DB password after the Postgres volume exists locks the stack out.
