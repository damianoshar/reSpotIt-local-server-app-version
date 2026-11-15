# Troubleshooting — Python server

## Full-stack

| Symptom | Cause | One-click fix |
| --- | --- | --- |
| Open WebUI says **Backend Required** | nginx `location /api/` shadows Open WebUI's own `/api/` | `prepare_full_stack.py` replaces it with `/respotit/` |
| **Your provider has not provided a trusted header** | `WEBUI_AUTH_TRUSTED_*` on the public instance | public instance has no trusted headers; API uses `open-webui-2` |
| **Sign-in with your identity provider failed** | `OAUTH_CLIENT_SECRET` / `KEYCLOAK_CLIENT_SECRET` mismatch | prepare syncs both with the live Keycloak client secret |
| OIDC callback cannot reach `https://<pc-lan-ip>` | container does not trust the mkcert root | prepare appends `certs/ca/rootCA.pem` to the CA bundle |
| Keycloak redirects to `localhost` | stale realm redirect URI or Open WebUI `webui.url` | prepare rewrites `.env`; bootstrap patches the live realm and DB |
| reSpotIt MCP has no tools in Open WebUI | connection not registered | bootstrap registers it and verifies 38 tools |
| AI assistant has no library data | service account has no Spotify link / no sync | bootstrap copies the owner's link and starts a sync |
| `.env` lost but Postgres volumes remain | new random DB passwords | restore the old `.env`; do not delete volumes |

## Solo stack

| Symptom | Cause | Fix |
|---|---|---|
| Browser shows a cert warning | Self-signed certificate (solo mode) | Accept it, or set `SSL_ENABLED=false` in `.env` for plain HTTP |
| Container exits at once | Port already in use (8000/8888/5433) | `docker compose ps`, then free the port or change the published port |
| "password authentication failed for user" | `.env` DB password changed after the volume was created | Restore the previous `.env`; do **not** delete `docker-data/postgres` |
| Library empty after a sync | Spotify account has no saved albums, or the job was cancelled | Open the sync panel for the stage log; `GET /api/v1/sync/history` |
| Android client cannot reach the API | Cert does not cover the LAN IP, or the port is not published | Reissue the leaf with the LAN IP in the SAN list; `scripts/lan-expose.ps1`; firewall rule |
| AI assistant disabled | No Open WebUI API key yet | The key is auto-generated on first deploy and stored at `/data/.owui_api_key`; re-run `scripts/bootstrap_full_stack.py` |
| Frontend builds but the API 404s | Wrong base URL / proxy prefix | Solo: the API is at `https://127.0.0.1:8000`; full stack: `/respotit/` (nginx) |
| Migration mismatch after pulling | New migration file not applied | Apply `server/migrations/*.sql` in order; the API runs them on start |
