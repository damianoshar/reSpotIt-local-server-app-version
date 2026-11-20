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
| Albums lose their date after a push | the pushed CSV has an empty `added_at` column, and the upsert overwrote the stored date | the upsert keeps the date it already knows (`COALESCE`); re-push a CSV that carries the column |
| Spotify link fails with **"invalid_client: Invalid client secret"** | the deployment has no client secret and the exchange sent an empty one | the flow uses PKCE when `SPOTIFY_CLIENT_SECRET` is empty; set the secret in `.env` if the Spotify app has one |
| Assistant answers **"could not authenticate with Spotify"** / an MCP tool returns `authentication_required` | the Spotify MCP's token file is missing (`OpenWebUI-stack/spotify-mcp-data/tokens.json`) | it is seeded automatically from the app's Spotify link at API startup and after every link; to force it `POST /api/v1/auth/spotify/mcp-grant` or run `scripts/spotify-mcp-auth.ps1` — see `docs/SPOTIFY-MCP-AUTH-PLAN.md` |
| Assistant **stops after a tool call** and needs a prompt like "continue" / "Make the playlist" / "Done?" | two causes, both fixed: the model wrote its call as `<tool_call>` **text** and the loop treated that as the final answer, and the replayed chain dropped the stored tool results so the model could not tell a step had run | see `docs/AI-TOOL-ROUTING-PLAN.md` §5.1–5.2; `tool_route.text_calls` shows calls that arrived as text |
| React media-server client cannot delete playlists | The remote playlist delete route was missing or the client still uses a local-only build | Update both the React client and server. `DELETE /api/v1/playlists/{playlist_id}` unfollows the playlist on Spotify and clears its server cache and retention rule. |
| Repeated `artist` filters return no/missing albums | An older server interpreted `artist` as one query value | Update the server. `/api/v1/albums` and `/api/v1/playlists/{playlist_id}/albums` accept repeated `artist` parameters and match any selected artist credit (OR), including collaborators. |
| The server web player does nothing for a single track | The web player sent a `spotify:track:` URI as `context_uri` | Update the server frontend. Single tracks are sent as `uris: [trackUri]`; albums and playlists keep `context_uri`. |
| Assistant shows raw **`<tool_call>` / `<function=…>`** markup in its reply | the model emitted the template markup as content instead of a native call | the loop now executes that markup and strips it from the answer (§5.2) |
| Assistant turn fails with **"Failed to parse tool call arguments as JSON"** (HTTP 500) | the model hit its token limit mid-call, and its truncated arguments string (a `spotify:track:` URI cut before the closing quote) was replayed to Open WebUI | the loop re-serialises parsed arguments, refuses to run a half-parsed call, and sanitises stored output items on replay; retry with fewer items per call if it repeats |
| Assistant searches all Spotify after **"search the library"** | the hot `search` tool searches the global catalog and could outrank local-library tools | update the API: library intent now prefers and pre-reveals local-library readers, hides catalog `search` for that turn when a local reader is available, and labels catalog tools as global search |
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
