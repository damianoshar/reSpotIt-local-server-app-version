<#
.SYNOPSIS
  Give the Spotify MCP the grant the app already holds, then verify it.

.DESCRIPTION
  The assistant's Spotify tools run on one Spotify account. That grant is the
  app's own link (`GET /api/v1/auth/spotify/login` -> `/api/v1/auth/callback`,
  rows in `spotify_tokens`); the MCP just needs a copy in its token file.
  `mcp_grant_service` (API) now does that automatically at startup and after
  every Spotify link or relink, so this script is only needed to force a
  re-seed, to seed from an explicit user, or to run the package's own OAuth flow
  when nobody has linked Spotify in the app yet.

  The running MCP re-reads its token file per call, so no restart is needed.

  Background and evidence: docs/SPOTIFY-MCP-AUTH-PLAN.md

.PARAMETER User
  API user id whose grant seeds the MCP. Defaults to the user the assistant
  already follows (recorded in the token file), then SPOTIFY_MCP_GRANT_USER,
  then the earliest linked user.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/spotify-mcp-auth.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/spotify-mcp-auth.ps1 -Browser
#>
[CmdletBinding()]
param(
    [string]$Container = 'respotit-local-server-app-spotify-mcp-1',
    [string]$ApiContainer = 'respotit-local-server-app-api-1',
    [string]$User = '',
    [switch]$Browser,
    [string]$ScopeProfile = 'full'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($message) { Write-Host "`n== $message" -ForegroundColor Cyan }

function Get-DotEnvValue([string]$name) {
    $line = Select-String -Path (Join-Path $repo '.env') -Pattern "^$name=(.*)$" | Select-Object -First 1
    if (-not $line) { return '' }
    return $line.Matches[0].Groups[1].Value.Trim()
}

$tokenFile = Join-Path $repo 'OpenWebUI-stack\spotify-mcp-data\tokens.json'

function Verify {
    Step '2/2 verify'
    docker exec $Container sh -c 'spotify-mcp doctor 2>&1 | sed -n "1,16p"'
    $probe = @'
from api.services import mcp_service as m

print("tools:", [s["name"] + ":" + str(s["tools"]) for s in m.server_status()])
print("devices:", m.call_tool("get_devices", {})[:200].replace("\n", " "))
'@
    $tmp = Join-Path $env:TEMP 'spotify-mcp-probe.py'
    Set-Content -Path $tmp -Value $probe -Encoding ascii
    docker cp $tmp "${ApiContainer}:/tmp/spotify-mcp-probe.py" | Out-Null
    docker exec -e 'PYTHONPATH=/app' -w /app $ApiContainer python /tmp/spotify-mcp-probe.py
}

function UseAppGrant {
    Step '1/2 seed the MCP from the app grant'
    $seed = @'
import json, os
from api.services import mcp_grant_service as grant

user = os.environ.get("GRANT_USER") or None
status = grant.seed(force=True, user_id=user)
print(json.dumps(status))
'@
    $tmp = Join-Path $env:TEMP 'spotify-mcp-seed.py'
    Set-Content -Path $tmp -Value $seed -Encoding ascii
    docker cp $tmp "${ApiContainer}:/tmp/spotify-mcp-seed.py" | Out-Null
    $json = docker exec -e "GRANT_USER=$User" -e 'PYTHONPATH=/app' -w /app $ApiContainer python /tmp/spotify-mcp-seed.py 2>$null
    if (-not $json) { throw 'seeding returned nothing' }
    $status = $json | ConvertFrom-Json
    Write-Host "   $($json.Trim())"
    if ($status.status -in 'written', 'unchanged') { return $true }
    Write-Host '   the API has no usable grant for that user' -ForegroundColor Yellow
    return $false
}

if (-not $Browser) {
    if (UseAppGrant) {
        Verify
        Write-Host "`nDone - the assistant has Spotify again, on the app's own grant." -ForegroundColor Green
        exit 0
    }
    Write-Host '   falling back to the browser flow: link Spotify in the app first to avoid it' -ForegroundColor Yellow
}

Step 'browser 1/2 OAuth consent'
$clientId = Get-DotEnvValue 'SPOTIFY_CLIENT_ID'
$redirect = Get-DotEnvValue 'SPOTIFY_MCP_REDIRECT_URI'
if (-not $redirect) { $redirect = 'http://127.0.0.1:8899/callback' }
if (-not $clientId) { throw 'SPOTIFY_CLIENT_ID is empty in .env' }
if (-not (Get-Command npx -ErrorAction SilentlyContinue)) { throw 'npx is not on PATH; install Node 22+.' }
$callbackPort = ([uri]$redirect).Port
if (Get-NetTCPConnection -LocalPort $callbackPort -State Listen -ErrorAction SilentlyContinue) {
    throw "127.0.0.1:$callbackPort is already in use. Set SPOTIFY_MCP_REDIRECT_URI in .env to a free port, add that URI to the Spotify app, and rerun."
}
Write-Host "   redirect URI    $redirect"
Write-Host '   it must be registered in the Spotify app exactly.' -ForegroundColor Yellow
$saved = @{}
foreach ($name in 'SPOTIFY_CLIENT_ID', 'SPOTIFY_REDIRECT_URI', 'SPOTIFY_MCP_TOKEN_FILE') {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name)
}
$env:SPOTIFY_CLIENT_ID = $clientId
$env:SPOTIFY_REDIRECT_URI = $redirect
$env:SPOTIFY_MCP_TOKEN_FILE = $tokenFile
try {
    npx -y '@novalux12/spotify-mcp@3.1.1' auth --scope-profile $ScopeProfile
    if ($LASTEXITCODE -ne 0) { throw "The auth flow exited with $LASTEXITCODE." }
}
finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
if (-not (Test-Path $tokenFile)) { throw "No token file at $tokenFile - the flow did not finish." }
Verify
