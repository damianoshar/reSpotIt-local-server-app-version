# Re-apply Windows portproxy + firewall rules that expose Docker Desktop ports
# (which are published on high host ports) back onto their original LAN ports.
# Run as Administrator. Requires Docker Desktop running with the shifted compose ports.
param()
$mappings = @(
  @{L=80;C=10080},
  @{L=443;C=10443},
  @{L=3000;C=13000},
  @{L=8000;C=18000},
  @{L=8888;C=18888},
  @{L=5433;C=15433},
  @{L=9000;C=19000},
  @{L=9001;C=19001},
  @{L=5540;C=15540}
)
foreach ($m in $mappings) {
  netsh interface portproxy delete v4tov4 listenport=$($m.L) listenaddress=0.0.0.0 2>$null | Out-Null
  netsh interface portproxy add v4tov4 listenport=$($m.L) listenaddress=0.0.0.0 connectport=$($m.C) connectaddress=127.0.0.1
  New-NetFirewallRule -DisplayName "DockerLAN_$($m.L)" -Direction Inbound -LocalPort $($m.L) -Protocol TCP -Action Allow -Profile Any -ErrorAction SilentlyContinue | Out-Null
}
Write-Host "LAN portproxy + firewall rules applied."
netsh interface portproxy show all