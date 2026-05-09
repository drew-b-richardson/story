# Hosting Setup — other-door.xyz

This documents how the public-facing version of the story app is hosted on other-door.xyz for a small group of friends.

---

## Architecture

- **App:** Flask server running on port 5000 with full features (TTS, Japanese support, story indexing)
- **LLM:** Ollama running locally on the Mac Mini M4 with the Mistral Nemo 12B model
- **Tunnel:** Cloudflare Tunnel routes `https://other-door.xyz` → `localhost:5000`

---

## How It Works

One launchd service starts automatically on login:

| Service | What it does |
|---|---|
| `xyz.otherdoor.tunnel` | Runs `cloudflared tunnel run other-door` |

The app itself (`python server.py`) can be started manually or configured as another launchd service if desired.

Logs are written to:
- `~/Library/Logs/otherdoor-tunnel.log`
- App logs (if configured) would go to `~/Library/Logs/otherdoor-server.log`

---

## Starting / Stopping the Tunnel

```bash
# Start tunnel service
launchctl load ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Stop tunnel service
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Restart the tunnel
launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.tunnel

# Restart the server
launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.server

# Stop the server
launchctl stop gui/$(id -u)/xyz.otherdoor.server          

# Check status
launchctl list | grep otherdoor

# Watch tunnel logs
tail -f ~/Library/Logs/otherdoor-tunnel.log
```

The app server itself runs via `python server.py` — either manually in a terminal or configured as a separate launchd service if you prefer it to auto-start.

---

## Completely Removing the Hosting Setup

### 1. Stop and remove the tunnel launchd service

```bash
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist
rm ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist
```

### 2. Delete the tunnel (Cloudflare side)

```bash
cloudflared tunnel delete other-door
```

### 3. Remove cloudflared credentials

```bash
rm ~/.cloudflared/d6593f84-9edb-4146-b28d-5c3157b7a8a7.json
rm ~/.cloudflared/config.yml
```

### 4. Remove the domain

Go to **dash.cloudflare.com** → other-door.xyz → **Overview** → scroll to bottom → **Cancel subscription** (if registered through Cloudflare Registrar).

### 5. Optionally remove cloudflared itself

```bash
brew uninstall cloudflared
```

### 6. Remove logs

```bash
rm ~/Library/Logs/otherdoor-tunnel.log
```

---

## Cloudflare Tunnel Details

| Field | Value |
|---|---|
| Tunnel name | `other-door` |
| Tunnel ID | `d6593f84-9edb-4146-b28d-5c3157b7a8a7` |
| Credentials | `~/.cloudflared/d6593f84-9edb-4146-b28d-5c3157b7a8a7.json` |
| Config | `~/.cloudflared/config.yml` |
| Hostname | `other-door.xyz` |
| Target | `http://localhost:5000` |

---

## Troubleshooting

**Site not loading:**
1. Check tunnel service is running: `launchctl list | grep otherdoor`
2. Check server is up: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/` (expect `200`)
3. Check tunnel logs: `tail -f ~/Library/Logs/otherdoor-tunnel.log` (look for `Registered tunnel connection`)

**Server not starting:**
- Port 5000 may already be in use: `lsof -i :5000`
- Stop other processes using that port or run on a different port via `PORT=5001 python server.py`

**Tunnel connected but site unreachable:**
- DNS may still be propagating (can take up to a few hours after first setup)
- Test from a phone on cellular data to bypass local DNS cache

**Tunnel drops when I step away from the Mac:**

Symptom in `~/Library/Logs/otherdoor-tunnel.log`:
```
WRN Serve tunnel error error="control stream encountered a failure while serving"
WRN Connection terminated error="control stream encountered a failure while serving"
```

This is macOS sleeping the machine (or suspending its network) — `cloudflared` loses its control stream and retries forever while the host is asleep. Also: LaunchAgents only run while you're logged in at the GUI, so logging out stops them entirely.

### Fix 1 — keep the Mac awake (required)

```bash
sudo pmset -a sleep 0          # never sleep
sudo pmset -a disksleep 0      # don't spin down disk
sudo pmset -a womp 1           # wake on network
sudo pmset -a powernap 1
sudo pmset -a tcpkeepalive 1
```

Also in **System Settings → Energy**: enable "Prevent automatic sleeping when the display is off" and "Start up automatically after a power failure."

### Fix 2 — run tunnel as LaunchDaemon so it survives logout (optional)

Only needed if you want the tunnel to work while fully logged out. Move the tunnel plist from per-user LaunchAgents to system-wide LaunchDaemons:

```bash
# Unload the user-level agent first
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Move to system location and fix ownership
sudo mv ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/xyz.otherdoor.tunnel.plist
sudo chmod 644 /Library/LaunchDaemons/xyz.otherdoor.tunnel.plist
```

Before loading, edit the plist and:

1. Add a `UserName` key so the daemon runs as you (not root), otherwise `~` paths break:
   ```xml
   <key>UserName</key>
   <string>drew</string>
   ```
2. Replace any `~/...` paths in `StandardOutPath`, `StandardErrorPath`, and `WorkingDirectory` with absolute `/Users/drew/...` paths — LaunchDaemons don't expand `~`.

Then load:

```bash
sudo launchctl load /Library/LaunchDaemons/xyz.otherdoor.tunnel.plist
```

All subsequent `launchctl` commands for the tunnel now need `sudo` and the `system/` target instead of `gui/$(id -u)/`:

```bash
sudo launchctl kickstart -k system/xyz.otherdoor.tunnel
sudo launchctl list | grep otherdoor
```
