# Hosting Setup — other-door.xyz

This documents how the public-facing version of the story app is hosted for a small group of friends.

---

## Architecture

- **App:** Flask server running in `SIMPLE_MODE` (no TTS, no Japanese, no indexing)
- **LLM:** Ollama running locally on the Mac Mini M4 with the Mistral Nemo 12B model
- **Tunnel:** Cloudflare Tunnel routes `https://other-door.xyz` → `localhost:5001`
- **Auth:** HTTP Basic Auth — username: anything, password: `drew`
- **Personal instance:** Still runs separately on port 5000 with full features (`python server.py`)

---

## How It Works

Two launchd services start automatically on login:

| Service | What it does |
|---|---|
| `xyz.otherdoor.server` | Runs `server.py` on port 5001 with `SIMPLE_MODE=1` |
| `xyz.otherdoor.tunnel` | Runs `cloudflared tunnel run other-door` |

Logs are written to:
- `~/Library/Logs/otherdoor-server.log`
- `~/Library/Logs/otherdoor-tunnel.log`

---

## SIMPLE_MODE

`server.py` reads the `SIMPLE_MODE` environment variable at startup. When set to `1`:

- `/tts`, `/kokoro_test`, `/tts_preview` → 404 (no TTS)
- `/analyze`, `/index_stories`, `/translate`, `/check-translation` → 404
- `?lang=ja` is ignored — always serves English
- HTTP Basic Auth is enforced on all story routes (password via `STORY_PASSWORD` env var)

Personal use (port 5000) is completely unaffected — just run `python server.py` as normal.

---

## Starting / Stopping Manually

```bash
# Start
launchctl load ~/Library/LaunchAgents/xyz.otherdoor.server.plist
launchctl load ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Stop
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.server.plist
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Restart a service
launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.server
launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.tunnel

# Check status
launchctl list | grep otherdoor

# Watch logs
tail -f ~/Library/Logs/otherdoor-server.log
tail -f ~/Library/Logs/otherdoor-tunnel.log
```

---

## Changing the Password

Edit the plist and reload:

```bash
nano ~/Library/LaunchAgents/xyz.otherdoor.server.plist
# Change the STORY_PASSWORD value

launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.server
```

---

## Completely Removing the Hosting Setup

### 1. Stop and remove launchd services

```bash
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.server.plist
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist
rm ~/Library/LaunchAgents/xyz.otherdoor.server.plist
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
rm ~/Library/Logs/otherdoor-server.log
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
| Target | `http://localhost:5001` |

---

## Troubleshooting

**Site not loading:**
1. Check services are running: `launchctl list | grep otherdoor`
2. Check server is up: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/` (expect `401`)
3. Check tunnel logs: `tail -f ~/Library/Logs/otherdoor-tunnel.log` (look for `Registered tunnel connection`)

**Server not starting (exit code 1):**
- Port 5001 may already be in use: `lsof -i :5001`
- Restart: `launchctl kickstart -k gui/$(id -u)/xyz.otherdoor.server`

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

### Fix 2 — run as LaunchDaemons so services survive logout (optional)

Only needed if you want the host to work while fully logged out. Move the plists from per-user LaunchAgents to system-wide LaunchDaemons:

```bash
# Unload the user-level agents first
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.server.plist
launchctl unload ~/Library/LaunchAgents/xyz.otherdoor.tunnel.plist

# Move to system location and fix ownership
sudo mv ~/Library/LaunchAgents/xyz.otherdoor.{server,tunnel}.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/xyz.otherdoor.*.plist
sudo chmod 644 /Library/LaunchDaemons/xyz.otherdoor.*.plist
```

Before loading, edit each plist and:

1. Add a `UserName` key so the daemon runs as you (not root), otherwise Ollama, the venv, and `~` paths break:
   ```xml
   <key>UserName</key>
   <string>drew</string>
   ```
2. Replace any `~/...` paths in `StandardOutPath`, `StandardErrorPath`, `WorkingDirectory`, and `ProgramArguments` with absolute `/Users/drew/...` paths — LaunchDaemons don't expand `~`.

Then load:

```bash
sudo launchctl load /Library/LaunchDaemons/xyz.otherdoor.server.plist
sudo launchctl load /Library/LaunchDaemons/xyz.otherdoor.tunnel.plist
```

All `launchctl` commands in this doc now need `sudo` and the `system/` target instead of `gui/$(id -u)/`:

```bash
sudo launchctl kickstart -k system/xyz.otherdoor.server
sudo launchctl kickstart -k system/xyz.otherdoor.tunnel
sudo launchctl list | grep otherdoor
```
