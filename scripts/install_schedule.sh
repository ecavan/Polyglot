#!/bin/zsh
# Install (or refresh) the launchd job that runs `polyglot watch` every 30 minutes.
# launchd is chosen over cron on a laptop because it manages a per-user agent cleanly.
# NOTE: with StartInterval, a firing that lands while the Mac is asleep is dropped and the
# next one happens interval-seconds after wake (it does not "catch up"); that's fine for a
# podcast/video pull. The initial run is triggered explicitly below, not via RunAtLoad, so
# logging in never kicks off a heavy dub.
#
# Usage:  scripts/install_schedule.sh [interval_seconds]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
LABEL="com.polyglot.watch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
INTERVAL="${1:-1800}"
LOGDIR="$HOME/Library/Logs/Polyglot"
UV="$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")"

mkdir -p "$HOME/Library/LaunchAgents" "$LOGDIR"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>            <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/zsh</string>
        <string>${REPO}/scripts/run_watch.sh</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>POLYGLOT_REPO</key> <string>${REPO}</string>
        <key>POLYGLOT_UV</key>   <string>${UV}</string>
    </dict>
    <key>StartInterval</key>    <integer>${INTERVAL}</integer>
    <key>RunAtLoad</key>        <false/>
    <key>StandardOutPath</key>  <string>${LOGDIR}/watch.log</string>
    <key>StandardErrorPath</key><string>${LOGDIR}/watch.log</string>
    <key>LowPriorityIO</key>    <true/>
</dict>
</plist>
PLIST

UID_NUM="$(id -u)"
launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl enable "gui/${UID_NUM}/${LABEL}"     # enable BEFORE bootstrap (a prior disable persists)
launchctl bootstrap "gui/${UID_NUM}" "$PLIST"
launchctl kickstart "gui/${UID_NUM}/${LABEL}"  # do the first run now (RunAtLoad is off)

echo "Installed ${LABEL} (every ${INTERVAL}s)."
echo "  plist : ${PLIST}"
echo "  log   : ${LOGDIR}/watch.log"
echo "Verify : launchctl list | grep polyglot"
echo "Run now: launchctl kickstart -k gui/${UID_NUM}/${LABEL}"
