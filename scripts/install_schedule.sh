#!/bin/zsh
# Install (or refresh) the launchd job that runs `polyglot watch` ONCE A DAY.
# Uses StartCalendarInterval, which (unlike StartInterval) FIRES ON WAKE if the Mac was
# asleep at the scheduled time — so a laptop that's closed at midnight still catches up.
# RunAtLoad is off, so logging in never kicks off a heavy dub; you trigger the first
# populate yourself (printed at the end).
#
# Usage:  scripts/install_schedule.sh [hour 0-23]      default 0 (midnight); e.g. `... 3` = 3 AM
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
LABEL="com.polyglot.watch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
HOUR="${1:-0}"
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
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>         <integer>${HOUR}</integer>
        <key>Minute</key>       <integer>0</integer>
    </dict>
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

echo "Installed ${LABEL}: runs daily at ${HOUR}:00 (catches up on wake if the Mac was asleep)."
echo "  plist : ${PLIST}"
echo "  log   : ${LOGDIR}/watch.log"
echo "Verify       : launchctl list | grep polyglot"
echo "Populate now : uv run polyglot watch        (or: launchctl kickstart -k gui/${UID_NUM}/${LABEL})"
