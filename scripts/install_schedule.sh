#!/bin/zsh
# Install (or refresh) the launchd job that runs `polyglot watch` every 30 minutes.
# launchd is preferred over cron on a laptop: if the Mac was asleep when the interval
# elapsed, the job runs on wake instead of being silently skipped.
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
    <key>RunAtLoad</key>        <true/>
    <key>StandardOutPath</key>  <string>${LOGDIR}/watch.log</string>
    <key>StandardErrorPath</key><string>${LOGDIR}/watch.log</string>
    <key>ProcessType</key>      <string>Background</string>
    <key>LowPriorityIO</key>    <true/>
    <key>Nice</key>             <integer>5</integer>
</dict>
</plist>
PLIST

UID_NUM="$(id -u)"
launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID_NUM}" "$PLIST"
launchctl enable "gui/${UID_NUM}/${LABEL}"

echo "Installed ${LABEL} (every ${INTERVAL}s)."
echo "  plist : ${PLIST}"
echo "  log   : ${LOGDIR}/watch.log"
echo "Verify : launchctl list | grep polyglot"
echo "Run now: launchctl kickstart -k gui/${UID_NUM}/${LABEL}"
