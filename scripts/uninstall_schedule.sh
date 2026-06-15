#!/bin/zsh
# Remove the launchd watch job.
set -euo pipefail
LABEL="com.polyglot.watch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
rm -f "$PLIST"
echo "Removed ${LABEL}."
