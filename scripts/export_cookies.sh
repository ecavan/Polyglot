#!/usr/bin/env bash
# Export YouTube cookies from Chrome to a static cookies.txt ONCE, so the unattended worker
# never needs a live-browser Keychain prompt (which hangs a detached run forever).
#
# Run this in the FOREGROUND and click "Always Allow" on the one macOS Keychain prompt.
# After it succeeds, settings.toml's [download] cookies_file is used automatically.
#
#   bash scripts/export_cookies.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-$HOME/.config/polyglot/youtube-cookies.txt}"
mkdir -p "$(dirname "$OUT")"

echo "Exporting Chrome's YouTube cookies -> $OUT"
echo "If a macOS Keychain prompt appears, click 'Always Allow'."
uv run yt-dlp --no-warnings --cookies-from-browser chrome \
  --cookies "$OUT" --skip-download \
  "https://www.youtube.com/watch?v=Ln60BiVAplY" >/dev/null

if [ -s "$OUT" ]; then
  chmod 600 "$OUT"
  echo "OK: wrote $(wc -l < "$OUT" | tr -d ' ') cookie lines to $OUT"
else
  echo "FAILED: no cookies written. Make sure you're logged into YouTube in Chrome." >&2
  exit 1
fi
