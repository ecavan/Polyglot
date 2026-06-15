#!/bin/zsh
# Wrapper invoked by launchd (which has no shell profile / PATH). Runs one watch pass.
set -euo pipefail

REPO="${POLYGLOT_REPO:-$(cd "$(dirname "$0")/.." && pwd -P)}"
cd "$REPO"

# launchd has no shell profile, so load secrets (e.g. ANTHROPIC_API_KEY for Claude
# translation) from a file the user creates. Without it, translation falls back to local.
[[ -f "$HOME/.config/polyglot/env" ]] && source "$HOME/.config/polyglot/env"

# Find uv: prefer an explicit path (only if it actually exists), then PATH, then the
# usual install locations. A stale baked-in POLYGLOT_UV must not be fatal when uv exists.
UV="${POLYGLOT_UV:-}"
[[ -n "$UV" && ! -x "$UV" ]] && UV=""
if [[ -z "$UV" ]]; then
  UV="$(command -v uv 2>/dev/null || true)"
fi
for cand in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
  [[ -z "$UV" && -x "$cand" ]] && UV="$cand"
done
if [[ ! -x "$UV" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: uv not found (set POLYGLOT_UV)" >&2
  exit 127
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') starting watch pass (repo=$REPO)"
exec "$UV" run polyglot watch
