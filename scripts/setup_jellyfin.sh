#!/bin/zsh
# Install + start Jellyfin and prepare the Polyglot library folders, then print the
# exact Roku/iPhone server URL and the one-time library-setup steps.
#
# Jellyfin's first-run wizard (create admin user) is interactive and cannot be safely
# automated, so this script gets you to the doorstep and tells you the few clicks left.
set -euo pipefail

LIB="${POLYGLOT_LIBRARY:-$HOME/PolyglotLibrary}"

echo "==> Library folders"
mkdir -p "$LIB/Videos" "$LIB/Podcasts"
echo "    $LIB/Videos"
echo "    $LIB/Podcasts"

echo "==> Jellyfin"
if [[ -d "/Applications/Jellyfin.app" ]]; then
  echo "    already installed."
elif command -v brew >/dev/null 2>&1; then
  echo "    installing via Homebrew (this prompts for your password)..."
  # don't let a non-zero brew result (e.g. already-installed, or a cask warning) abort the
  # script before it prints the URL + wizard steps below.
  brew install --cask jellyfin || echo "    (brew returned non-zero; continuing — check above if Jellyfin didn't install)"
else
  echo "    Homebrew not found. Install it from https://brew.sh then re-run, or"
  echo "    download Jellyfin from https://jellyfin.org/downloads/"
  exit 1
fi

open -a Jellyfin || true
sleep 3

IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '<your-mac-LAN-ip>')"

cat <<DONE

==> Jellyfin is starting. Finish setup in your browser:

    http://localhost:8096

    One-time wizard:
      1. Create your admin user.
      2. Add Media Library -> Content type: "Home videos and photos"
         - Display name: Polyglot Videos
         - Folder: $LIB/Videos
      3. Add Media Library -> Content type: "Home videos and photos"
         - Display name: Polyglot Podcasts
         - Folder: $LIB/Podcasts
      (Home videos avoids internet metadata matching and auto-loads the
       same-basename .srt as a selectable subtitle track.)

==> Watch on Roku / iPhone (must be on the SAME Wi-Fi as this Mac):

    Server URL:  http://${IP}:8096

      Roku   : Channel Store -> install "Jellyfin" -> Add Server -> the URL above
      iPhone : App Store -> "Jellyfin Mobile" -> Add Server -> the URL above

    Away from home later? Put both devices on Tailscale (https://tailscale.com)
    and use this Mac's Tailscale IP instead of the LAN IP.

==> New dubs land automatically once you install the schedule:

    scripts/install_schedule.sh
DONE
