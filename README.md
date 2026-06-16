# Polyglot

Local-first pipeline that dubs English podcasts and YouTube videos into **Québécois French**
with bilingual transcripts, and serves them to your **Roku + iPhone via Jellyfin** — fully on
your own machine, nothing uploaded.

## How it works

```
download → Demucs (split voice / music bed) → mlx-whisper transcribe → SpeechBrain diarize
        → merge segments → translate → Orpheus 3B French TTS → mix over music bed → publish
```

- **Videos** keep the picture with the side-by-side transcript **burned in** (French in a blue
  box, English in a red box) and the French dub over the original music/SFX.
- **Podcasts** publish two files: an `.mp3` (+ `.srt`) for the phone, and a static-cover `.mp4`
  with the transcript burned in for the TV (Jellyfin's Roku client handles subtitles on
  audio-only items poorly).
- The unattended `watch` loop pulls new items, dubs them, publishes to the library, and prunes
  old ones — idempotent (never re-dubs) and self-cleaning.

## Requirements

- Apple Silicon Mac (tuned for an **M3 Pro / 18 GB**), macOS
- Python 3.12 via [`uv`](https://docs.astral.sh/uv/), `ffmpeg` (with `librubberband` + `libass`)
- The Orpheus French TTS model at `~/.cache/polyglot/orpheus/Orpheus-3b-French-FT-Q8_0.gguf`
  (`lex-au/Orpheus-3b-French-FT-Q8_0.gguf`) plus the `hubertsiuzdak/snac_24khz` decoder

```bash
uv sync          # install deps into .venv
```

**Translation** defaults to **Claude** (best Québécois) — provide `ANTHROPIC_API_KEY`. Without
it, translation automatically falls back to the local Qwen model (free, offline), so the
pipeline always works. Switch to local-only with `translate_backend = "mlx"` in
`config/settings.toml`. Compare the options on a real episode with
`uv run python scripts/compare_translation.py`.

Put the key where **both** your shell and the scheduled `watch` job can read it (launchd does
not load your shell profile):

```bash
mkdir -p ~/.config/polyglot
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.config/polyglot/env   # watch loop sources this
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc                 # interactive `uv run` commands
```

## Go live (one-time)

```bash
scripts/setup_jellyfin.sh        # installs Jellyfin, makes the library folders, prints your
                                 # Roku/iPhone server URL + the wizard steps
scripts/install_schedule.sh      # launchd job: runs `polyglot watch` once a day at midnight
                                 # (pass an hour to change, e.g. `install_schedule.sh 3` = 3 AM)
```

Then enable a show in `config/shows.toml` (`enabled = true`) and run the first populate yourself
when convenient (`uv run polyglot watch`) — the first pass dubs the newest ~10 items and is slow.
After that the daily run just keeps up and purges anything past the retention window. It uses
`StartCalendarInterval`, so if the Mac is asleep at the scheduled time the run **catches up on
wake**. Logs: `~/Library/Logs/Polyglot/watch.log`. Remove with `scripts/uninstall_schedule.sh`.

## Commands

```bash
uv run polyglot show <show_id>                 # print the resolved config for a show
uv run polyglot run <show_id> [--latest|--url U|--file F] [--clip-seconds N]   # one podcast
uv run polyglot video <youtube_url> [--clip-seconds N] [--speakers N]          # one video
uv run polyglot watch                          # one pass over all enabled shows (what launchd runs)
uv run polyglot cleanup                        # purge the transient cache/
```

## Configuration

- `config/settings.toml` — models, TTS/Orpheus, separation, mix, **`[library]` path**,
  **`[retention]` keep / max_age_days**, video length cap, speeds.
- `config/shows.toml` — the shows/channels to dub and which are enabled.
- `config/prompts/<lang>.txt` — the translation system prompt (Québécois rules).
- `voices/` — reference clips for voice cloning (optional; built-in Orpheus voices are the default).

## Library layout (what Jellyfin reads)

Three Jellyfin libraries, because Jellyfin only surfaces audio in a *Music* library:

```
~/PolyglotLibrary/
  Videos/<show>/<title> [id].mp4                 -> "Home videos and photos" library  (burned-in transcript)
  Podcasts/<show>/<title> [id] (TV).mp4          -> "Home videos and photos" library  (podcast read-along on TV/phone)
  PodcastAudio/<show>/<title> [id].mp3 + .lrc    -> "Music" library  (screen-off phone listening + synced lyrics)
```

- **Video** = transcript burned in (no sidecar — a sidecar would render doubled).
- **Podcast audio** (`.mp3`) lives in a Music library for screen-off, offline phone listening
  (Finamp or the Jellyfin app); the `.lrc` gives a scrolling read-along transcript while it plays.
- **Podcast video** (`(TV).mp4`) is the burned-in read-along for when you want to watch/read.

Retention keeps the newest `keep` items per show and purges anything older than
`max_age_days` (by the episode's real air date), deleting the files but remembering it already
processed them so they're never re-dubbed.
