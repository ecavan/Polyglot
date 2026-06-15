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

## Go live (one-time)

```bash
scripts/setup_jellyfin.sh        # installs Jellyfin, makes the library folders, prints your
                                 # Roku/iPhone server URL + the wizard steps
scripts/install_schedule.sh      # launchd job: runs `polyglot watch` every 30 min
```

Then enable a show in `config/shows.toml` (`enabled = true`). The first scheduled pass dubs the
newest ~10 items (slow); after that it just keeps up and purges anything past the retention
window. Logs: `~/Library/Logs/Polyglot/watch.log`. Remove the schedule with
`scripts/uninstall_schedule.sh`.

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

```
~/PolyglotLibrary/
  Videos/<show>/<title> [id].mp4   + .srt
  Podcasts/<show>/<title> [id].mp3 + .srt   (+ <title> [id].mp4 for the TV)
```

Retention keeps the newest `keep` items per show and purges anything older than
`max_age_days` (by the episode's real air date), deleting the files but remembering it already
processed them so they're never re-dubbed.
