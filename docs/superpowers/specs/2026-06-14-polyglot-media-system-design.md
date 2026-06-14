# Polyglot Media System — Design Spec

**Date:** 2026-06-14
**Status:** Approved for planning (pending user review)
**Author:** Elijah Cavan (design), refined collaboratively

A personal, local-first pipeline that turns English podcasts / YouTube into a target
language (French / Montréal first), delivered as a podcast RSS feed + bilingual
subtitles. Built to be configurable across languages, shows, and voices.

This spec is the source of truth. It supersedes the original planning notes by
incorporating the actual machine environment (Apple Silicon, Python 3.12, tight disk)
and **verified** library APIs for the chosen backends (MLX for translation +
transcription, XTTS for synthesis). Where the original notes assumed Ollama +
faster-whisper, this spec records the new defaults and keeps the originals as
pluggable fallbacks.

---

## 1. Goal

Subscribe in Apple Podcasts to English shows, automatically re-voiced into spoken
Québécois French, with bilingual (EN+FR) subtitles whose timing matches the *generated*
French audio. Generation happens locally; finished audio is hosted on a free static
bucket so Apple can fetch the feed.

---

## 2. Design principles

1. **Config-driven, not hard-coded.** Behaviour comes from `config/`. Code is generic.
2. **A language = a prompt file + a language code.** `target_lang` selects
   `config/prompts/{lang}.txt` and the TTS `language` parameter. Nothing language-specific
   is hard-coded.
3. **A show = a config block.** Adding a show never touches code.
4. **A voice = a wav (or folder of wavs).** Cloning is opt-in per show.
5. **Local compute, free hosting.** Generate on-machine (MLX Whisper + MLX LLM + XTTS);
   host finished audio on Cloudflare R2 (S3-compatible) so Apple can fetch the feed.
6. **Idempotent + episode-isolated.** Each episode is keyed by its GUID; one failing
   episode never breaks a run.
7. **Subtitle timing is derived from the *generated* audio**, not the English timestamps
   (the translated language is longer; see §12 Gotchas).
8. **Pluggable backends.** Transcription and translation each have a backend abstraction.
   Defaults are MLX (GPU on Apple Silicon); Ollama and faster-whisper remain selectable
   via config with no code change.

---

## 3. Environment & key decisions

The target machine is an **Apple Silicon Mac**, currently **97% full (~15 GB free)**.
These decisions are driven by that reality and by API verification done on 2026-06-14.

| Decision | Choice | Why |
|---|---|---|
| Python | **3.12**, managed by `uv` | ML stack (torch, coqui-tts, mlx-whisper) lacks 3.14 wheels; mlx-whisper breaks on 3.13; 3.12 is the verified sweet spot. `uv` already installed; 3.12.9 already on disk. |
| Project tooling | **`uv`** (pyproject + `uv.lock`, run via `uv run polyglot …`) | Already installed; reproducible lockfile; manages the pinned 3.12. |
| Transcription | **mlx-whisper** (default) → `faster-whisper` (fallback) | mlx-whisper is GPU-accelerated on Apple Silicon (~2× whisper.cpp); faster-whisper is CPU-only on Mac and its wheels claim 3.9–3.11 (best-effort on 3.12). |
| Whisper model | `mlx-community/whisper-large-v3-turbo` (1.61 GB) | Best speed/accuracy for English podcasts; 8× faster than large-v3 at similar quality. |
| Translation | **MLX in-process** with `mlx-community/Qwen2.5-7B-Instruct-4bit` (default) → Ollama (fallback) | Reuses the AuthorGPT MLX toolchain; 4-bit 7B ≈ 5.6 GB RAM, 15–25 tok/s on GPU; clean instruct output (no reasoning tags). |
| TTS | **XTTS v2** via `coqui-tts`, **`device='cpu'`** | XTTS supports fr/it; runs on 3.12. **MPS hangs the system (wontfix)** → CPU only. Slowest stage but acceptable for batch. |
| Hosting | **Cloudflare R2** (boto3, S3-compatible) | Free tier, S3 API; public via r2.dev or custom domain. |

### 3.1 Disk reclamation (do this first, it's a net win)

The "Qwen model already pulled" for AuthorGPT is a **broken 81% download**: all four
`.safetensors` shards are `.incomplete` blobs (~12.35 GB of the 15.23 GB full model) in
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/`. It cannot load and
is pure waste.

**Plan:** delete those `.incomplete` blobs → reclaim ~12.35 GB → download the clean 4-bit
MLX model. Net disk impact is **positive (~+8 GB free)**.

| Step | Δ disk | Running free (start ≈ 15 GB) |
|---|---|---|
| Delete broken Qwen `.incomplete` blobs | **+12.35 GB** | ~27.4 GB |
| Download `mlx-community/Qwen2.5-7B-Instruct-4bit` | −4.28 GB | ~23.1 GB |
| Download `mlx-community/whisper-large-v3-turbo` | −1.61 GB | ~21.5 GB |
| XTTS v2 first run (`~/.local/share/tts/`) | −1.87 GB | ~19.6 GB |
| Python venv + deps (torch, mlx, etc.) | ≈ −3–4 GB | ~16 GB |

Cache of intermediate audio is bounded by the Phase 4 storage policy.

---

## 4. Repository layout

```
Polyglot/
├── config/
│   ├── settings.toml          # global: models, backends, paths, hosting, defaults
│   ├── shows.toml             # per-show definitions (source, lang, voice)
│   └── prompts/
│       └── fr.txt             # French (Montréal) translation system prompt
├── voices/
│   ├── fr_montreal.wav        # single reference clip for cloning, OR…
│   └── fr_montreal/           # …a folder of clips (richer reference)
├── src/polyglot/
│   ├── __init__.py
│   ├── config.py              # load + validate config; build a JobSpec per show
│   ├── feeds.py               # list episodes from an RSS feed / YouTube source
│   ├── download.py            # fetch + (optionally) trim audio → 16 kHz mono wav
│   ├── transcribe.py          # backend: mlx-whisper | faster-whisper → segments
│   ├── translate.py           # backend: mlx | ollama → adds 'translation'
│   ├── tts.py                 # backend: xtts (cpu) | piper; per-segment synth
│   ├── assemble.py            # concat segment audio + measure durations → mp3 + timeline
│   ├── subtitles.py           # write synced .srt/.vtt (bilingual + target-only)
│   ├── publish_rss.py         # feedgen → per-show RSS; upload audio to R2
│   ├── storage.py             # cache + cleanup policy
│   ├── state.py               # processed-episode ledger (idempotency)
│   └── pipeline.py            # orchestrates one episode end-to-end
├── cli.py                     # entrypoints (console_script: polyglot)
├── state/processed.json
├── cache/                     # transient; cleaned by storage policy
├── output/
│   ├── audio/<show_id>/<ep>.mp3
│   ├── subs/<show_id>/<ep>.{srt,vtt}
│   └── rss/<show_id>.xml
├── docs/superpowers/specs/    # this spec + future ones
├── pyproject.toml
└── README.md
```

`pyproject.toml` exposes a console entry point: `polyglot = "polyglot.cli:main"`, run as
`uv run polyglot …`.

---

## 5. The data contract (one Segment shape, used everywhere)

Every stage reads and enriches a list of these dicts. Keep the key generic
(`translation`, **not** `fr`) so it works for any language.

```python
Segment = {
    "index":      int,          # order
    "start":      float,        # ORIGINAL (source) start, seconds
    "end":        float,        # ORIGINAL end
    "text":       str,          # source language (English)
    "translation":str | None,   # target-language text  (added by translate.py)
    "speaker":    str | None,   # e.g. "SPEAKER_00" (added by whisperx if diarize=on)
    "audio_path": str | None,   # per-segment synthesized clip (added by tts.py)
    "audio_dur":  float | None, # measured duration of that clip (added by tts.py)
}
```

The **dubbed timeline** (cumulative `audio_dur + gap`) is produced by `assemble.py` and is
the only timeline subtitles are written against.

---

## 6. Configuration

### `config/settings.toml`

```toml
[models]
# --- transcription ---
transcribe_backend = "mlx-whisper"                       # "mlx-whisper" | "faster-whisper"
mlx_whisper_repo   = "mlx-community/whisper-large-v3-turbo"
faster_whisper     = "medium.en"                          # used only by faster-whisper backend

# --- translation ---
translate_backend  = "mlx"                                # "mlx" | "ollama"
mlx_llm_repo       = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ollama_model       = "mistral"                            # used only by ollama backend
ollama_url         = "http://localhost:11434/api/chat"

# --- tts ---
tts_backend        = "xtts"                               # "xtts" | "piper"
tts_device         = "cpu"                                # XTTS on Apple Silicon: MUST be "cpu"

[paths]
cache   = "cache"
output  = "output"
voices  = "voices"
prompts = "config/prompts"
state   = "state/processed.json"

[hosting]
type            = "r2"                                    # "r2" | "local"
public_base_url = "https://media.example.com"             # r2.dev domain or custom CNAME
bucket          = "polyglot-media"
# credentials come from env vars, never this file:
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY

[defaults]
clip_seconds = 0        # 0 = full episode; >0 = trim (handy for testing)
diarize      = false
temperature  = 0.3      # translation sampler temp (0.0 = greedy/deterministic)
max_tokens   = 512      # per-segment translation cap
gap_ms       = 200      # silence inserted between synthesized segments
```

### `config/shows.toml`

```toml
[[show]]
id          = "pti-fr"
title       = "PTI (Français)"
source      = "https://feeds.megaphone.fm/pti"   # RSS enclosure feed
source_type = "rss"                              # "rss" | "youtube"
target_lang = "fr"            # → prompts/fr.txt  AND  TTS language="fr"
voice       = "fr_montreal"   # → voices/fr_montreal.wav or voices/fr_montreal/ (omit → built-in voice)
enabled     = true

# Adding Italian later = this block + a config/prompts/it.txt. No code change.
# [[show]]
# id = "thedaily-it"; title = "The Daily (Italiano)"
# source = "..."; source_type = "rss"
# target_lang = "it"; voice = "it_rai"; enabled = true
```

### `config/prompts/fr.txt` (canonical content)

```
Tu es un traducteur professionnel anglais -> français.
Traduis le texte en français parlé, naturel et idiomatique, tel qu'on le parle à
Montréal (registre québécois courant, pas joual extrême).

Règles:
- Traduis le SENS, jamais mot à mot.
- Utilise le vocabulaire québécois quand c'est naturel (char, courriel, magasiner,
  c'est correct, faque, en tout cas, pantoute).
- Garde le ton conversationnel de l'original (deux gars qui jasent de sports, etc.).
- Conserve les noms propres et les termes techniques tels quels.
- Réponds UNIQUEMENT avec la traduction française. Aucune note, aucune explication,
  aucun guillemet.
```

A new language is the same file shape written in that language. `translate.py` just loads
`prompts/{target_lang}.txt`.

---

## 7. Module specifications

Code snippets below use **verified** APIs (checked against current package versions on
2026-06-14). They are the intended calls, not pseudocode.

### `config.py`
- **Responsibility:** load + validate `settings.toml` and `shows.toml`; resolve a `JobSpec`.
- **Interface:**
  ```python
  @dataclass
  class JobSpec:
      show_id: str; title: str; source: str; source_type: str
      target_lang: str
      prompt_path: Path        # prompts/{target_lang}.txt  (error if missing)
      voice_refs: list[Path]   # [] if built-in; else the wav(s) for cloning
      settings: Settings       # resolved global settings attached

  def load_settings() -> Settings: ...
  def load_shows() -> list[ShowConfig]: ...
  def build_job(show_id: str) -> JobSpec: ...   # validates prompt + voice exist
  ```
- **Voice resolution:** if `voices/<voice>.wav` exists → `[that file]`; elif `voices/<voice>/`
  is a dir → `sorted(glob("*.wav"))`; else `voice_refs=[]` (built-in speaker).
- **TOML parsing:** Python 3.12 has `tomllib` in the stdlib — use it (no `tomli` dependency).
- **Acceptance-relevant:** raises a clear error if `prompts/{lang}.txt` is missing, and if a
  named `voice` resolves to neither a file nor a dir.

### `feeds.py`
- **Responsibility:** list available episodes for a source.
- **Interface:** `list_episodes(job: JobSpec, limit: int | None) -> list[Episode]`
  where `Episode = {guid, title, published, media_url}`.
- **RSS (`feedparser`):**
  ```python
  import feedparser
  d = feedparser.parse(job.source)
  for e in d.entries[:limit]:
      guid      = e.get("id") or e.get("guid") or e.enclosures[0].href
      media_url = e.enclosures[0].href           # first audio enclosure
      published = e.get("published")             # RFC822 str; e.published_parsed = struct_time
  ```
- **YouTube (`yt_dlp`):**
  ```python
  from yt_dlp import YoutubeDL
  with YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
      info = ydl.extract_info(job.source, download=False)
  # info["entries"] → each has "id" (guid), "title", "url"
  ```

### `download.py`
- **Responsibility:** fetch one episode's audio to `cache/`, optionally trim, return a
  **16 kHz mono wav** (the format whisper wants).
- **Interface:** `fetch_audio(media_url, out_dir, clip_seconds=0) -> Path`.
- **RSS enclosure:** `requests.get(url, stream=True)` → write mp3 to cache.
- **YouTube:** `yt-dlp -x --audio-format mp3` (CLI) or the `YoutubeDL` postprocessor.
- **Normalize with ffmpeg:**
  ```bash
  ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav            # full
  ffmpeg -i in.mp3 -t {clip_seconds} -ac 1 -ar 16000 out.wav   # trimmed (-t = duration secs)
  ```

### `transcribe.py`  *(pluggable; default mlx-whisper)*
- **Responsibility:** source 16 kHz wav → `list[Segment]` (English text + timestamps).
- **Interface:** `transcribe(wav: Path, cfg) -> list[Segment]` (dispatches on `transcribe_backend`).
- **mlx-whisper backend (default, GPU on Apple Silicon):**
  ```python
  import mlx_whisper
  result = mlx_whisper.transcribe(
      str(wav),
      path_or_hf_repo=cfg.mlx_whisper_repo,   # "mlx-community/whisper-large-v3-turbo"
      language="en",                          # force English (skip detection)
  )
  segments = [
      {"index": i, "start": s["start"], "end": s["end"], "text": s["text"].strip(),
       "translation": None, "speaker": None, "audio_path": None, "audio_dur": None}
      for i, s in enumerate(result["segments"])
  ]
  ```
  `result` is a dict with `text`, `segments` (each has `id, start, end, text, …`), `language`.
- **faster-whisper backend (fallback, CPU on Mac):**
  ```python
  from faster_whisper import WhisperModel
  model = WhisperModel(cfg.faster_whisper, device="cpu", compute_type="int8")
  segments_gen, _info = model.transcribe(str(wav), language="en", beam_size=5)
  segments = list(segments_gen)   # generator — must be drained to execute
  # each seg has .start, .end, .text
  ```
- **Note on timestamps:** source timestamps are used only for ordering and reference. The
  subtitle timeline is rebuilt from generated audio durations (§9, §12). Segment-level
  granularity (~0.2 s) is sufficient; `word_timestamps` is not required for Phase 1.
- **Diarization (Phase 5 only):** swap to whisperx + pyannote when `diarize=true`; populate
  `Segment["speaker"]`. Needs a Hugging Face token.

### `translate.py`  *(pluggable; default mlx)*
- **Responsibility:** add `translation` to each segment using the language's prompt.
- **Interface:** `translate(segments, job: JobSpec, cfg) -> list[Segment]` (dispatches on
  `translate_backend`).
- **mlx backend (default, in-process, GPU):**
  ```python
  from mlx_lm import load, generate
  from mlx_lm.sample_utils import make_sampler

  system  = job.prompt_path.read_text(encoding="utf-8")
  model, tokenizer = load(cfg.mlx_llm_repo)            # load ONCE; MLX is not thread-safe
  sampler = make_sampler(temp=cfg.temperature)         # temp=0.0 ⇒ greedy/deterministic

  for seg in segments:
      messages = [{"role": "system", "content": system},
                  {"role": "user",   "content": seg["text"]}]
      prompt = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True)
      seg["translation"] = generate(
          model, tokenizer, prompt=prompt,
          max_tokens=cfg.max_tokens, sampler=sampler, verbose=False).strip()
  ```
- **ollama backend (fallback):**
  ```python
  import requests
  r = requests.post(cfg.ollama_url, json={
      "model": cfg.ollama_model, "stream": False,
      "options": {"temperature": cfg.temperature},
      "messages": [{"role": "system", "content": system},
                   {"role": "user",   "content": seg["text"]}],
  }, timeout=120)
  seg["translation"] = r.json()["message"]["content"].strip()
  ```
- **Quality nicety (recommended):** prefix the previous English line as context in the user
  turn so sentences split across segments translate coherently; keep the *response* to the
  current segment only.
- **Memory:** load the LLM, translate all segments, then **release it** (`del model; del
  tokenizer; gc.collect()`) before TTS so the 4-bit LLM (~5.6 GB) and torch/XTTS are not
  co-resident. See `pipeline.py`.

### `tts.py`  *(pluggable; default xtts, CPU)*
- **Responsibility:** synthesize each segment's `translation` to a wav; record duration.
- **Interface:** `synthesize(segments, job: JobSpec, cfg) -> list[Segment]`
  (fills `audio_path` + `audio_dur`).
- **XTTS backend (default).** Two compatibility shims are **required**:
  ```python
  import os
  os.environ.setdefault("COQUI_TOS_AGREED", "1")                 # else model load hangs
  os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1") # torch>=2.6 unpickling fix
  # (belt-and-suspenders alternative to the env var, after `import torch` + `from TTS...`:
  #  torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, ...]) )

  from TTS.api import TTS
  import soundfile as sf

  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(cfg.tts_device)  # "cpu"
  SR = 24000  # XTTS output sample rate

  for seg in segments:
      out = out_dir / f"seg_{seg['index']:04d}.wav"
      common = dict(text=seg["translation"], language=job.target_lang, split_sentences=False)
      if job.voice_refs:
          wav = tts.tts(**common, speaker_wav=[str(p) for p in job.voice_refs])  # CLONING
      else:
          wav = tts.tts(**common, speaker="Claribel Dervla")  # built-in; print(tts.speakers) to list
      sf.write(str(out), wav, SR, subtype="FLOAT")
      seg["audio_path"] = str(out)
      seg["audio_dur"]  = len(wav) / SR
  ```
  - **Apple Silicon:** `device="cpu"` is mandatory — `mps` hangs the system (upstream wontfix).
  - XTTS supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, **zh-cn**, ja, hu, ko, hi.
    So `target_lang="it"` works with no backend change. (Note `zh-cn`, not `zh`.)
  - `split_sentences=False` for coherent long-form; segments are already short.
  - Cloning: `speaker_wav` accepts a list; multiple refs are averaged. Use clean clips
    ≥6 s (10–30 s better).
- **Piper backend (fallback):** `piper -m voices/<lang>.onnx -f seg.wav` via subprocess
  (fast, lighter, robotic; no cloning).
- **Speaker→voice mapping (future multi-voice):** if a show defines a `speaker_map` and
  segments carry `speaker`, pick the matching voice ref per segment. Off by default.

### `assemble.py`
- **Responsibility:** concatenate per-segment audio (with `gap_ms` silence) into one episode
  track; encode mp3; produce the **dubbed timeline**.
- **Interface:** `assemble(segments, out_mp3: Path, cfg) -> EpisodeAudio`
  returning total duration, byte length (for the RSS enclosure), and `timeline`
  (list of `(start, end)` per segment on the dubbed clock).
- **Method:**
  ```python
  import numpy as np, soundfile as sf, subprocess
  SR, gap = 24000, np.zeros(int(cfg.gap_ms/1000*24000), dtype=np.float32)
  parts, timeline, t = [], [], 0.0
  for seg in segments:
      wav, _ = sf.read(seg["audio_path"], dtype="float32")
      timeline.append((t, t + seg["audio_dur"]))
      t += seg["audio_dur"] + cfg.gap_ms/1000
      parts.extend([wav, gap])
  full = np.concatenate(parts)
  sf.write(tmp_wav, full, SR, subtype="FLOAT")
  subprocess.run(["ffmpeg","-y","-i",tmp_wav,"-c:a","libmp3lame","-b:a","128k",str(out_mp3)])
  ```
  `byte_length = out_mp3.stat().st_size`; total duration = `len(full)/SR`.

### `subtitles.py`
- **Responsibility:** write synced subtitles against the **dubbed** timeline.
- **Interface:** `write_subs(segments, timeline, out_dir, show_id, ep_id) -> None`.
- **Outputs:** `*.srt` and `*.vtt`, in two flavours: **target-only** and **bilingual**
  (source line + target line per cue). Prefix `speaker` when present. Timestamps come
  entirely from `timeline` (never the source `start`/`end`).

### `state.py`
- **Responsibility:** idempotency ledger. `is_done(show_id, guid) -> bool`,
  `mark_done(show_id, guid, meta)`. Backed by `state/processed.json` (atomic write:
  temp file + `os.replace`).

### `publish_rss.py`
- **Responsibility:** upload audio to R2 and build the per-show podcast RSS.
- **Interface:** `publish(show_id, cfg) -> str` (returns the public RSS URL).
- **Upload (R2 via boto3, S3-compatible):**
  ```python
  import boto3, os
  s3 = boto3.client(
      "s3",
      endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
      region_name="auto",                     # R2 requires "auto"
  )
  s3.upload_file(local_mp3, cfg.bucket, key, ExtraArgs={"ContentType": "audio/mpeg"})
  public_url = f"{cfg.public_base_url}/{key}"
  ```
  - Do **not** set `AWS_DEFAULT_REGION` / `AWS_REGION` (causes R2 signature failures).
  - Public serving requires an R2 **public bucket (r2.dev)** or a **custom domain CNAME**;
    `public_base_url` points at whichever is configured.
- **feedgen:**
  ```python
  from feedgen.feed import FeedGenerator
  fg = FeedGenerator(); fg.load_extension("podcast")
  fg.title(show.title); fg.link(href=public_rss_url); fg.description(show.title)
  fg.language("fr")
  for ep in episodes_for(show_id):
      fe = fg.add_entry()
      fe.id(ep.guid); fe.title(ep.title)
      fe.pubDate(ep.published)                 # tz-aware datetime or RFC822 string
      fe.enclosure(ep.audio_public_url, str(ep.byte_length), "audio/mpeg")
  fg.rss_file(f"output/rss/{show_id}.xml", pretty=True)
  ```
  - For Apple validity, set channel-level itunes fields via the podcast extension
    (author, image, explicit, category) — minimally author + image.

### `pipeline.py`
- **Responsibility:** orchestrate one episode, **loading one heavy model at a time** to bound
  peak RAM:
  ```
  process_episode(job, episode, cfg):
      wav      = download.fetch_audio(...)
      segments = transcribe.transcribe(wav, cfg)     # mlx-whisper loads+frees
      segments = translate.translate(segments, ...)  # mlx LLM loads, then freed
      segments = tts.synthesize(segments, ...)        # XTTS (torch) loads after LLM freed
      audio    = assemble.assemble(segments, out_mp3, cfg)
      subtitles.write_subs(segments, audio.timeline, ...)
      state.mark_done(job.show_id, episode.guid, meta)
  ```
  Wrap in try/except so one bad episode is logged and skipped.

### `cli.py`
```
polyglot show <id>                         # print resolved JobSpec (no processing)
polyglot run  <id> [--latest | --url U | --file F] [--clip-seconds N]
polyglot watch [--once]                    # all enabled shows; process only new episodes
polyglot publish <id>                      # (re)build RSS + upload
polyglot cleanup [--days N]                # apply storage policy
```
Run via `uv run polyglot …`.

---

## 8. Build phases (each ends with an acceptance check)

Implement **one phase at a time**; don't start the next until the current passes.

### Phase 0 — Environment + scaffold + config
- Reclaim disk (§3.1): delete the broken Qwen `.incomplete` blobs.
- `uv init` the project pinned to **Python 3.12**; add deps; create the repo layout.
- Implement `config.py` (+ dataclasses), `settings.toml`, `shows.toml`, `prompts/fr.txt`.
- Pre-download the models (Qwen 4-bit, whisper-turbo) so Phase 1 isn't gated on network.

**Acceptance:** `uv run polyglot show pti-fr` prints the resolved JobSpec (source,
target_lang, prompt_path, voice_refs, resolved backends) and errors clearly if the prompt
or voice is missing.

### Phase 1 — Core single-episode pipeline (audio-only) ← the quality gate
`feeds`, `download`, `transcribe` (mlx-whisper), `translate` (mlx), `tts` (xtts/cpu),
`assemble`, `subtitles`, `pipeline`, wired for one show, run manually. Per-segment TTS +
derived subtitle timeline.

**Acceptance:** `uv run polyglot run pti-fr --latest --clip-seconds 180` produces
`output/audio/pti-fr/<ep>.mp3` + synced bilingual `.srt`. **You listen and judge: is the
French good enough to enjoy daily, and do the subtitles line up?** Everything after this is
plumbing — don't build it until this clears the bar.

### Phase 2 — Multi-language + voice cloning
Add `config/prompts/it.txt` and a show with `target_lang="it"`; wire `voice_refs` into XTTS
`speaker_wav` (single file and folder-of-clips).

**Acceptance:** the *same* pipeline produces Italian audio from only config additions, and a
show pointed at `voices/fr_montreal/` audibly uses the cloned voice.

### Phase 3 — Distribution (reaches Apple Podcasts)
`state.py`, `publish_rss.py`, R2 upload. Requires an R2 bucket + public domain configured
and the three `R2_*` env vars.

**Acceptance:** `uv run polyglot run pti-fr --latest && uv run polyglot publish pti-fr` →
the show appears in Apple Podcasts (added by URL) and plays the French episode.

### Phase 4 — Automation + storage
`feeds` new-episode detection against the state ledger; `watch`; `storage.py` cleanup; a
cron entry (`*/30 * * * * cd … && uv run polyglot watch --once`).

**Acceptance:** a new upstream episode yields a French episode in the feed within the cron
window; `cache/` stays bounded after `polyglot cleanup`.

### Phase 5 — Optional extras
- **Diarization / speaker labels** (whisperx + HF token; per-speaker voices via `speaker_map`).
- **Video** (`publish_video.py`: mux target audio + dual subs → mp4; unlisted YouTube).
  Re-timing to the *original* video is the hard case — accept drift with the dubbed-audio
  timeline, or use a duration-aware TTS.
- **Bespoke voice:** fine-tune XTTS on ~10+ min of a single speaker (GPU) if zero-shot
  cloning isn't faithful enough.

---

## 9. Subtitle timing (the core correctness rule)

The translated language is **~15–20% longer** than English. Never reuse English timestamps
for the dubbed audio. `assemble.py` measures each synthesized clip and builds the timeline
from cumulative durations + `gap_ms`. `subtitles.py` writes cues from that timeline only.
Concatenated audio is simply a bit longer — fine for podcasts.

---

## 10. Dependencies & setup

```bash
# system (already present on this machine)
ffmpeg            # brew install ffmpeg   (8.0.1 verified)

# project (uv-managed, Python 3.12)
uv init --python 3.12
uv add mlx mlx-lm mlx-whisper            # MLX transcription + translation (GPU)
uv add coqui-tts                         # XTTS v2 synthesis (import is `from TTS.api import TTS`)
uv add yt-dlp feedparser feedgen requests soundfile numpy boto3
# tomllib is stdlib in 3.12 (no tomli needed)
# optional fallbacks:
uv add faster-whisper                    # CPU transcription fallback (3.12 best-effort)
# optional Phase 5: uv add whisperx ; piper-tts
```

Model assets download to:
- HF cache (`~/.cache/huggingface/hub/`): Qwen 4-bit (4.28 GB), whisper-turbo (1.61 GB)
- TTS cache (`~/.local/share/tts/`): XTTS v2 (1.87 GB)

Env vars (never in config): `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.

---

## 11. Verified library reference (as of 2026-06-14)

| Package | Version | Python 3.12 | Apple Silicon | Notes |
|---|---|---|---|---|
| `mlx-lm` | 0.21.1+ (0.31.x current) | ✅ | GPU (Metal) | `load/generate/make_sampler`; not thread-safe; greedy at temp=0. |
| `mlx-whisper` | 0.4.3 | ✅ | GPU (Metal) | `transcribe(... path_or_hf_repo=..., language='en')` → dict. **3.13 breaks.** |
| `coqui-tts` | 0.27.5 | ✅ | **CPU only** | XTTS v2; `mps` hangs; torch≥2.6 needs weights_only shim; `COQUI_TOS_AGREED=1`. |
| `faster-whisper` | 1.2.1 | ⚠️ best-effort | CPU only | wheels claim 3.9–3.11; fallback backend. |
| `soundfile` | 0.14.0 | ✅ | n/a | write float32 `subtype='FLOAT'`. |
| `feedparser` / `feedgen` | current | ✅ | n/a | `feedgen.feed.FeedGenerator`, `load_extension('podcast')`. |
| `yt-dlp` | current | ✅ | n/a | `extract_flat` to list; postprocessor/CLI to extract mp3; needs ffmpeg. |
| `boto3` | 1.43.x | ✅ | n/a | R2: `region_name='auto'`; don't set AWS_DEFAULT_REGION. |

---

## 12. Key gotchas (consolidated)

- **Translated audio is longer.** Build the subtitle timeline from measured durations, not
  English timestamps (§9).
- **XTTS on Apple Silicon is CPU-only.** `device='mps'` hangs the system (upstream wontfix).
  It's the slowest stage; keep `clip_seconds` low while iterating.
- **torch ≥ 2.6 breaks XTTS load** (`weights_only=True` unpickling). Set
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before importing, or `add_safe_globals([...])`.
- **`COQUI_TOS_AGREED=1`** must be set before constructing `TTS(...)` or it hangs on input.
- **MLX is not thread-safe.** Load the LLM once, serialize requests; free it before XTTS to
  avoid co-resident peak RAM.
- **mlx-whisper requires Python 3.12, not 3.13.** Our pin is 3.12 — keep it.
- **faster-whisper 3.12 is best-effort** (metadata says 3.9–3.11). Default is mlx-whisper;
  only fall back if needed and test the install.
- **R2 signature:** `region_name='auto'`; do not set `AWS_DEFAULT_REGION`/`AWS_REGION`.
- **Apple Podcasts needs a publicly fetchable feed.** No truly-private Apple feed; host on R2
  at an obscure URL and don't share it.
- **Idempotency:** always check the state ledger by GUID before processing.
- **Disk:** stay aware of the ~16 GB working headroom; the Phase 4 cleanup policy keeps
  `cache/` bounded.

---

## 13. Extensibility recipes

**Add a language (e.g., Italian):**
1. Create `config/prompts/it.txt` (same shape as `fr.txt`, written in Italian).
2. Add a `[[show]]` block with `target_lang = "it"`.
3. (Optional) drop an Italian reference clip in `voices/` and set `voice`.
No code changes — XTTS already supports `it`.

**Add a show:** append a `[[show]]` block to `shows.toml`. Done.

**Add / clone a voice:** put `voices/<name>.wav` (or `voices/<name>/*.wav` for a richer
reference) and set `voice = "<name>"`. Omit `voice` for a built-in voice.

**Swap a backend:** change `transcribe_backend` / `translate_backend` / `tts_backend` in
`settings.toml`. No code change.

---

## 14. Open items — resolved 2026-06-14

Setup details verified and captured in `docs/setup/setup-notes.md`:

- **First feed (Phase 1): RESOLVED.** PTI (ESPN) = `https://feeds.megaphone.fm/ESP7239282233`
  — fetch-verified: valid RSS, direct `audio/mpeg` enclosures. Backups: ESPN Daily
  `ESP8348692127`, First Take `ESP1539938155`. Already in `config/shows.toml`.
- **Reference voice (Phase 1/2): RESOLVED (recommendation).** No clean free downloadable
  Québécois clip exists (Wikimedia options are 0.97 s or a music track). Phase 1 uses the
  built-in XTTS speaker (no clip); for cloning, record ~20–30 s yourself (Option B in setup
  notes), or assemble CC0 Common Voice Canadian-French clips.
- **R2 specifics (Phase 3): RESOLVED (procedure).** Free tier, no credit card, $0 egress;
  use the free r2.dev public URL for `public_base_url`. Account ID / bucket / keys are created
  by the user following setup notes §3; env vars `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`.
