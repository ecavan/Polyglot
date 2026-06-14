# Polyglot — Setup Notes

Practical, verified setup details that back the implementation plan. Everything here is **free**.

Last verified: **2026-06-14**.

---

## 1. First feed URL (Phase 1)

**Use PTI (ESPN's "Pardon the Interruption").** It matches the translation prompt ("deux gars
qui jasent de sports") and is a clean audio RSS feed.

```
https://feeds.megaphone.fm/ESP7239282233
```

**Verified:** fetched 2026-06-14 → `<title>PTI</title>`, latest item present, enclosures are direct
`type="audio/mpeg"` MP3s (via podtrac redirect). Daily weekday episodes, ~24–25 min each.

> The source enclosure reports `length="0"` — harmless. Our pipeline re-encodes the dubbed audio
> and measures its own byte length for the output feed.

**Backups** (also verified, same Megaphone infra) if PTI ever changes its feed:

| Show | Feed URL |
|---|---|
| ESPN Daily | `https://feeds.megaphone.fm/ESP8348692127` |
| First Take | `https://feeds.megaphone.fm/ESP1539938155` |

This URL is already in `config/shows.toml` (created in plan Task 0.7).

---

## 2. Québécois reference voice clip (Phase 1 optional, Phase 2)

**Honest summary:** there is no perfect, ready-made, free, license-clean Québécois clip to just
download. The free Wikimedia options are unusable for cloning:

- `FR-Québécois.ogg` — **0.97 s** (XTTS needs **≥6 s**). ✗
- `Un Canadien errant.ogg` — 36 s but it's a **folk song with music** (music wrecks the voice
  embedding). ✗

So the recommended path is in this order:

### Option A (recommended for Phase 1): use a built-in voice, no clip
Phase 1 is the **quality gate** — judge the *French translation + timing* first. Leave `voice`
out of the show block (already done) and the pipeline uses XTTS's built-in `Claribel Dervla`
speaker. No download, no licensing. Add a cloned voice only after the French clears your bar.

### Option B (best clone, free, authentic): record ~20–30 s yourself
A clean self-recording (you, or a Québécois family member/friend) beats any scraped clip — XTTS
cloning rewards a **clean, single-speaker** reference over a long noisy one.

1. Record ~20–30 s of natural speech (QuickTime Player → File → New Audio Recording, or any app).
2. Convert to a clean mono WAV and drop it in `voices/`:
   ```bash
   ffmpeg -i recording.m4a -ar 22050 -ac 1 -af "highpass=f=80,lowpass=f=8000" voices/fr_montreal.wav
   ```
3. Set `voice = "fr_montreal"` in the show block. (For a richer reference, put several clips in
   `voices/fr_montreal/*.wav` instead — the config resolves a folder automatically.)

Tips: quiet room, no background music/TV, consistent volume, conversational tone.

### Option C (downloadable, CC0, more effort): Mozilla Common Voice — Canadian French
`https://commonvoice.mozilla.org/en/datasets` (or HF mirror
`https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0`). License is **CC0**.
Caveat: clips are short single sentences and accent/region is a column in the dataset TSV
(`accents`), so you must download the French set and filter for Canada/Québec, then concatenate a
few same-speaker clips to reach ≥6 s. Doable but more work than Option B.

> Bottom line: **start with Option A**, then do **Option B** when you want the cloned voice.

---

## 3. Cloudflare R2 hosting — free, step by step (Phase 3)

R2's free tier is generous and **needs no credit card**, and egress is **$0** (the key win for
podcast hosting). Verified against Cloudflare docs 2026-06-14.

### Free-tier limits (per month)
| Resource | Free allowance |
|---|---|
| Storage | 10 GB-month |
| Class A ops (PUT/POST/DELETE) | 1,000,000 |
| Class B ops (GET/LIST) | 10,000,000 |
| Egress (bandwidth out) | **Unlimited / $0** |

A personal French-podcast feed is nowhere near these limits.

### Steps
1. **Create a free account** — `https://dash.cloudflare.com/sign-up`. No card required.
2. **Create a bucket** — Dashboard → *Storage & databases → R2 → Create bucket*. Name it
   `polyglot-media` (lowercase, hyphens OK). Location: *Automatic* (or hint `enam`/`wnam`).
3. **Enable public access (free)** — open the bucket → *Settings → Public Development URL →
   Enable* (type `allow` to confirm). You get a public URL like:
   ```
   https://polyglot-media.<random>.r2.dev/<key>
   ```
   This is what `public_base_url` becomes in `config/settings.toml`.
   - r2.dev is rate-limited ("hundreds of req/s" → 429) and labelled non-production, but that's
     **fine** for a personal podcast + Apple's occasional feed fetches.
4. **Create S3 API credentials** — Dashboard → *R2 → Manage API Tokens → Create API token*.
   Permission: **Object Read & Write**, scoped to the `polyglot-media` bucket. Copy the
   **Access Key ID** and **Secret Access Key** now (the secret is shown once).
5. **Get your Account ID** — shown on the R2 overview page (32-hex string). The S3 endpoint is
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
6. **Set env vars** (the project's `publish_rss.py` reads these and builds the endpoint from
   `R2_ACCOUNT_ID`; do **not** set `AWS_DEFAULT_REGION`):
   ```bash
   # add to ~/.zshrc, then `source ~/.zshrc`
   export R2_ACCOUNT_ID="your-32-hex-account-id"
   export R2_ACCESS_KEY_ID="your-access-key-id"
   export R2_SECRET_ACCESS_KEY="your-secret-access-key"
   ```
7. **Point the config at it** — in `config/settings.toml` `[hosting]`:
   ```toml
   type            = "r2"
   public_base_url = "https://polyglot-media.<random>.r2.dev"
   bucket          = "polyglot-media"
   ```

### boto3 specifics (already in the spec's `publish_rss.py`)
- `region_name="auto"` (R2 requirement).
- Always upload with `ExtraArgs={"ContentType": "audio/mpeg"}` so Apple/browsers stream it.
- Public URL = `public_base_url + "/" + key`.

### Cost check
Account $0 · storage (10 GB free) $0 · ops (1M/10M free) $0 · egress $0 · r2.dev $0.
A custom domain is optional and only costs money if you register a *new* domain (~$9/yr at-cost);
not needed — r2.dev is free.

---

## Sources
- PTI / feeds: verified by direct fetch (Megaphone), Apple Podcasts `id147232181`.
- Voice: Wikimedia Commons (durations confirmed via ffprobe), XTTS-v2 model card (≥6 s, clean,
  single-speaker), Mozilla Common Voice (CC0).
- R2: Cloudflare developer docs — pricing, public-buckets, get-started/s3, platform/limits.
