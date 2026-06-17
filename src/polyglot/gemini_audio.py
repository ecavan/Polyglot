"""Gemini multimodal audio: transcribe + diarize + translate to Québécois in ONE call.

Replaces the whisper -> diarize -> translate steps for the noisy/accented multi-speaker case
(it handles quiet table chatter and non-native accents far better than whisper). Returns
segments already carrying English text, French translation, a speaker label, and a start time.
Long audio is sent via the Files API. Pipeline falls back to the whisper path if this fails.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from polyglot.config import Settings
from polyglot.segments import new_segment

_PROMPT = (
    "You are given audio from a podcast or video. It may contain background noise, music, chips/"
    "crowd sounds, and non-native English accents. Do the following:\n"
    "1. Transcribe ALL clearly intelligible speech (skip music, applause, and unintelligible noise).\n"
    "2. Translate each line into natural, idiomatic spoken Québécois French (Montréal register; keep "
    "proper nouns and common anglicisms like 'call', 'all-in', 'show').\n"
    "3. Diarize: give each distinct voice a consistent label across the whole audio "
    "('speaker 1', 'speaker 2', ... — use the SAME label every time that person talks).\n"
    "Return ONLY a JSON array of objects {start, speaker, en, fr}, ordered by start, where start is "
    "the seconds offset into the audio when the line begins."
)

# Hybrid: a rough whisper transcript (accurate timestamps, error-prone words) is provided; Gemini
# corrects the words against the audio and keeps whisper's precise (sub-second) timing.
_PROMPT_HYBRID = (
    "You are given audio AND a rough machine transcript of it (index, start, end, text — tab-"
    "separated). The timestamps are accurate; the words contain errors (accents, noise). Using the "
    "AUDIO to correct what is actually said:\n"
    "1. Return one line per real utterance, KEEPING the transcript's start/end (you may merge "
    "adjacent fragments of one sentence — keep the merged span's start and end). Drop lines that "
    "are just noise/music.\n"
    "2. Correct 'en' to the true words; translate to natural spoken Québécois French ('fr').\n"
    "3. Give each distinct voice a consistent speaker label.\n"
    "Return ONLY a JSON array of {start, end, speaker, en, fr}, ordered by start."
)


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _domain_context(settings: Settings, domain: str | None) -> str:
    """Domain-specific guidance (vocab to keep, register, speaker roles) appended to the base
    prompt. Lives in config/prompts/audio/<domain>.txt so it's tunable per show."""
    name = (domain or "general").strip().lower()
    path = settings.prompts_dir / "audio" / f"{name}.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _compact_mp3(src: Path, dst: Path) -> Path:
    """Mono 16 kHz ~48 kbps mp3 — tiny to upload, plenty for speech recognition."""
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-b:a", "48k", str(dst)], check=True, capture_output=True)
    return dst


def _rows_to_segments(rows: list[dict]) -> list[dict]:
    clean = [r for r in rows if r.get("en") and r.get("start") is not None]
    clean.sort(key=lambda r: float(r["start"]))
    out = []
    for i, r in enumerate(clean):
        seg = new_segment(i, float(r["start"]), 0.0, str(r["en"]).strip())
        seg["translation"] = str(r.get("fr", "")).strip()
        seg["speaker"] = str(r.get("speaker") or "speaker 1")
        e = r.get("end")
        seg["end"] = float(e) if (e is not None and float(e) > seg["start"]) else None
        out.append(seg)
    for i, seg in enumerate(out):                       # fill missing ends: until the next line
        if seg["end"] is None:
            seg["end"] = max(seg["start"], out[i + 1]["start"]) if i + 1 < len(out) else seg["start"] + 4.0
    return out


def transcribe_translate(audio_path: Path, settings: Settings, domain: str | None = None,
                         draft: list[dict] | None = None) -> list[dict]:
    """Audio -> EN+FR+speaker+timestamps. With `draft` (whisper segments) it runs the HYBRID:
    Gemini corrects the words against the audio while keeping whisper's precise timestamps."""
    from google import genai
    from google.genai import types

    ctx = _domain_context(settings, domain)
    ctx_block = f"\n\n--- Domain context ({domain}) ---\n{ctx}" if ctx else ""
    props = {"start": types.Schema(type=types.Type.NUMBER),
             "speaker": types.Schema(type=types.Type.STRING),
             "en": types.Schema(type=types.Type.STRING),
             "fr": types.Schema(type=types.Type.STRING)}
    if draft:
        lines = "\n".join(f"{i}\t{d['start']:.2f}\t{(d.get('end') or d['start']):.2f}\t{d['text']}"
                          for i, d in enumerate(draft))
        prompt = _PROMPT_HYBRID + ctx_block + "\n\nRough machine transcript:\n" + lines
        props["end"] = types.Schema(type=types.Type.NUMBER)
    else:
        prompt = _PROMPT + ctx_block

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.OBJECT, properties=props, required=["start", "en", "fr"]),
    )
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema,
        temperature=0.3, max_output_tokens=65536,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    with tempfile.TemporaryDirectory() as td:
        mp3 = _compact_mp3(audio_path, Path(td) / "a.mp3")
        f = client.files.upload(file=str(mp3))
        try:
            while getattr(f.state, "name", "ACTIVE") == "PROCESSING":
                time.sleep(2)
                f = client.files.get(name=f.name)
            if getattr(f.state, "name", "") == "FAILED":
                raise RuntimeError("Gemini file processing failed")
            resp = client.models.generate_content(
                model=settings.gemini_model, contents=[f, prompt], config=cfg)
        finally:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass

    rows = json.loads(resp.text)
    return _rows_to_segments(rows)
