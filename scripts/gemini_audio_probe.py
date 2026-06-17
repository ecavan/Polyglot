"""Prototype: transcribe + translate + diarize + timestamp audio in ONE Gemini call.

Evaluates whether Gemini's multimodal audio understanding can replace the
whisper -> diarize -> translate steps for noisy/accented multi-speaker video
(e.g. poker). Downloads a clip, sends it to Gemini, prints aligned EN/FR lines.

Usage:
  uv run python scripts/gemini_audio_probe.py <youtube_url> [start-end]
  e.g. ... "https://www.youtube.com/watch?v=81IEMHTsZ9A" 120-210
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = (
    "This is English audio from a high-stakes poker stream: a commentator/narrator plus players "
    "talking at the table, with chip/crowd background noise and some non-native accents. Transcribe "
    "the speech, then translate each line into natural spoken Québécois French. Return a JSON array "
    "of {start, speaker, en, fr} where start is the seconds offset into THIS clip, speaker is "
    "'narrator' or 'player'. Only real speech."
)
SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT",
          "properties": {"start": {"type": "NUMBER"}, "speaker": {"type": "STRING"},
                         "en": {"type": "STRING"}, "fr": {"type": "STRING"}},
          "required": ["start", "en", "fr"]}}


def transcribe_audio(mp3_path: Path, model="gemini-3.1-pro-preview") -> list[dict]:
    key = os.environ["GEMINI_API_KEY"]
    audio = base64.b64encode(mp3_path.read_bytes()).decode()
    body = {
        "contents": [{"parts": [{"inline_data": {"mime_type": "audio/mp3", "data": audio}},
                                 {"text": PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA,
                             "thinkingConfig": {"thinkingLevel": "low"},
                             "maxOutputTokens": 16384, "temperature": 0.3},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    return json.loads(txt)


def main(argv):
    url = argv[0]
    start, end = (argv[1].split("-") if len(argv) > 1 else ("120", "210"))
    clip = Path("/tmp/gemini_probe.mp3")
    clip.unlink(missing_ok=True)
    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "5",
                    "--download-sections", f"*{start}-{end}", "-o", "/tmp/gemini_probe.%(ext)s", url],
                   check=True, capture_output=True)
    t = time.time()
    rows = transcribe_audio(clip)
    print(f"{len(rows)} lines in {time.time()-t:.1f}s\n" + "=" * 80)
    for r in rows:
        print(f"[{r['start']:>5.1f}s] ({r.get('speaker','?'):8}) {r['en']}\n           {r['fr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
