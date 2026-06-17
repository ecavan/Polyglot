"""Pick a voice-pitch scheme BY EAR without re-dubbing whole videos.

Synthesizes one French phrase once with Orpheus (Pierre), renders it at each pitch offset
(duration-preserving), strings them back-to-back with a big on-screen label per pitch, and
publishes the clip to the Jellyfin Videos library ("Voice Test") so you can play it on the TV
or phone and choose. Then set [orpheus] pitch_mode / voice_pitch in settings.toml to match.

Usage:
  uv run python scripts/voice_pitch_test.py                 # default 0 +1 -1 +2 -2 (the "spread")
  uv run python scripts/voice_pitch_test.py 0 1 -1 2 -2     # custom pitches (semitones)
  uv run python scripts/voice_pitch_test.py 0 -1 -2 -3      # the old deeper-only ladder
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot import library, orpheus_tts
from polyglot.config import load_settings
from polyglot.subtitles import ass_timestamp
from polyglot.tts import SR, _pitch_shift

PHRASE = ("Bonjour, je m'appelle Pierre. Écoutez bien la hauteur de ma voix, "
          "c'est comme ça que je vous parlerais dans le doublage.")

_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BIG,Arial,96,&H00FFFFFF,&H00202020,&H00000000,-1,1,3,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _label(p: float) -> str:
    p = int(p) if float(p).is_integer() else p
    sign = "+" if (isinstance(p, int) and p > 0) or (isinstance(p, float) and p > 0) else ""
    return f"Pierre {sign}{p}" if p else "Pierre 0 (natural)"


def main(argv: list[str]) -> int:
    pitches = [float(x) for x in argv] if argv else [0, 1, -1, 2, -2]
    s = load_settings()
    print(f"pitches: {pitches}  — synthesizing once, then pitch-shifting", flush=True)
    base = orpheus_tts.build_synth(s)
    dry = np.asarray(base(PHRASE, "Pierre"), dtype=np.float32)

    gap = np.zeros(int(0.7 * SR), dtype=np.float32)
    parts, cues = [], []
    t = 0.0
    for p in pitches:
        clip = _pitch_shift(dry, p)
        dur = len(clip) / SR
        cues.append(f"Dialogue: 0,{ass_timestamp(t)},{ass_timestamp(t + dur)},BIG,,0,0,0,,{_label(p)}")
        parts += [clip, gap]
        t += dur + 0.7

    full = np.concatenate(parts)
    td = Path(tempfile.mkdtemp())
    wav, assf, mp4 = td / "v.wav", td / "v.ass", td / "voice_pitch_test.mp4"
    sf.write(str(wav), full, SR, subtype="FLOAT")
    assf.write_text(_ASS + "\n".join(cues) + "\n", encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x111418:s=1280x720:r=10:d={len(full) / SR:.2f}",
         "-i", str(wav), "-vf", f"ass={assf}",
         "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", str(mp4)],
        check=True,
    )
    title = "Pierre pitch test (" + " ".join(_label(p).replace("Pierre ", "") for p in pitches) + ")"
    files = library.publish_to_library("video", "Voice Test", title, [mp4], s, ep_id="voicetest")
    print(f"published to library: {files}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
