from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot.assemble import build_timeline, concat_audio
from polyglot.tts import SR
from polyglot.segments import new_segment


def _seg(i, dur):
    s = new_segment(i, 0.0, 0.0, f"t{i}")
    s["audio_dur"] = dur
    return s


def test_build_timeline_accumulates_with_gap():
    segs = [_seg(0, 1.0), _seg(1, 2.0)]
    tl = build_timeline(segs, gap_ms=200)
    assert tl[0] == (0.0, 1.0)
    assert abs(tl[1][0] - 1.2) < 1e-9
    assert abs(tl[1][1] - 3.2) < 1e-9


def test_concat_audio_inserts_silence(tmp_path: Path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(a, np.ones(SR, dtype=np.float32), SR, subtype="FLOAT")
    sf.write(b, np.ones(SR * 2, dtype=np.float32), SR, subtype="FLOAT")
    segs = [new_segment(0, 0, 0, "a"), new_segment(1, 0, 0, "b")]
    segs[0]["audio_path"] = str(a)
    segs[1]["audio_path"] = str(b)
    full = concat_audio(segs, gap_ms=200)
    gap_samples = int(0.2 * SR)
    assert len(full) == SR + gap_samples + SR * 2 + gap_samples
