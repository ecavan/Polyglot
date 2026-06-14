from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot.assemble import build_timeline, concat_audio, mix_speech_and_bed, build_synced_track
from polyglot.tts import SR
from polyglot.segments import new_segment


def test_build_synced_track_anchors_to_source_time(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(a, np.ones(SR, dtype=np.float32), SR, subtype="FLOAT")   # 1s
    sf.write(b, np.ones(SR, dtype=np.float32), SR, subtype="FLOAT")   # 1s
    segs = [new_segment(0, 0.0, 2.0, "x"), new_segment(1, 2.0, 3.0, "y")]
    segs[0]["audio_path"] = str(a)
    segs[1]["audio_path"] = str(b)
    track = build_synced_track(segs, source_duration=3.0)
    assert len(track) >= int(3 * SR) - 2
    assert track[10] > 0.5                          # seg0 (1s in 2s slot) plays at natural pace from t=0
    assert abs(track[int(1.5 * SR)]) < 1e-6         # gap (1-2s) is silent (pause preserved)
    assert track[int(2.0 * SR):].max() > 0.5        # seg1 anchored at/after its source start (t=2s)


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


def test_mix_speech_and_bed_pads_and_clamps():
    speech = np.ones(100, dtype=np.float32) * 0.8
    bed = np.ones(60, dtype=np.float32)            # shorter -> padded
    mixed = mix_speech_and_bed(speech, bed, bed_gain=0.3)
    assert len(mixed) == 100
    # first 60 samples: 0.8 + 0.3*1.0 = 1.1 -> clamped via /peak to <= 1.0
    assert float(np.max(np.abs(mixed))) <= 1.0 + 1e-6
    # tail (padded bed = 0) should be just the speech, scaled by the same peak factor
    assert mixed[80] < mixed[0]  # tail (0.8/peak) quieter than head (1.1/peak)


def test_mix_speech_and_bed_truncates_long_bed():
    speech = np.zeros(50, dtype=np.float32)
    bed = np.ones(200, dtype=np.float32)
    mixed = mix_speech_and_bed(speech, bed, bed_gain=0.5)
    assert len(mixed) == 50
