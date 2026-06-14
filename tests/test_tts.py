from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot.tts import synthesize_with, assign_voices, select_reference_spans, SR, CLONE
from polyglot.segments import new_segment


def test_synthesize_with_fake_writes_clips_and_durations(tmp_path: Path):
    segs = [new_segment(0, 0.0, 1.0, "Bonjour")]
    segs[0]["translation"] = "Bonjour"

    def fake_synth(seg):
        return np.zeros(int(0.5 * SR), dtype=np.float32)

    out = synthesize_with(segs, fake_synth, out_dir=tmp_path)
    p = Path(out[0]["audio_path"])
    assert p.exists()
    assert abs(out[0]["audio_dur"] - 0.5) < 1e-6
    data, sr = sf.read(p)
    assert sr == SR
    assert len(data) == int(0.5 * SR)


def test_assign_voices_no_clone_round_robin():
    m = assign_voices(["SPEAKER_01", "SPEAKER_00", "SPEAKER_00"], ["A", "B"], clone_available=False)
    assert m == {"SPEAKER_00": "A", "SPEAKER_01": "B"}


def test_assign_voices_single_speaker():
    m = assign_voices(["SPEAKER_00"], ["A", "B"], clone_available=False)
    assert m == {"SPEAKER_00": "A"}


def test_assign_voices_with_clone_uses_clone_for_first():
    m = assign_voices(["SPEAKER_00", "SPEAKER_01"], ["A", "B"], clone_available=True)
    assert m["SPEAKER_00"] == CLONE
    assert m["SPEAKER_01"] == "A"


def test_assign_voices_wraps_pool():
    m = assign_voices(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], ["A"], clone_available=False)
    assert m == {"SPEAKER_00": "A", "SPEAKER_01": "A", "SPEAKER_02": "A"}


def test_select_reference_spans_picks_longest_per_speaker():
    s0a = new_segment(0, 0.0, 1.0, "x"); s0a["speaker"] = "SPEAKER_00"     # 1s
    s0b = new_segment(1, 1.0, 6.0, "y"); s0b["speaker"] = "SPEAKER_00"     # 5s
    s1a = new_segment(2, 6.0, 7.0, "z"); s1a["speaker"] = "SPEAKER_01"     # 1s
    s1b = new_segment(3, 7.0, 17.0, "w"); s1b["speaker"] = "SPEAKER_01"    # 10s
    spans = select_reference_spans([s0a, s0b, s1a, s1b], target_seconds=4.0)
    assert [s["index"] for s in spans["SPEAKER_00"]] == [1]   # longest reaches target
    assert [s["index"] for s in spans["SPEAKER_01"]] == [3]
