from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot.tts import synthesize_with, SR
from polyglot.segments import new_segment


def test_synthesize_with_fake_writes_clips_and_durations(tmp_path: Path):
    segs = [new_segment(0, 0.0, 1.0, "Bonjour")]
    segs[0]["translation"] = "Bonjour"

    def fake_synth(text):
        return np.zeros(int(0.5 * SR), dtype=np.float32)

    out = synthesize_with(segs, fake_synth, out_dir=tmp_path)
    p = Path(out[0]["audio_path"])
    assert p.exists()
    assert abs(out[0]["audio_dur"] - 0.5) < 1e-6
    data, sr = sf.read(p)
    assert sr == SR
    assert len(data) == int(0.5 * SR)
