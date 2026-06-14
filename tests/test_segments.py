from polyglot.segments import new_segment, merge_short_segments, SEGMENT_KEYS


def test_new_segment_has_all_keys_and_defaults():
    seg = new_segment(index=0, start=1.5, end=2.0, text="Hello")
    assert set(seg.keys()) == set(SEGMENT_KEYS)
    assert seg["index"] == 0
    assert seg["start"] == 1.5
    assert seg["end"] == 2.0
    assert seg["text"] == "Hello"
    assert seg["translation"] is None
    assert seg["speaker"] is None
    assert seg["audio_path"] is None
    assert seg["audio_dur"] is None


def _spk(i, start, end, text, speaker):
    s = new_segment(i, start, end, text)
    s["speaker"] = speaker
    return s


def test_merge_short_segments_combines_same_speaker_fragments():
    segs = [
        _spk(0, 0.0, 0.5, "No,", "S0"),
        _spk(1, 0.5, 1.0, "no,", "S0"),
        _spk(2, 1.0, 1.6, "I'm not going.", "S0"),
        _spk(3, 1.6, 2.0, "Are you sure?", "S1"),   # different speaker -> not merged in
    ]
    out = merge_short_segments(segs, min_chars=50)
    assert len(out) == 2
    assert out[0]["text"] == "No, no, I'm not going."
    assert out[0]["speaker"] == "S0"
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.6
    assert out[0]["index"] == 0
    assert out[1]["text"] == "Are you sure?"
    assert out[1]["index"] == 1


def test_merge_short_segments_keeps_long_segments_separate():
    long_text = "x" * 60
    segs = [_spk(0, 0, 2, long_text, "S0"), _spk(1, 2, 3, "next", "S0")]
    out = merge_short_segments(segs, min_chars=50)
    assert len(out) == 2  # first already exceeds min_chars -> not merged
