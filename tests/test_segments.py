from polyglot.segments import new_segment, SEGMENT_KEYS


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
