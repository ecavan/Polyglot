from polyglot.transcribe import segments_from_mlx_result
from polyglot.segments import SEGMENT_KEYS


def test_segments_from_mlx_result():
    result = {
        "text": "Hello there. How are you?",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": " Hello there."},
            {"id": 1, "start": 1.2, "end": 2.8, "text": " How are you?"},
        ],
    }
    segs = segments_from_mlx_result(result)
    assert len(segs) == 2
    assert set(segs[0].keys()) == set(SEGMENT_KEYS)
    assert segs[0]["index"] == 0
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.2
    assert segs[0]["text"] == "Hello there."
    assert segs[1]["index"] == 1
    assert segs[1]["text"] == "How are you?"
