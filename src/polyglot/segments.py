SEGMENT_KEYS = (
    "index", "start", "end", "text",
    "translation", "speaker", "audio_path", "audio_dur",
)


def merge_short_segments(segments: list[dict], min_chars: int = 50) -> list[dict]:
    """Combine consecutive SAME-speaker segments while the running text is shorter than
    min_chars. XTTS (especially cloned voices) rambles/hallucinates on very short inputs,
    so feeding it fuller phrases is both more stable and more natural. Re-indexes the result.
    Operates on source `text`; run after diarization, before translation."""
    merged: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        if cur is not None and cur.get("speaker") == seg.get("speaker") and len(cur["text"]) < min_chars:
            cur = dict(cur)
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
            cur["end"] = seg["end"]
            merged[-1] = cur
        else:
            cur = dict(seg)
            merged.append(cur)
    for i, s in enumerate(merged):
        s["index"] = i
    return merged


def new_segment(index: int, start: float, end: float, text: str) -> dict:
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "translation": None,
        "speaker": None,
        "audio_path": None,
        "audio_dur": None,
    }
