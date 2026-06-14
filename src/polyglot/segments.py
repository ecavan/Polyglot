SEGMENT_KEYS = (
    "index", "start", "end", "text",
    "translation", "speaker", "audio_path", "audio_dur",
)


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
