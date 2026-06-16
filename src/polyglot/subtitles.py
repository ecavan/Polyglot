from pathlib import Path


def _hms(seconds: float) -> tuple[int, int, int, int]:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def srt_timestamp(seconds: float) -> str:
    h, m, s, ms = _hms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    h, m, s, ms = _hms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _multi_speaker(segments: list[dict]) -> bool:
    return len({s.get("speaker") for s in segments if s.get("speaker")}) > 1


def _lines_for(seg: dict, bilingual: bool, show_speaker: bool) -> list[str]:
    prefix = f"{seg['speaker']}: " if (show_speaker and seg.get("speaker")) else ""
    lines = [f"{prefix}{seg['translation']}"]
    if bilingual:
        lines.append(seg["text"])
    return lines


def build_srt(segments: list[dict], timeline: list[tuple[float, float]], bilingual: bool) -> str:
    show_speaker = _multi_speaker(segments)   # only label speakers when there's more than one
    blocks = []
    for n, (seg, (start, end)) in enumerate(zip(segments, timeline), start=1):
        body = "\n".join(_lines_for(seg, bilingual, show_speaker))
        blocks.append(f"{n}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{body}\n")
    return "\n".join(blocks)


def build_vtt(segments: list[dict], timeline: list[tuple[float, float]], bilingual: bool) -> str:
    show_speaker = _multi_speaker(segments)
    blocks = ["WEBVTT\n"]
    for seg, (start, end) in zip(segments, timeline):
        body = "\n".join(_lines_for(seg, bilingual, show_speaker))
        blocks.append(f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}\n{body}\n")
    return "\n".join(blocks)


def ass_timestamp(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(s: str) -> str:
    return s.replace("\n", " ").replace("{", "(").replace("}", ")").strip()


# Side-by-side burned-in transcript: French in a blue box on the LEFT, English in a red
# box on the RIGHT (PlayRes 1920x1080; margins split the screen in half).
_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: FR,Arial,40,&H00FFFFFF,&H000000FF,&H20C04510,&H80000000,-1,0,0,0,100,100,0,0,3,5,0,1,40,980,60,1
Style: EN,Arial,40,&H00FFFFFF,&H000000FF,&H201030B0,&H80000000,0,0,0,0,100,100,0,0,3,5,0,3,980,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(segments: list[dict], timeline: list[tuple[float, float]]) -> str:
    lines = [_ASS_HEADER]
    for seg, (start, end) in zip(segments, timeline):
        a, b = ass_timestamp(start), ass_timestamp(end)
        lines.append(f"Dialogue: 0,{a},{b},FR,,0,0,0,,{_ass_text(seg['translation'])}")
        lines.append(f"Dialogue: 0,{a},{b},EN,,0,0,0,,{_ass_text(seg['text'])}")
    return "\n".join(lines) + "\n"


def lrc_timestamp(seconds: float) -> str:
    cs = int(round(seconds * 100))
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


def build_lrc(segments: list[dict], timeline: list[tuple[float, float]]) -> str:
    """Synced French lyrics for the podcast .mp3 — Finamp / the Jellyfin app render this as a
    scrolling, karaoke-style transcript while the audio plays (read-along on the phone)."""
    lines = []
    for seg, (start, _end) in zip(segments, timeline):
        text = seg["translation"].replace("\n", " ").strip()
        lines.append(f"{lrc_timestamp(start)}{text}")
    return "\n".join(lines) + "\n"


def write_subs(segments, timeline, out_dir: Path, show_id: str, ep_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ep_id}.srt").write_text(build_srt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.vtt").write_text(build_vtt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.target.srt").write_text(build_srt(segments, timeline, False), encoding="utf-8")
    (out_dir / f"{ep_id}.target.vtt").write_text(build_vtt(segments, timeline, False), encoding="utf-8")
    (out_dir / f"{ep_id}.ass").write_text(build_ass(segments, timeline), encoding="utf-8")  # styled side-by-side
    (out_dir / f"{ep_id}.lrc").write_text(build_lrc(segments, timeline), encoding="utf-8")  # synced lyrics (mp3)
