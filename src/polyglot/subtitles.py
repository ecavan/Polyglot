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


def write_subs(segments, timeline, out_dir: Path, show_id: str, ep_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ep_id}.srt").write_text(build_srt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.vtt").write_text(build_vtt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.target.srt").write_text(build_srt(segments, timeline, False), encoding="utf-8")
    (out_dir / f"{ep_id}.target.vtt").write_text(build_vtt(segments, timeline, False), encoding="utf-8")
