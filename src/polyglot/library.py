import re
import shutil
from pathlib import Path

from polyglot.config import Settings


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]", "_", s).strip()[:120] or "episode"


def publish_to_library(kind: str, show_title: str, ep_title: str, media_srcs,
                       settings: Settings, ep_id: str = "", srt_src: Path | None = None) -> list[Path]:
    """Copy the dubbed media into the Jellyfin library folder. `media_srcs` may be a single
    path or a list (podcasts publish both the .mp3 for the phone and a subtitle-burned .mp4
    for the TV). The episode id is appended to the basename so two distinct episodes with the
    same human title can't overwrite each other. kind: "video" -> Videos/, else -> Podcasts/.

    Subtitles: VIDEO files have the FR/EN transcript BURNED IN, so they get NO sidecar (Jellyfin
    would render a sidecar as a second centered subtitle on top). PODCASTS ship a bilingual .srt
    next to the audio-only .mp3 (read-along on the phone); the companion TV .mp4 is named
    "... (TV).mp4" so that .srt does NOT also attach to it and double up over its burned-in subs."""
    if isinstance(media_srcs, (str, Path)):
        media_srcs = [media_srcs]
    sub = "Videos" if kind == "video" else "Podcasts"
    dest_dir = settings.library_path / sub / safe_name(show_title)
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(ep_title)
    if ep_id:
        base = f"{base} [{safe_name(ep_id)[:12]}]"
    out: list[Path] = []
    try:
        for m in media_srcs:
            suffix = Path(m).suffix
            # podcast TV video gets a distinct stem so the sidecar .srt (which matches the .mp3)
            # doesn't also attach to it and render doubled subtitles over the burned-in ones.
            stem = f"{base} (TV)" if (kind == "audio" and suffix == ".mp4") else base
            media_dest = dest_dir / f"{stem}{suffix}"
            shutil.copy2(m, media_dest)
            out.append(media_dest)
        if kind == "audio" and srt_src:        # read-along transcript for the audio-only .mp3
            srt_dest = dest_dir / f"{base}.srt"
            shutil.copy2(srt_src, srt_dest)
            out.append(srt_dest)
    except OSError:
        for p in out:                          # don't leave a half-published item behind
            p.unlink(missing_ok=True)
        raise
    return out
