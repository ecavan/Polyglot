import re
import shutil
from pathlib import Path

from polyglot.config import Settings


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]", "_", s).strip()[:120] or "episode"


def publish_to_library(kind: str, show_title: str, ep_title: str,
                       media_srcs, srt_src: Path, settings: Settings) -> list[Path]:
    """Copy the dubbed media + bilingual subtitle into the Jellyfin library folder.
    `media_srcs` may be a single path or a list (podcasts publish both the .mp3 for the
    phone and a subtitle-burned .mp4 for the TV). All media + the .srt share one basename
    so Jellyfin auto-loads the subtitle. kind: "video" -> Videos/, else -> Podcasts/."""
    if isinstance(media_srcs, (str, Path)):
        media_srcs = [media_srcs]
    sub = "Videos" if kind == "video" else "Podcasts"
    dest_dir = settings.library_path / sub / safe_name(show_title)
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(ep_title)
    out: list[Path] = []
    for m in media_srcs:
        media_dest = dest_dir / f"{base}{Path(m).suffix}"
        shutil.copy2(m, media_dest)
        out.append(media_dest)
    srt_dest = dest_dir / f"{base}.srt"
    shutil.copy2(srt_src, srt_dest)
    out.append(srt_dest)
    return out
