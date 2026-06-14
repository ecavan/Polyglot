import re
import shutil
from pathlib import Path

from polyglot.config import Settings


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]", "_", s).strip()[:120] or "episode"


def publish_to_library(kind: str, show_title: str, ep_title: str,
                       media_src: Path, srt_src: Path, settings: Settings) -> list[Path]:
    """Copy the dubbed media + bilingual subtitle into the Jellyfin library folder.
    The .srt shares the media's basename so Jellyfin auto-loads it as a subtitle track.
    kind: "video" -> Videos/, else -> Podcasts/."""
    sub = "Videos" if kind == "video" else "Podcasts"
    dest_dir = settings.library_path / sub / safe_name(show_title)
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(ep_title)
    media_dest = dest_dir / f"{base}{Path(media_src).suffix}"
    srt_dest = dest_dir / f"{base}.srt"
    shutil.copy2(media_src, media_dest)
    shutil.copy2(srt_src, srt_dest)
    return [media_dest, srt_dest]
