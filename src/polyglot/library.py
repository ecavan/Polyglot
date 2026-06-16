import re
import shutil
from pathlib import Path

from polyglot.config import Settings


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]", "_", s).strip()[:120] or "episode"


def publish_to_library(kind: str, show_title: str, ep_title: str,
                       media_srcs, settings: Settings, ep_id: str = "") -> list[Path]:
    """Copy the dubbed media into the Jellyfin library folder. `media_srcs` may be a single
    path or a list (podcasts publish both the .mp3 for the phone and a subtitle-burned .mp4
    for the TV). The episode id is appended to the basename so two distinct episodes with the
    same human title can't overwrite each other (which would also let retention delete a
    still-live episode's file). kind: "video" -> Videos/, else -> Podcasts/.

    We deliberately do NOT ship a sidecar .srt: the FR/EN transcript is BURNED INTO the video
    (.ass side-by-side boxes), and Jellyfin would render a sidecar as a second, centered white
    subtitle track on top of it."""
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
            media_dest = dest_dir / f"{base}{Path(m).suffix}"
            shutil.copy2(m, media_dest)
            out.append(media_dest)
    except OSError:
        for p in out:                       # don't leave a half-published item behind
            p.unlink(missing_ok=True)
        raise
    return out
