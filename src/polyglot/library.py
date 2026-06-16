import re
import shutil
import subprocess
from pathlib import Path

from polyglot.config import Settings


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]", "_", s).strip()[:120] or "episode"


def _subdir(kind: str, suffix: str) -> str:
    """Which Jellyfin library folder a file belongs in. Jellyfin only surfaces audio in a
    Music-type library, so podcast .mp3s go to a dedicated PodcastAudio/ (point a *Music*
    library at it); the burned-in podcast video goes to Podcasts/, and everything else to Videos/."""
    if kind != "audio":
        return "Videos"
    return "PodcastAudio" if suffix == ".mp3" else "Podcasts"


def _copy_mp3_tagged(src: Path, dst: Path, title: str, album: str) -> None:
    """Copy an .mp3 setting ID3 tags so it lists cleanly in a Music library (grouped by show)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c", "copy",
         "-metadata", f"title={title}", "-metadata", f"album={album}",
         "-metadata", f"artist={album}", "-metadata", "genre=Podcast", str(dst)],
        check=True, capture_output=True,
    )


def publish_to_library(kind: str, show_title: str, ep_title: str, media_srcs,
                       settings: Settings, ep_id: str = "", lrc_src: Path | None = None) -> list[Path]:
    """Copy the dubbed media into the Jellyfin library. `media_srcs` may be a single path or a
    list (podcasts publish both an .mp3 for screen-off phone listening and a subtitle-burned
    .mp4 for read-along). The episode id is appended to the basename so two distinct episodes
    with the same title can't overwrite each other.

    Files are routed by type (see _subdir): podcast .mp3 -> PodcastAudio/ (Music library, tagged
    by show), podcast .mp4 -> Podcasts/, other video -> Videos/. The podcast .mp3 also gets a
    same-basename .lrc (synced French lyrics) so Finamp/Jellyfin show a scrolling transcript while
    you listen. No sidecar .srt: the video transcript is BURNED IN (a sidecar would double up)."""
    if isinstance(media_srcs, (str, Path)):
        media_srcs = [media_srcs]
    base = safe_name(ep_title)
    if ep_id:
        base = f"{base} [{safe_name(ep_id)[:12]}]"
    out: list[Path] = []
    try:
        for m in media_srcs:
            suffix = Path(m).suffix
            dest_dir = settings.library_path / _subdir(kind, suffix) / safe_name(show_title)
            dest_dir.mkdir(parents=True, exist_ok=True)
            if suffix == ".mp3":
                dest = dest_dir / f"{base}.mp3"
                _copy_mp3_tagged(m, dest, title=ep_title, album=show_title)
                out.append(dest)
                if lrc_src:                         # synced lyrics next to the audio
                    lrc_dest = dest_dir / f"{base}.lrc"
                    shutil.copy2(lrc_src, lrc_dest)
                    out.append(lrc_dest)
            else:                                   # podcast TV video labelled, other video plain
                stem = f"{base} (TV)" if kind == "audio" else base
                dest = dest_dir / f"{stem}{suffix}"
                shutil.copy2(m, dest)
                out.append(dest)
    except Exception:
        for p in out:                               # don't leave a half-published item behind
            p.unlink(missing_ok=True)
        raise
    return out
