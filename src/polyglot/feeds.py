import calendar
from dataclasses import dataclass
from datetime import datetime

import feedparser

from polyglot.config import JobSpec


@dataclass
class Episode:
    guid: str
    title: str
    published: str | None
    media_url: str
    published_ts: float | None = None   # real air/upload date as epoch seconds (for retention age)
    duration_sec: float | None = None   # episode length (for the min/max length filters)


def _parse_duration(s) -> float | None:
    """itunes:duration -> seconds. Accepts 'H:M:S', 'M:S', or plain seconds; None if unknown."""
    if s is None:
        return None
    s = str(s).strip()
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            sec = 0.0
            for p in parts:
                sec = sec * 60 + p
            return sec
        return float(s)
    except (ValueError, TypeError):
        return None


def _struct_to_epoch(st) -> float | None:
    return calendar.timegm(st) if st else None


def _yyyymmdd_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return calendar.timegm(datetime.strptime(s, "%Y%m%d").timetuple())
    except (ValueError, TypeError):
        return None


def _episode_from_entry(e) -> Episode | None:
    enclosures = e.get("enclosures") or []
    if not enclosures:
        return None
    media_url = enclosures[0].get("href")
    if not media_url:
        return None
    guid = e.get("id") or e.get("guid") or media_url
    return Episode(
        guid=guid,
        title=e.get("title", "(untitled)"),
        published=e.get("published"),
        media_url=media_url,
        published_ts=_struct_to_epoch(e.get("published_parsed")),
        duration_sec=_parse_duration(e.get("itunes_duration")),
    )


def list_episodes_from_url(url: str, limit: int | None, min_seconds: float = 0) -> list[Episode]:
    parsed = feedparser.parse(url)
    # feedparser never raises on a dead/unreachable/malformed feed — it sets bozo and
    # returns salvaged (often zero) entries. Surface that so a broken feed isn't silently
    # mistaken for "no new episodes".
    if parsed.bozo and not parsed.entries:
        exc = parsed.get("bozo_exception")
        print(f"  WARNING feed fetch/parse failed ({url}): {exc}")
        return []
    out: list[Episode] = []
    for e in parsed.entries:
        ep = _episode_from_entry(e)
        if ep is None:
            continue
        if min_seconds and ep.duration_sec is not None and ep.duration_sec < min_seconds:
            continue  # skip previews / trailers / short clips
        out.append(ep)
        if limit is not None and len(out) >= limit:
            break
    return out


def list_youtube(url: str, limit: int | None, max_minutes: int = 60,
                 min_seconds: float = 0) -> list[Episode]:
    from yt_dlp import YoutubeDL
    opts = {"extract_flat": True, "quiet": True, "noprogress": True}
    if limit:
        opts["playlistend"] = limit
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    out: list[Episode] = []
    for e in (info.get("entries") or []):
        dur = e.get("duration") or 0
        if max_minutes and dur and dur > max_minutes * 60:
            continue  # flat-listing may omit duration; fetch_video re-checks the hard limit
        if min_seconds and dur and dur < min_seconds:
            continue  # skip shorts / clips
        vid = e.get("id")
        if not vid:
            continue
        out.append(Episode(
            guid=vid,
            title=e.get("title", "(untitled)"),
            published=e.get("upload_date"),
            media_url=f"https://www.youtube.com/watch?v={vid}",
            published_ts=_yyyymmdd_to_epoch(e.get("upload_date")),
            duration_sec=dur or None,
        ))
        if limit and len(out) >= limit:
            break
    return out


def list_episodes(job: JobSpec, limit: int | None, max_minutes: int = 60,
                  min_minutes: float = 0) -> list[Episode]:
    min_seconds = (min_minutes or 0) * 60
    if job.source_type == "rss":
        return list_episodes_from_url(job.source, limit, min_seconds=min_seconds)
    if job.source_type == "youtube":
        return list_youtube(job.source, limit, max_minutes, min_seconds=min_seconds)
    raise NotImplementedError(f"source_type '{job.source_type}' not supported")
