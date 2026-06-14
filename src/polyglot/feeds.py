from dataclasses import dataclass

import feedparser

from polyglot.config import JobSpec


@dataclass
class Episode:
    guid: str
    title: str
    published: str | None
    media_url: str


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
    )


def list_episodes_from_url(url: str, limit: int | None) -> list[Episode]:
    parsed = feedparser.parse(url)
    out: list[Episode] = []
    for e in parsed.entries:
        ep = _episode_from_entry(e)
        if ep is not None:
            out.append(ep)
        if limit is not None and len(out) >= limit:
            break
    return out


def list_youtube(url: str, limit: int | None, max_minutes: int = 60) -> list[Episode]:
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
        vid = e.get("id")
        if not vid:
            continue
        out.append(Episode(
            guid=vid,
            title=e.get("title", "(untitled)"),
            published=e.get("upload_date"),
            media_url=f"https://www.youtube.com/watch?v={vid}",
        ))
        if limit and len(out) >= limit:
            break
    return out


def list_episodes(job: JobSpec, limit: int | None, max_minutes: int = 60) -> list[Episode]:
    if job.source_type == "rss":
        return list_episodes_from_url(job.source, limit)
    if job.source_type == "youtube":
        return list_youtube(job.source, limit, max_minutes)
    raise NotImplementedError(f"source_type '{job.source_type}' not supported")
