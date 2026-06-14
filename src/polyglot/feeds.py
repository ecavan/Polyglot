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


def list_episodes(job: JobSpec, limit: int | None) -> list[Episode]:
    if job.source_type == "rss":
        return list_episodes_from_url(job.source, limit)
    raise NotImplementedError(f"source_type '{job.source_type}' not supported in Phase 1")
