from polyglot import feeds, library, pipeline, retention, state, storage
from polyglot.config import build_job, load_settings, load_shows


def run_watch(settings, shows) -> int:
    """One pass: for each enabled show, dub new items into the Jellyfin library,
    record them, then enforce retention. Returns how many new items were published."""
    published = 0
    for show in shows:
        if not show.enabled:
            continue
        try:
            job = build_job(show.id, settings, shows)
        except (FileNotFoundError, KeyError) as e:
            print(f"skip show {show.id}: {e}")
            continue
        eps = feeds.list_episodes(job, limit=settings.retention_keep,
                                  max_minutes=settings.max_video_minutes)
        for ep in eps:
            if state.is_done(settings.state_path, show.id, ep.guid):
                continue
            if show.source_type == "youtube":
                res = pipeline.process_video(job, ep, settings)
                kind, media = "video", res.get("mp4")
            else:
                res = pipeline.process_episode(job, ep, settings)
                kind, media = "audio", res.get("mp3")
            if not res.get("ok"):
                print(f"  FAILED {show.id}/{ep.guid}: {res.get('error')}")
                continue
            lib_files = library.publish_to_library(kind, show.title, ep.title, media, res["srt"], settings)
            files = lib_files + res.get("files", [])   # library copies + output artifacts: all purgeable
            state.mark_done(settings.state_path, show.id, ep.guid, kind, files, ep.title,
                            ts=ep.published_ts)   # purge by real air date, not ingest time
            published += 1
            print(f"  published {show.id}: {ep.title}")
        evicted = retention.apply_retention(
            settings.state_path, show.id, settings.retention_keep, settings.retention_max_age_days)
        if evicted:
            print(f"  retention: removed {len(evicted)} old item(s) from {show.id}")
    storage.cleanup_cache(settings)
    return published


def watch() -> int:
    return run_watch(load_settings(), load_shows())
