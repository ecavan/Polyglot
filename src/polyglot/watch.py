import fcntl
from contextlib import contextmanager

from polyglot import feeds, library, pipeline, retention, state, storage
from polyglot.config import build_job, load_settings, load_shows


@contextmanager
def _lock(lock_path):
    """Exclusive non-blocking file lock so overlapping cron/launchd runs don't stack
    (a slow dub can outlast the interval). Yields True if acquired, False if held."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        f.close()


def _process_show(show, settings, shows) -> dict:
    """Dub + publish all new items for one show, then enforce retention. Isolated so one
    show's feed/processing failure never aborts the others."""
    result = {"published": 0, "failed": 0}
    try:
        job = build_job(show.id, settings, shows)
    except (FileNotFoundError, KeyError) as e:
        print(f"skip show {show.id}: {e}")
        return result
    try:
        eps = feeds.list_episodes(job, limit=settings.retention_keep,
                                  max_minutes=settings.max_video_minutes)
    except Exception as e:  # yt-dlp / network errors must not abort the whole pass
        print(f"  WARNING listing {show.id} failed: {e}")
        return result
    for ep in eps:
        if state.is_done(settings.state_path, show.id, ep.guid):
            continue
        ep_id = pipeline._safe_id(ep.guid)
        try:
            if show.source_type == "youtube":
                res = pipeline.process_video(job, ep, settings)
                kind = "video"
            else:
                res = pipeline.process_episode(job, ep, settings)
                kind = "audio"
            if not res.get("ok"):
                print(f"  FAILED {show.id}/{ep.guid}: {res.get('error')}")
                result["failed"] += 1
                continue
            media = res.get("media") or [res.get("mp4") or res.get("mp3")]
            lib_files = library.publish_to_library(kind, show.title, ep.title, media,
                                                   settings, ep_id=ep_id)
            files = lib_files + res.get("files", [])   # library copies + output artifacts: all purgeable
            state.mark_done(settings.state_path, show.id, ep.guid, kind, files, ep.title,
                            ts=ep.published_ts)   # purge by real air date, not ingest time
            result["published"] += 1
            print(f"  published {show.id}: {ep.title}")
        except Exception as e:  # publish/ledger I/O error on ONE item must not abort the rest
            print(f"  FAILED {show.id}/{ep.guid}: {e}")
            result["failed"] += 1
        finally:
            storage.cleanup_episode_cache(settings, show.id, ep_id)   # free just this item's work dir
    try:
        evicted = retention.apply_retention(
            settings.state_path, show.id, settings.retention_keep, settings.retention_max_age_days)
        if evicted:
            print(f"  retention: removed {len(evicted)} old item(s) from {show.id}")
    except Exception as e:  # retention bookkeeping failure for one show must not abort others
        print(f"  WARNING retention failed for {show.id}: {e}")
    return result


def run_watch(settings, shows) -> dict:
    """One pass over all enabled shows. Returns {published, failed, skipped_locked}.
    A held lock (another run in progress) is a no-op, not a failure."""
    totals = {"published": 0, "failed": 0, "skipped_locked": False}
    # Lock lives next to the ledger, NOT under cache_dir (which `polyglot cleanup` wipes).
    with _lock(settings.state_path.parent / ".watch.lock") as acquired:
        if not acquired:
            print("another watch run is in progress; skipping this pass")
            totals["skipped_locked"] = True
            return totals
        for show in shows:
            if not show.enabled:
                continue
            try:
                r = _process_show(show, settings, shows)
            except Exception as e:  # last-resort guard: one show can never abort the others
                print(f"  ERROR processing {show.id}: {e}")
                r = {"published": 0, "failed": 1}
            totals["published"] += r["published"]
            totals["failed"] += r["failed"]
    return totals


def watch() -> dict:
    return run_watch(load_settings(), load_shows())
