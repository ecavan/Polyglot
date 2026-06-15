import shutil

from polyglot.config import Settings


def cleanup_cache(settings: Settings) -> None:
    """Remove ALL transient per-episode working files under cache/ (manual `cleanup`).
    The published library + ledger are untouched. NOTE: the watch loop must NOT use this
    — a concurrent run's in-flight work dir lives here too; it uses cleanup_episode_cache."""
    cache = settings.cache_dir
    if not cache.exists():
        return
    for p in cache.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def cleanup_episode_cache(settings: Settings, show_id: str, ep_id: str) -> None:
    """Remove just ONE episode's working dir (cache/<show>/<ep>). Scoped so an
    overlapping watch run never deletes another run's in-flight files."""
    work = settings.cache_dir / show_id / ep_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
