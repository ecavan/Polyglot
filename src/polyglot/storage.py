import shutil

from polyglot.config import Settings


def cleanup_cache(settings: Settings) -> None:
    """Remove transient per-episode working files under cache/ (separation stems,
    segment clips, downloads). The published library + ledger are untouched."""
    cache = settings.cache_dir
    if not cache.exists():
        return
    for p in cache.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
