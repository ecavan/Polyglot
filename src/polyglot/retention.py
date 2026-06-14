import time
from pathlib import Path

from polyglot import state


def to_evict(items: list[dict], keep: int, max_age_days: int, now: float) -> list[dict]:
    """Which items to remove: anything beyond the newest `keep`, OR older than
    max_age_days. (An item is kept only if it's both within the newest N and recent.)"""
    by_new = sorted(items, key=lambda i: i.get("published_at", 0), reverse=True)
    cutoff = now - max_age_days * 86400
    evict = []
    for rank, item in enumerate(by_new):
        if rank >= keep or item.get("published_at", 0) < cutoff:
            evict.append(item)
    return evict


def apply_retention(state_path: Path, show_id: str, keep: int, max_age_days: int,
                    now: float | None = None) -> list[dict]:
    """Delete evicted items' files from the library and drop them from the ledger."""
    now = now if now is not None else time.time()
    evicted = to_evict(state.published(state_path, show_id), keep, max_age_days, now)
    for item in evicted:
        for f in item.get("files", []):
            Path(f).unlink(missing_ok=True)
        state.remove(state_path, show_id, item["guid"])
    return evicted
