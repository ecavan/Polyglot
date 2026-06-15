import json
import time
from pathlib import Path


def _load(path: Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"items": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("ledger missing items[]")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as e:
        # A corrupt/empty ledger must NOT brick the unattended loop. Quarantine it and
        # start fresh; worst case we re-dub recent items (idempotency is best-effort here).
        backup = p.with_name(f"{p.name}.corrupt-{int(p.stat().st_mtime)}")
        print(f"WARNING ledger {p} unreadable ({e}); quarantined to {backup.name}, starting fresh")
        try:
            p.replace(backup)
        except OSError:
            pass
        return {"items": []}


def _save(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _find(data: dict, show_id: str, guid: str) -> dict | None:
    for i in data["items"]:
        if i["show_id"] == show_id and i["guid"] == guid:
            return i
    return None


def is_done(state_path: Path, show_id: str, guid: str) -> bool:
    """True if we have EVER processed this item (live or purged). This is the
    idempotency gate: it stays True forever so retention freeing the files never
    makes an old, still-in-feed item look new and trigger a re-dub."""
    return _find(_load(state_path), show_id, guid) is not None


def mark_done(state_path: Path, show_id: str, guid: str, kind: str,
              files: list, title: str = "", ts: float | None = None) -> None:
    data = _load(state_path)
    if _find(data, show_id, guid) is not None:
        return
    data["items"].append({
        "show_id": show_id, "guid": guid, "kind": kind, "title": title,
        "files": [str(f) for f in files],
        "published_at": ts if ts is not None else time.time(),
        "purged": False,
    })
    _save(state_path, data)


def published(state_path: Path, show_id: str) -> list[dict]:
    """LIVE items for a show (files still on disk). Retention operates on these;
    purged items are excluded but remain in the ledger for idempotency."""
    return [i for i in _load(state_path)["items"]
            if i["show_id"] == show_id and not i.get("purged")]


def mark_purged(state_path: Path, show_id: str, guid: str) -> None:
    """Retention eviction: keep the ledger entry (so is_done stays True) but mark
    it purged and forget its files — the files themselves are deleted by the caller."""
    data = _load(state_path)
    item = _find(data, show_id, guid)
    if item is None or item.get("purged"):
        return
    item["purged"] = True
    item["files"] = []
    _save(state_path, data)


def remove(state_path: Path, show_id: str, guid: str) -> None:
    """Hard-delete a ledger entry entirely (forgets it was ever seen)."""
    data = _load(state_path)
    data["items"] = [i for i in data["items"]
                     if not (i["show_id"] == show_id and i["guid"] == guid)]
    _save(state_path, data)
