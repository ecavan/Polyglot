import json
import time
from pathlib import Path


def _load(path: Path) -> dict:
    if Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"items": []}


def _save(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_done(state_path: Path, show_id: str, guid: str) -> bool:
    return any(i["show_id"] == show_id and i["guid"] == guid for i in _load(state_path)["items"])


def mark_done(state_path: Path, show_id: str, guid: str, kind: str,
              files: list, title: str = "", ts: float | None = None) -> None:
    data = _load(state_path)
    if any(i["show_id"] == show_id and i["guid"] == guid for i in data["items"]):
        return
    data["items"].append({
        "show_id": show_id, "guid": guid, "kind": kind, "title": title,
        "files": [str(f) for f in files],
        "published_at": ts if ts is not None else time.time(),
    })
    _save(state_path, data)


def published(state_path: Path, show_id: str) -> list[dict]:
    return [i for i in _load(state_path)["items"] if i["show_id"] == show_id]


def remove(state_path: Path, show_id: str, guid: str) -> None:
    data = _load(state_path)
    data["items"] = [i for i in data["items"]
                     if not (i["show_id"] == show_id and i["guid"] == guid)]
    _save(state_path, data)
