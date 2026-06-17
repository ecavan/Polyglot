"""On-demand dub queue + worker, behind the Streamlit control panel.

The app appends jobs (a YouTube URL, or "latest episode" of a podcast) to a JSON queue; a
single worker drains them one at a time (so the Mac isn't thrashed) and can run detached
overnight. Heavy pipeline imports are lazy so importing this for the UI stays fast.
"""
import fcntl
import hashlib
import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from polyglot import download, feeds, library, state, storage
from polyglot.config import JobSpec, build_job, load_settings, load_shows
from polyglot.feeds import Episode


# ---------------------------------------------------------------- queue file

def _queue_path(settings) -> Path:
    return settings.state_path.parent / "queue.json"


@contextmanager
def _queue_lock(settings):
    p = settings.state_path.parent / ".queue.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _read(settings) -> dict:
    p = _queue_path(settings)
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("jobs"), list):
                return d
        except (json.JSONDecodeError, OSError):
            pass
    return {"jobs": []}


def _mutate(settings, fn):
    with _queue_lock(settings):
        data = _read(settings)
        result = fn(data)
        p = _queue_path(settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)
        return result


def _job(**kw) -> dict:
    base = {"id": hashlib.sha1(f"{kw}{time.time()}".encode()).hexdigest()[:10],
            "status": "pending", "error": "", "added": time.time(), "finished": None}
    base.update(kw)
    return base


def list_jobs(settings) -> list[dict]:
    return _read(settings)["jobs"]


def add_video(settings, url, title="", channel="", video_id="", duration=0,
              published_ts=None, speakers=1) -> dict:
    job = _job(type="video", url=url, title=title or url, channel=channel or "YouTube",
               video_id=video_id, duration=duration, published_ts=published_ts, speakers=speakers)
    _mutate(settings, lambda d: d["jobs"].append(job))
    return job


def add_podcast(settings, show_id, title="") -> dict:
    job = _job(type="podcast", show_id=show_id, title=title or show_id)
    _mutate(settings, lambda d: d["jobs"].append(job))
    return job


def remove_job(settings, job_id):
    _mutate(settings, lambda d: d.__setitem__("jobs", [j for j in d["jobs"] if j["id"] != job_id]))


def clear_finished(settings):
    _mutate(settings, lambda d: d.__setitem__(
        "jobs", [j for j in d["jobs"] if j["status"] in ("pending", "running")]))


def _claim_next(settings):
    def fn(d):
        for j in d["jobs"]:
            if j["status"] == "pending":
                j["status"] = "running"
                return dict(j)
        return None
    return _mutate(settings, fn)


def _mark(settings, job_id, status, error=""):
    def fn(d):
        for j in d["jobs"]:
            if j["id"] == job_id:
                j["status"], j["error"], j["finished"] = status, error, time.time()
    _mutate(settings, fn)


# ---------------------------------------------------------------- library view / delete

def library_items(settings) -> list[dict]:
    """Live (non-purged) ledger items with a computed on-disk size, for the app's Library tab."""
    out = []
    for it in state._load(settings.state_path)["items"]:
        if it.get("purged"):
            continue
        size = sum(Path(f).stat().st_size for f in it.get("files", []) if Path(f).exists())
        out.append({**it, "size_mb": size / 1e6})
    return out


def delete_item(settings, show_id, guid):
    """Delete an item's files (frees space) and drop it from the ledger so it can be re-pulled."""
    dirs = set()
    for it in state._load(settings.state_path)["items"]:
        if it["show_id"] == show_id and it["guid"] == guid:
            for f in it.get("files", []):
                p = Path(f)
                p.unlink(missing_ok=True)
                dirs.add(p.parent)
    state.remove(settings.state_path, show_id, guid)
    for d in dirs:                                  # tidy now-empty show folders
        try:
            if d.is_dir() and d != settings.library_path and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------- dubbing

def _publish_and_record(settings, kind, show_id, show_title, ep, res, ep_id):
    media = res.get("media") or [res.get("mp4") or res.get("mp3")]
    files = library.publish_to_library(kind, show_title, ep.title, media, settings,
                                       ep_id=ep_id, lrc_src=res.get("lrc"))
    files += res.get("files", [])
    state.mark_done(settings.state_path, show_id, ep.guid, kind, files, ep.title, ts=ep.published_ts)


def _run_video(settings, job):
    from polyglot import pipeline
    settings.num_speakers = max(1, int(job.get("speakers") or 1))   # 1 = solo narrator; >1 = e.g. poker
    settings.tts_speed = 1.0
    url = job["url"]
    vid = job.get("video_id") or "yt-" + hashlib.sha1(url.encode()).hexdigest()[:12]
    show_id = "video"
    if state.is_done(settings.state_path, show_id, vid):
        return
    js = JobSpec(show_id, job.get("channel", "YouTube"), url, "youtube", "fr",
                 settings.prompts_dir / "fr.txt", [], settings)
    ep = Episode(guid=vid, title=job.get("title") or "(video)", published=None, media_url=url,
                 published_ts=job.get("published_ts"))
    ep_id = pipeline._safe_id(vid)
    try:
        res = pipeline.process_video(js, ep, settings)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "dub failed"))
        _publish_and_record(settings, "video", show_id, job.get("channel", "YouTube"), ep, res, ep_id)
    finally:
        storage.cleanup_episode_cache(settings, show_id, ep_id)


def _run_podcast(settings, job):
    from polyglot import pipeline
    show_id = job["show_id"]
    shows = load_shows()
    show = next((s for s in shows if s.id == show_id), None)
    if show is None:
        raise RuntimeError(f"unknown show {show_id}")
    js = build_job(show_id, settings, shows)
    eps = [e for e in feeds.list_episodes(js, limit=3, max_minutes=settings.max_video_minutes)
           if not state.is_done(settings.state_path, show_id, e.guid)]
    if not eps:
        return                                                  # nothing new
    ep = eps[0]
    ep_id = pipeline._safe_id(ep.guid)
    try:
        res = pipeline.process_episode(js, ep, settings)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "dub failed"))
        _publish_and_record(settings, "audio", show_id, show.title, ep, res, ep_id)
    finally:
        storage.cleanup_episode_cache(settings, show_id, ep_id)


def run_one(settings, job):
    if job["type"] == "video":
        _run_video(settings, job)
    else:
        _run_podcast(settings, job)


# ---------------------------------------------------------------- worker

def _load_secrets():
    """launchd / detached worker has no shell profile — load API keys from the env file."""
    p = Path.home() / ".config" / "polyglot" / "env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        line = line[7:] if line.startswith("export ") else line
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            k = k.strip()
            if not os.environ.get(k):      # fill if missing OR empty (the file is the source of truth)
                os.environ[k] = v.strip().strip('"').strip("'")


@contextmanager
def _worker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        f.close()


def drain() -> int:
    """Process every pending job, one at a time, until the queue is empty. Single-instance."""
    _load_secrets()
    base = load_settings()
    with _worker_lock(base.state_path.parent / ".worker.lock") as acquired:
        if not acquired:
            print("[worker] another worker is already running; exiting")
            return 0
        # reclaim any job a previously-crashed/killed worker left mid-flight (we hold the lock,
        # so no other worker owns it) — otherwise it'd be stuck 'running' forever.
        def _reset(d):
            for j in d["jobs"]:
                if j["status"] == "running":
                    j["status"] = "pending"
        _mutate(base, _reset)
        n = 0
        while True:
            settings = load_settings()              # fresh per job (video mutates speaker/speed)
            job = _claim_next(settings)
            if not job:
                break
            label = job.get("title") or job.get("url") or job["type"]
            print(f"[worker] dubbing {job['type']}: {label}", flush=True)
            try:
                run_one(settings, job)
                _mark(settings, job["id"], "done")
                n += 1
                print(f"[worker] done: {label}", flush=True)
            except Exception as e:
                traceback.print_exc()
                _mark(settings, job["id"], "failed", str(e))
                print(f"[worker] FAILED: {label}: {e}", flush=True)
    print(f"[worker] processed {n} job(s)")
    return n


def worker_running(settings) -> bool:
    """True if a worker currently holds the lock (best-effort, non-blocking probe)."""
    p = settings.state_path.parent / ".worker.lock"
    if not p.exists():
        return False
    f = open(p, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        f.close()


def start_worker(settings):
    """Spawn a detached worker that drains the queue and exits (survives the browser tab)."""
    import subprocess
    import sys
    log = settings.state_path.parent / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([sys.executable, "-m", "polyglot.cli", "worker"],
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
