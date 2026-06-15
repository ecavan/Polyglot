import polyglot.watch as W
from polyglot import state
from polyglot.config import ShowConfig, JobSpec
from polyglot.feeds import Episode
from tests.test_pipeline import _settings


def test_run_watch_publishes_and_is_idempotent(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.retention_keep = 2          # list size == keep -> no churn
    shows = [ShowConfig("s", "Show", "u", "rss", "fr", None, True),
             ShowConfig("off", "Off", "u", "rss", "fr", None, False)]  # disabled -> skipped

    job = JobSpec("s", "Show", "u", "rss", "fr", tmp_path / "fr.txt", [], settings)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    monkeypatch.setattr(W.feeds, "list_episodes",
                        lambda job, limit, max_minutes: [
                            Episode("g1", "Ep One", None, "u1"),
                            Episode("g2", "Ep Two", None, "u2"),
                        ])

    def fake_proc(job, ep, settings):
        mp3 = tmp_path / f"{ep.guid}.mp3"; mp3.write_text("audio")
        mp4 = tmp_path / f"{ep.guid}.tv.mp4"; mp4.write_text("tv")
        srt = tmp_path / f"{ep.guid}.srt"; srt.write_text("subs")
        return {"ok": True, "mp3": str(mp3), "tv_mp4": str(mp4),
                "media": [str(mp3), str(mp4)], "srt": str(srt)}
    monkeypatch.setattr(W.pipeline, "process_episode", fake_proc)

    res = W.run_watch(settings, shows)
    assert res["published"] == 2                           # both new items dubbed + published
    assert res["failed"] == 0
    assert len(state.published(settings.state_path, "s")) == 2
    libdir = settings.library_path / "Podcasts" / "Show"
    assert len(list(libdir.glob("*.mp3"))) == 2           # phone copies
    assert len(list(libdir.glob("*.mp4"))) == 2           # TV copies (burned subs)
    assert len(list(libdir.glob("*.srt"))) == 2           # bilingual subtitle alongside each

    assert W.run_watch(settings, shows)["published"] == 0  # second pass: nothing new (idempotent)


def _one_show(tmp_path):
    settings = _settings(tmp_path)
    settings.retention_keep = 2
    shows = [ShowConfig("s", "Show", "u", "rss", "fr", None, True)]
    job = JobSpec("s", "Show", "u", "rss", "fr", tmp_path / "fr.txt", [], settings)
    return settings, shows, job


def test_run_watch_isolates_feed_listing_failure(tmp_path, monkeypatch):
    settings, shows, job = _one_show(tmp_path)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)

    def boom(job, limit, max_minutes):
        raise RuntimeError("yt-dlp exploded")
    monkeypatch.setattr(W.feeds, "list_episodes", boom)

    res = W.run_watch(settings, shows)                     # must not raise
    assert res == {"published": 0, "failed": 0, "skipped_locked": False}


def test_run_watch_counts_failed_items_without_marking_done(tmp_path, monkeypatch):
    settings, shows, job = _one_show(tmp_path)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    monkeypatch.setattr(W.feeds, "list_episodes",
                        lambda job, limit, max_minutes: [Episode("g1", "Ep", None, "u1")])
    monkeypatch.setattr(W.pipeline, "process_episode",
                        lambda job, ep, settings: {"ok": False, "error": "tts blew up"})

    res = W.run_watch(settings, shows)
    assert res["failed"] == 1 and res["published"] == 0
    assert W.state.is_done(settings.state_path, "s", "g1") is False  # failed item retried next time


def test_run_watch_skips_when_locked(tmp_path, monkeypatch):
    import fcntl
    settings, shows, job = _one_show(tmp_path)
    lock_path = settings.state_path.parent / ".watch.lock"   # lock lives next to the ledger
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = open(lock_path, "w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)     # simulate a run already in progress
    try:
        res = W.run_watch(settings, shows)
        assert res["skipped_locked"] is True
    finally:
        holder.close()


def test_run_watch_releases_lock_after_normal_run(tmp_path, monkeypatch):
    import fcntl
    settings, shows, job = _one_show(tmp_path)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    monkeypatch.setattr(W.feeds, "list_episodes", lambda job, limit, max_minutes: [])
    W.run_watch(settings, shows)                          # normal pass
    f = open(settings.state_path.parent / ".watch.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)     # lock must be free again
    finally:
        f.close()


def _writing_proc(tmp_path):
    def proc(job, ep, settings):
        mp3 = tmp_path / f"{ep.guid}.mp3"; mp3.write_text("a")
        mp4 = tmp_path / f"{ep.guid}.tv.mp4"; mp4.write_text("v")
        srt = tmp_path / f"{ep.guid}.srt"; srt.write_text("s")
        return {"ok": True, "mp3": str(mp3), "tv_mp4": str(mp4),
                "media": [str(mp3), str(mp4)], "srt": str(srt), "files": []}
    return proc


def test_run_watch_purge_path_no_redub(tmp_path, monkeypatch):
    settings, shows, job = _one_show(tmp_path)
    settings.retention_keep = 1                            # keep only the newest -> older purged
    settings.retention_max_age_days = 36500                # isolate the keep-N rule from age
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    feed = [Episode("g_new", "New", None, "u2", published_ts=200.0),
            Episode("g_old", "Old", None, "u1", published_ts=100.0)]
    monkeypatch.setattr(W.feeds, "list_episodes", lambda job, limit, max_minutes: feed)
    monkeypatch.setattr(W.pipeline, "process_episode", _writing_proc(tmp_path))

    r1 = W.run_watch(settings, shows)
    assert r1["published"] == 2
    libdir = settings.library_path / "Podcasts" / "Show"
    assert len(list(libdir.glob("*.mp3"))) == 1           # older one purged by retention
    assert W.state.is_done(settings.state_path, "s", "g_old") is True  # ...but still seen

    r2 = W.run_watch(settings, shows)                     # both still in feed
    assert r2["published"] == 0                           # purged item is NOT re-dubbed
    assert len(list(libdir.glob("*.mp3"))) == 1


def test_run_watch_uses_scoped_cache_cleanup_not_destructive(tmp_path, monkeypatch):
    settings, shows, job = _one_show(tmp_path)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    monkeypatch.setattr(W.feeds, "list_episodes",
                        lambda job, limit, max_minutes: [Episode("g1", "Ep", None, "u1")])
    monkeypatch.setattr(W.pipeline, "process_episode", _writing_proc(tmp_path))
    scoped = []
    monkeypatch.setattr(W.storage, "cleanup_episode_cache",
                        lambda s, sid, eid: scoped.append((sid, eid)))
    def boom(_s):
        raise AssertionError("run_watch must NOT wipe all of cache/")
    monkeypatch.setattr(W.storage, "cleanup_cache", boom)

    W.run_watch(settings, shows)
    assert scoped == [("s", "g1")]                         # per-episode scoped cleanup only


def test_run_watch_publish_failure_isolated(tmp_path, monkeypatch):
    settings, shows, job = _one_show(tmp_path)
    monkeypatch.setattr(W, "build_job", lambda sid, s, sh: job)
    monkeypatch.setattr(W.feeds, "list_episodes",
                        lambda job, limit, max_minutes: [Episode("g1", "Ep", None, "u1")])
    monkeypatch.setattr(W.pipeline, "process_episode", _writing_proc(tmp_path))
    def explode(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(W.library, "publish_to_library", explode)

    res = W.run_watch(settings, shows)                     # must not raise
    assert res["failed"] == 1 and res["published"] == 0
    assert W.state.is_done(settings.state_path, "s", "g1") is False  # not marked done -> retried
