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

    published = W.run_watch(settings, shows)
    assert published == 2                                  # both new items dubbed + published
    assert len(state.published(settings.state_path, "s")) == 2
    libdir = settings.library_path / "Podcasts" / "Show"
    assert len(list(libdir.glob("*.mp3"))) == 2           # phone copies
    assert len(list(libdir.glob("*.mp4"))) == 2           # TV copies (burned subs)
    assert len(list(libdir.glob("*.srt"))) == 2           # bilingual subtitle alongside each

    assert W.run_watch(settings, shows) == 0              # second pass: nothing new (idempotent)
