import polyglot.cli as cli
from polyglot import watch


def test_cmd_watch_exit_codes(monkeypatch):
    # success -> 0
    monkeypatch.setattr(watch, "watch", lambda: {"published": 2, "failed": 0, "skipped_locked": False})
    assert cli.cmd_watch() == 0
    # any failure -> non-zero so launchd/cron surfaces it
    monkeypatch.setattr(watch, "watch", lambda: {"published": 1, "failed": 1, "skipped_locked": False})
    assert cli.cmd_watch() == 1
    # skipped because another run held the lock is NOT a failure -> 0
    monkeypatch.setattr(watch, "watch", lambda: {"published": 0, "failed": 0, "skipped_locked": True})
    assert cli.cmd_watch() == 0
