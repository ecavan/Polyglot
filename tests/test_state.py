from polyglot import state


def test_state_ledger_roundtrip(tmp_path):
    p = tmp_path / "processed.json"
    assert state.is_done(p, "show", "g1") is False
    state.mark_done(p, "show", "g1", "audio", [tmp_path / "a.mp3"], title="Ep 1", ts=100.0)
    assert state.is_done(p, "show", "g1") is True
    state.mark_done(p, "show", "g1", "audio", [], ts=100.0)  # duplicate ignored
    items = state.published(p, "show")
    assert len(items) == 1
    assert items[0]["title"] == "Ep 1"
    assert items[0]["published_at"] == 100.0


def test_state_remove(tmp_path):
    p = tmp_path / "processed.json"
    state.mark_done(p, "s", "g1", "audio", [], ts=1.0)
    state.mark_done(p, "s", "g2", "audio", [], ts=2.0)
    state.remove(p, "s", "g1")
    assert state.is_done(p, "s", "g1") is False
    assert state.is_done(p, "s", "g2") is True


def test_corrupt_ledger_is_quarantined_not_fatal(tmp_path):
    p = tmp_path / "processed.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    assert state.is_done(p, "s", "g1") is False          # recovers instead of raising
    assert list(tmp_path.glob("processed.json.corrupt-*"))  # bad file quarantined
    state.mark_done(p, "s", "g1", "audio", [], ts=1.0)   # ledger is usable again
    assert state.is_done(p, "s", "g1") is True


def test_empty_ledger_file_treated_as_fresh(tmp_path):
    p = tmp_path / "processed.json"
    p.write_text("", encoding="utf-8")                    # 0-byte file (e.g. interrupted write)
    assert state.is_done(p, "s", "g1") is False


def test_mark_purged_keeps_seen_but_drops_from_live(tmp_path):
    p = tmp_path / "processed.json"
    state.mark_done(p, "s", "g1", "audio", [tmp_path / "a.mp3"], ts=1.0)
    state.mark_done(p, "s", "g2", "audio", [tmp_path / "b.mp3"], ts=2.0)
    state.mark_purged(p, "s", "g1")
    # purged item is NO LONGER live (retention won't re-count it, library forgets it)...
    live = state.published(p, "s")
    assert [i["guid"] for i in live] == ["g2"]
    # ...but is_done stays True forever, so a still-in-feed g1 is never re-dubbed
    assert state.is_done(p, "s", "g1") is True
    assert state.published(p, "s")[0]["files"] != []          # live item keeps its files
    # purged entry has its files forgotten (they were deleted on disk by the caller)
    purged = next(i for i in state._load(p)["items"] if i["guid"] == "g1")
    assert purged["purged"] is True and purged["files"] == []
