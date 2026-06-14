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
