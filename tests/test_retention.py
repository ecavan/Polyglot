from polyglot import retention, state


def _items(n, base_ts):
    return [{"guid": f"g{i}", "published_at": base_ts + i, "files": []} for i in range(n)]


def test_to_evict_caps_to_keep():
    now = 1000.0
    items = _items(12, now - 12)          # 12 recent items
    evict = retention.to_evict(items, keep=10, max_age_days=7, now=now)
    assert len(evict) == 2                # the 2 oldest beyond the cap
    assert {e["guid"] for e in evict} == {"g0", "g1"}


def test_to_evict_age_purges_even_within_cap():
    now = 1_000_000.0
    items = [
        {"guid": "old", "published_at": now - 10 * 86400, "files": []},   # 10 days -> too old
        {"guid": "new", "published_at": now - 1 * 86400, "files": []},
    ]
    evict = retention.to_evict(items, keep=10, max_age_days=7, now=now)
    assert [e["guid"] for e in evict] == ["old"]


def test_apply_retention_deletes_files_but_keeps_seen(tmp_path):
    p = tmp_path / "processed.json"
    old_file = tmp_path / "old.mp3"
    old_file.write_text("x")
    keep_file = tmp_path / "keep.mp3"
    keep_file.write_text("y")
    state.mark_done(p, "s", "old", "audio", [old_file], ts=100.0)
    state.mark_done(p, "s", "keep", "audio", [keep_file], ts=200.0)
    retention.apply_retention(p, "s", keep=1, max_age_days=3650, now=300.0)
    assert not old_file.exists()          # evicted file deleted from disk (freed)
    assert keep_file.exists()
    # evicted item is gone from the LIVE library...
    assert [i["guid"] for i in state.published(p, "s")] == ["keep"]
    # ...but stays "seen" so a still-in-feed episode is never re-dubbed
    assert state.is_done(p, "s", "old") is True
    assert state.is_done(p, "s", "keep") is True
