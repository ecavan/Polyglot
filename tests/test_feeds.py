from pathlib import Path

from polyglot.feeds import list_episodes_from_url, _yyyymmdd_to_epoch

FIXTURE = Path(__file__).parent / "fixtures" / "feed.xml"


def test_list_episodes_parses_rss():
    eps = list_episodes_from_url(FIXTURE.as_uri(), limit=None)
    assert len(eps) == 2
    assert eps[0].guid == "guid-2"
    assert eps[0].title == "Episode Two"
    assert eps[0].media_url == "https://example.com/ep2.mp3"
    assert eps[0].published is not None
    assert eps[0].published_ts is not None and eps[0].published_ts > 0  # real air date parsed


def test_yyyymmdd_to_epoch():
    assert _yyyymmdd_to_epoch("20260613") == 1781308800   # 2026-06-13 00:00:00 UTC
    assert _yyyymmdd_to_epoch(None) is None
    assert _yyyymmdd_to_epoch("garbage") is None


def test_list_episodes_respects_limit():
    eps = list_episodes_from_url(FIXTURE.as_uri(), limit=1)
    assert len(eps) == 1
    assert eps[0].guid == "guid-2"


def test_broken_feed_returns_empty_not_crash(tmp_path):
    # a dead/unreachable/malformed feed must yield [] (treated as "no new"), never raise
    missing = (tmp_path / "nope.xml").as_uri()
    assert list_episodes_from_url(missing, limit=None) == []
    bad = tmp_path / "bad.xml"
    bad.write_text("<rss><channel><item>truncated", encoding="utf-8")
    assert list_episodes_from_url(bad.as_uri(), limit=None) == []
