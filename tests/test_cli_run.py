from polyglot.cli import select_episode
from polyglot.feeds import Episode


def _eps():
    return [
        Episode("g2", "Two", None, "http://x/2.mp3"),
        Episode("g1", "One", None, "http://x/1.mp3"),
    ]


def test_select_latest():
    ep = select_episode(_eps(), latest=True, url=None)
    assert ep.guid == "g2"


def test_select_by_url():
    ep = select_episode(_eps(), latest=False, url="http://manual/x.mp3")
    assert ep.media_url == "http://manual/x.mp3"
    assert ep.guid
