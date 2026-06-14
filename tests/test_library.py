from polyglot.library import safe_name, publish_to_library


class _S:
    def __init__(self, lib):
        self.library_path = lib


def test_safe_name():
    assert safe_name("PTI: Knicks?! / Game 5") == "PTI_ Knicks__ _ Game 5"
    assert safe_name("") == "episode"


def test_publish_to_library_places_media_and_srt(tmp_path):
    media = tmp_path / "dub.mp4"
    media.write_text("video")
    srt = tmp_path / "ep.srt"
    srt.write_text("subs")
    settings = _S(tmp_path / "lib")
    out = publish_to_library("video", "GothamChess (FR)", "0 Elo Chess", media, srt, settings)
    media_dest, srt_dest = out
    assert media_dest.parent == tmp_path / "lib" / "Videos" / "GothamChess _FR_"
    assert media_dest.name == "0 Elo Chess.mp4"
    assert srt_dest.name == "0 Elo Chess.srt"   # same basename -> Jellyfin auto-loads
    assert media_dest.read_text() == "video"
    assert srt_dest.read_text() == "subs"
