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


def test_publish_to_library_copies_multiple_media(tmp_path):
    mp3 = tmp_path / "dub.mp3"; mp3.write_text("audio")
    mp4 = tmp_path / "dub.tv.mp4"; mp4.write_text("tv")
    srt = tmp_path / "ep.srt"; srt.write_text("subs")
    settings = _S(tmp_path / "lib")
    out = publish_to_library("audio", "PTI (FR)", "Game 5", [mp3, mp4], srt, settings)
    dest_dir = tmp_path / "lib" / "Podcasts" / "PTI _FR_"
    names = sorted(p.name for p in out)
    assert names == ["Game 5.mp3", "Game 5.mp4", "Game 5.srt"]   # phone + TV + shared-basename subs
    assert (dest_dir / "Game 5.mp3").read_text() == "audio"
    assert (dest_dir / "Game 5.mp4").read_text() == "tv"


def test_publish_to_library_disambiguates_same_title(tmp_path):
    settings = _S(tmp_path / "lib")
    a = tmp_path / "a.mp4"; a.write_text("A")
    b = tmp_path / "b.mp4"; b.write_text("B")
    srt = tmp_path / "s.srt"; srt.write_text("subs")
    # two DIFFERENT episodes with the SAME human title must not collide / overwrite
    out_a = publish_to_library("video", "Show", "Daily", a, srt, settings, ep_id="guid-aaaa")
    out_b = publish_to_library("video", "Show", "Daily", b, srt, settings, ep_id="guid-bbbb")
    assert {p.name for p in out_a}.isdisjoint({p.name for p in out_b})  # no shared filenames
    assert all(p.exists() for p in out_a) and all(p.exists() for p in out_b)
