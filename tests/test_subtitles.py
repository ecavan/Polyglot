from pathlib import Path

from polyglot.subtitles import srt_timestamp, vtt_timestamp, build_srt, build_ass, write_subs
from polyglot.segments import new_segment


def test_build_ass_side_by_side():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    a = new_segment(0, 0, 0, "Hello"); a["translation"] = "Bonjour"
    out = build_ass([a], tl[:1])
    assert "Style: FR" in out and "Style: EN" in out      # two styled boxes
    assert ",FR,," in out and ",EN,," in out               # a dialogue line per language
    assert "Bonjour" in out and "Hello" in out


def test_srt_timestamp():
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(3661.5) == "01:01:01,500"


def test_vtt_timestamp():
    assert vtt_timestamp(3661.5) == "01:01:01.500"


def _segs():
    a = new_segment(0, 0, 0, "Hello")
    a["translation"] = "Bonjour"
    b = new_segment(1, 0, 0, "Bye")
    b["translation"] = "Salut"
    return [a, b]


def test_build_srt_bilingual():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    out = build_srt(_segs(), tl, bilingual=True)
    assert "1\n00:00:00,000 --> 00:00:01,000" in out
    assert "Bonjour" in out and "Hello" in out
    assert "2\n00:00:01,200 --> 00:00:02,000" in out


def test_build_srt_target_only_excludes_source():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    out = build_srt(_segs(), tl, bilingual=False)
    assert "Bonjour" in out
    assert "Hello" not in out


def test_speaker_prefix_only_when_multiple_speakers():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    one = _segs()
    for s in one:
        s["speaker"] = "SPEAKER_00"          # solo -> no prefix
    assert "SPEAKER_00:" not in build_srt(one, tl, bilingual=True)
    two = _segs()
    two[0]["speaker"] = "SPEAKER_00"
    two[1]["speaker"] = "SPEAKER_01"          # multi -> prefixes shown
    assert "SPEAKER_00:" in build_srt(two, tl, bilingual=True)


def test_write_subs_creates_four_files(tmp_path: Path):
    tl = [(0.0, 1.0), (1.2, 2.0)]
    write_subs(_segs(), tl, out_dir=tmp_path, show_id="s", ep_id="e")
    assert (tmp_path / "e.srt").exists()
    assert (tmp_path / "e.vtt").exists()
    assert (tmp_path / "e.target.srt").exists()
    assert (tmp_path / "e.target.vtt").exists()
