from polyglot.cli import cmd_show
from polyglot.config import load_settings, load_shows
from tests.test_config import SETTINGS_TOML, SHOWS_TOML


def _prep(tmp_path):
    sp = tmp_path / "settings.toml"
    sp.write_text(SETTINGS_TOML, encoding="utf-8")
    shp = tmp_path / "shows.toml"
    shp.write_text(SHOWS_TOML, encoding="utf-8")
    s = load_settings(sp)
    s.prompts_dir = tmp_path / "prompts"
    s.voices_dir = tmp_path / "voices"
    s.prompts_dir.mkdir()
    s.voices_dir.mkdir()
    (s.prompts_dir / "fr.txt").write_text("p", encoding="utf-8")
    (s.voices_dir / "fr_montreal.wav").write_bytes(b"RIFF")
    return s, load_shows(shp)


def test_cmd_show_prints_jobspec(tmp_path, capsys):
    s, shows = _prep(tmp_path)
    rc = cmd_show("pti-fr", settings=s, shows=shows)
    out = capsys.readouterr().out
    assert rc == 0
    assert "pti-fr" in out
    assert "fr" in out
    assert "fr.txt" in out
    assert "fr_montreal.wav" in out


def test_cmd_show_missing_voice_reports_error(tmp_path, capsys):
    s, shows = _prep(tmp_path)
    (s.voices_dir / "fr_montreal.wav").unlink()
    rc = cmd_show("pti-fr", settings=s, shows=shows)
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert rc != 0
    assert "voice" in combined
