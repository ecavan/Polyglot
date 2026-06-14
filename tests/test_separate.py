from pathlib import Path

from polyglot.separate import stem_paths


def test_stem_paths():
    v, nv = stem_paths(Path("/x/source_44k.wav"), Path("/work/sep"))
    assert v == Path("/work/sep/htdemucs/source_44k/vocals.wav")
    assert nv == Path("/work/sep/htdemucs/source_44k/no_vocals.wav")
