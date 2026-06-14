from pathlib import Path

import numpy as np

import polyglot.pipeline as pipeline
from polyglot.config import JobSpec, Settings
from polyglot.feeds import Episode
from polyglot.segments import new_segment


def _settings(tmp_path) -> Settings:
    return Settings(
        transcribe_backend="mlx-whisper", mlx_whisper_repo="r", faster_whisper="m",
        translate_backend="mlx", mlx_llm_repo="r", ollama_model="m", ollama_url="u",
        tts_backend="xtts", tts_device="cpu", voice_mode="pool",
        orpheus_gguf=tmp_path / "orph.gguf", orpheus_voices=["Pierre"],
        orpheus_temperature=0.6, orpheus_max_tokens=1800,
        tts_temperature=0.82, tts_repetition_penalty=5.0, tts_top_p=0.9,
        tts_length_penalty=1.0, tts_speed=1.0,
        voice_pool=["Damien Black", "Claribel Dervla"],
        num_speakers=2, diarize_threshold=0.6,
        separate_enabled=False, separate_device="cpu", separate_segment=15,
        mix_bed=False, bed_gain=0.3,
        cache_dir=tmp_path / "cache", output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices", prompts_dir=tmp_path / "prompts",
        state_path=tmp_path / "state.json",
        hosting_type="r2", public_base_url="x", bucket="b",
        library_path=tmp_path / "lib", retention_keep=10, retention_max_age_days=7,
        clip_seconds=0, max_video_minutes=60, diarize=False, temperature=0.3,
        max_tokens=512, gap_ms=200,
    )


def test_process_episode_writes_outputs(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "fr.txt").write_text("p")
    job = JobSpec("pti-fr", "PTI", "src", "rss", "fr",
                  tmp_path / "prompts" / "fr.txt", [], s)
    ep = Episode(guid="g1", title="Ep 1", published=None, media_url="http://x/ep.mp3")

    monkeypatch.setattr(pipeline.download, "fetch_audio", lambda *a, **k: tmp_path / "in.wav")
    monkeypatch.setattr(pipeline.download, "to_16k_mono", lambda *a, **k: tmp_path / "in16.wav")

    def fake_transcribe(wav, settings):
        return [new_segment(0, 0, 1, "Hello"), new_segment(1, 1, 2, "Bye")]
    monkeypatch.setattr(pipeline.transcribe, "transcribe", fake_transcribe)

    def fake_translate(segs, job, settings):
        for sg in segs:
            sg["translation"] = "FR-" + sg["text"]
        return segs
    monkeypatch.setattr(pipeline.translate, "translate", fake_translate)

    def fake_synth(segs, job, settings, out_dir, source_wav=None):
        import soundfile as sf
        out_dir.mkdir(parents=True, exist_ok=True)
        for sg in segs:
            p = out_dir / f"seg_{sg['index']}.wav"
            sf.write(str(p), np.zeros(24000, dtype=np.float32), 24000, subtype="FLOAT")
            sg["audio_path"] = str(p)
            sg["audio_dur"] = 1.0
        return segs
    monkeypatch.setattr(pipeline.tts, "synthesize", fake_synth)

    out = pipeline.process_episode(job, ep, s)
    assert out["ok"] is True
    assert Path(out["mp3"]).exists()
    assert Path(out["srt"]).exists()
