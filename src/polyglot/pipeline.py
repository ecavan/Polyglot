import re
import traceback

from polyglot import (
    assemble, diarize, download, publish_video,
    segments as segmod, separate, subtitles, transcribe, translate, tts,
)
from polyglot.config import JobSpec, Settings
from polyglot.feeds import Episode


def _safe_id(guid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", guid)[:120]


def _sub_files(subs_dir, ep_id: str) -> list[str]:
    """Every subtitle sidecar write_subs produces (so retention can delete them all)."""
    return [str(subs_dir / f"{ep_id}.{ext}")
            for ext in ("srt", "vtt", "target.srt", "target.vtt", "ass", "lrc")]


def dub_audio(src_audio, job: JobSpec, settings: Settings, work, out_audio,
              sync_to_source=False, source_duration=None):
    """Shared dub stages (podcast + video): separate -> transcribe -> diarize ->
    merge -> translate -> synthesize -> assemble (with music bed). Returns (segments, EpisodeAudio).
    sync_to_source=True (video) fits each line to its original time slot so audio tracks the picture."""
    bed = None
    speech_src = src_audio
    if settings.separate_enabled:
        speech_src, bed = separate.separate(src_audio, work / "sep", settings)  # vocals + music bed
    wav16 = download.to_16k_mono(speech_src, work)                # for whisper + diarizer
    segments = transcribe.transcribe(wav16, settings)
    if settings.diarize:
        segments = diarize.diarize(wav16, segments, settings)
    segments = segmod.merge_short_segments(segments)             # fuller phrases -> stable TTS
    segments = translate.translate(segments, job, settings)
    segments = tts.synthesize(segments, job, settings, work / "segments", source_wav=speech_src)
    audio = assemble.assemble(segments, out_audio, settings, bed_path=bed,
                              sync_to_source=sync_to_source, source_duration=source_duration)
    return segments, audio


def process_episode(job: JobSpec, episode: Episode, settings: Settings) -> dict:
    ep_id = _safe_id(episode.guid)
    work = settings.cache_dir / job.show_id / ep_id
    audio_dir = settings.output_dir / "audio" / job.show_id
    subs_dir = settings.output_dir / "subs" / job.show_id
    out_mp3 = audio_dir / f"{ep_id}.mp3"
    try:
        src = download.fetch_audio(episode.media_url, work, settings.clip_seconds)
        segments, audio = dub_audio(src, job, settings, work, out_mp3)
        subtitles.write_subs(segments, audio.timeline, subs_dir, job.show_id, ep_id)
        tv_mp4 = audio_dir / f"{ep_id}.tv.mp4"   # static-cover video w/ burned FR/EN subs, for the TV
        publish_video.make_audio_video(out_mp3, subs_dir / f"{ep_id}.ass", tv_mp4)
        files = [str(out_mp3), str(tv_mp4), *_sub_files(subs_dir, ep_id)]
        return {
            "ok": True, "mp3": str(out_mp3), "tv_mp4": str(tv_mp4),
            "media": [str(out_mp3), str(tv_mp4)],   # phone (mp3) + TV (mp4) -> both into library
            "srt": str(subs_dir / f"{ep_id}.srt"),
            "lrc": str(subs_dir / f"{ep_id}.lrc"),  # synced lyrics shipped beside the mp3
            "duration": audio.duration, "byte_length": audio.byte_length, "files": files,
        }
    except Exception as e:  # episode isolation
        traceback.print_exc()
        return {"ok": False, "error": str(e), "guid": episode.guid}


def process_video(job: JobSpec, episode: Episode, settings: Settings) -> dict:
    ep_id = _safe_id(episode.guid)
    work = settings.cache_dir / job.show_id / ep_id
    video_dir = settings.output_dir / "video" / job.show_id
    subs_dir = settings.output_dir / "subs" / job.show_id
    out_mp4 = video_dir / f"{ep_id}.mp4"
    try:
        video = download.fetch_video(
            episode.media_url, work, settings.clip_seconds, settings.max_video_minutes
        )
        src_audio = download.extract_audio(video, work)
        source_duration = assemble._audio_duration(src_audio)   # = video length
        segments, audio = dub_audio(src_audio, job, settings, work, work / "dub.mp3",
                                    sync_to_source=True, source_duration=source_duration)
        subtitles.write_subs(segments, audio.timeline, subs_dir, job.show_id, ep_id)
        styled_ass = subs_dir / f"{ep_id}.ass"                  # side-by-side FR/EN, burned in
        publish_video.mux(video, work / "dub.mp3", out_mp4, subtitle=styled_ass)
        files = [str(out_mp4), *_sub_files(subs_dir, ep_id)]
        return {
            "ok": True, "mp4": str(out_mp4), "media": [str(out_mp4)],
            "srt": str(subs_dir / f"{ep_id}.srt"),
            "duration": audio.duration, "files": files,
        }
    except Exception as e:  # episode isolation
        traceback.print_exc()
        return {"ok": False, "error": str(e), "guid": episode.guid}
