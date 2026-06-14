import re
import traceback

from polyglot import assemble, download, subtitles, transcribe, translate, tts
from polyglot.config import JobSpec, Settings
from polyglot.feeds import Episode


def _safe_id(guid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", guid)[:120]


def process_episode(job: JobSpec, episode: Episode, settings: Settings) -> dict:
    ep_id = _safe_id(episode.guid)
    work = settings.cache_dir / job.show_id / ep_id
    audio_dir = settings.output_dir / "audio" / job.show_id
    subs_dir = settings.output_dir / "subs" / job.show_id
    out_mp3 = audio_dir / f"{ep_id}.mp3"
    try:
        wav = download.fetch_audio(episode.media_url, work, settings.clip_seconds)
        segments = transcribe.transcribe(wav, settings)
        segments = translate.translate(segments, job, settings)   # loads LLM, frees it
        segments = tts.synthesize(segments, job, settings, work / "segments")
        audio = assemble.assemble(segments, out_mp3, settings)
        subtitles.write_subs(segments, audio.timeline, subs_dir, job.show_id, ep_id)
        return {
            "ok": True,
            "mp3": str(out_mp3),
            "srt": str(subs_dir / f"{ep_id}.srt"),
            "duration": audio.duration,
            "byte_length": audio.byte_length,
        }
    except Exception as e:  # episode isolation: log + skip
        traceback.print_exc()
        return {"ok": False, "error": str(e), "guid": episode.guid}
