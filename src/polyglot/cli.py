import argparse
import hashlib
import sys

from polyglot import feeds, pipeline
from polyglot.config import build_job, load_settings, JobSpec
from polyglot.feeds import Episode


def cmd_show(show_id: str, settings=None, shows=None) -> int:
    try:
        job = build_job(show_id, settings=settings, shows=shows)
    except (FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    s = job.settings
    print(f"show_id        : {job.show_id}")
    print(f"title          : {job.title}")
    print(f"source         : {job.source}  ({job.source_type})")
    print(f"target_lang    : {job.target_lang}")
    print(f"prompt_path    : {job.prompt_path}")
    print(f"voice_refs     : {[str(p) for p in job.voice_refs] or '(built-in voice)'}")
    print(f"transcribe     : {s.transcribe_backend} ({s.mlx_whisper_repo})")
    print(f"translate      : {s.translate_backend} ({s.mlx_llm_repo})")
    print(f"tts            : {s.tts_backend} (device={s.tts_device})")
    return 0


def select_episode(episodes, latest: bool, url: str | None) -> Episode:
    if url:
        guid = "manual-" + hashlib.sha1(url.encode()).hexdigest()[:12]
        return Episode(guid=guid, title="(manual)", published=None, media_url=url)
    if not episodes:
        raise ValueError("no episodes found in feed")
    return episodes[0]  # feeds are newest-first


def cmd_run(show_id: str, latest: bool, url: str | None, file: str | None,
            clip_seconds: int | None) -> int:
    job = build_job(show_id)
    if clip_seconds is not None:
        job.settings.clip_seconds = clip_seconds
    if file:
        ep = Episode(guid=f"file-{file}", title="(file)", published=None, media_url=file)
    else:
        episodes = feeds.list_episodes(job, limit=5)
        ep = select_episode(episodes, latest=latest, url=url)
    result = pipeline.process_episode(job, ep, job.settings)
    if result["ok"]:
        print(f"OK  mp3: {result['mp3']}")
        print(f"    srt: {result['srt']}")
        print(f"    duration: {result['duration']:.1f}s")
        return 0
    print(f"FAILED: {result['error']}", file=sys.stderr)
    return 1


def cmd_video(url: str, lang: str, clip_seconds: int | None, speakers: int | None) -> int:
    settings = load_settings()
    if clip_seconds is not None:
        settings.clip_seconds = clip_seconds
    settings.num_speakers = speakers if speakers is not None else 1  # solo narrator by default
    settings.tts_speed = 1.0  # video fits each line to its slot in assemble; no global speed-up
    prompt = settings.prompts_dir / f"{lang}.txt"
    if not prompt.is_file():
        print(f"error: prompt for lang '{lang}' missing: {prompt}", file=sys.stderr)
        return 1
    job = JobSpec("video", "Video", url, "youtube", lang, prompt, [], settings)
    guid = "yt-" + hashlib.sha1(url.encode()).hexdigest()[:12]
    ep = Episode(guid=guid, title="(video)", published=None, media_url=url)
    result = pipeline.process_video(job, ep, settings)
    if result["ok"]:
        print(f"OK  mp4: {result['mp4']}")
        print(f"    srt: {result['srt']}")
        print(f"    duration: {result['duration']:.1f}s")
        return 0
    print(f"FAILED: {result['error']}", file=sys.stderr)
    return 1


def cmd_watch() -> int:
    """Process new episodes/videos for all enabled shows -> Jellyfin library +
    retention. Designed to run from launchd/cron; exit non-zero if any item failed
    so the scheduler/log surfaces it."""
    from polyglot import watch
    res = watch.watch()
    print(f"watch: published={res['published']} failed={res['failed']}")
    return 1 if res["failed"] else 0


def cmd_cleanup() -> int:
    from polyglot import storage
    storage.cleanup_cache(load_settings())
    print("cache cleaned")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="polyglot")
    sub = parser.add_subparsers(dest="command")

    p_show = sub.add_parser("show", help="print resolved JobSpec for a show")
    p_show.add_argument("show_id")

    p_run = sub.add_parser("run", help="process one episode end-to-end")
    p_run.add_argument("show_id")
    p_run.add_argument("--latest", action="store_true")
    p_run.add_argument("--url")
    p_run.add_argument("--file")
    p_run.add_argument("--clip-seconds", type=int, default=None)

    p_video = sub.add_parser("video", help="dub a YouTube video -> mp4")
    p_video.add_argument("url")
    p_video.add_argument("--lang", default="fr")
    p_video.add_argument("--clip-seconds", type=int, default=None)
    p_video.add_argument("--speakers", type=int, default=None)

    sub.add_parser("watch", help="process new items for all enabled shows -> library (cron)")
    sub.add_parser("cleanup", help="purge transient cache/")

    args = parser.parse_args()

    if args.command == "show":
        return cmd_show(args.show_id)
    if args.command == "run":
        return cmd_run(args.show_id, args.latest, args.url, args.file, args.clip_seconds)
    if args.command == "video":
        return cmd_video(args.url, args.lang, args.clip_seconds, args.speakers)
    if args.command == "watch":
        return cmd_watch()
    if args.command == "cleanup":
        return cmd_cleanup()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
