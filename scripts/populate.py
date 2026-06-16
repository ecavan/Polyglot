"""Populate the Jellyfin library with the latest N items per show, on demand.

Like the watch loop but with explicit per-show counts, so you can fill the library to test
without waiting for the schedule. Publishes to the library AND marks items done (so the
scheduled watch won't re-dub them). Processes in the order given — list short items first.

Usage:
  uv run python scripts/populate.py nbc-news-fr:1 hustler-fr:2 tushi-fr:1
  # show_id:count   (count defaults to 1)
"""
import sys
import time

from polyglot import feeds, library, pipeline, state, storage
from polyglot.config import build_job, load_settings, load_shows


def main(argv: list[str]) -> int:
    settings = load_settings()
    shows = {s.id: s for s in load_shows()}
    orig_speakers, orig_speed = settings.num_speakers, settings.tts_speed

    specs = []
    for a in argv:
        sid, _, cnt = a.partition(":")
        specs.append((sid, int(cnt) if cnt else 1))

    done = 0
    for sid, count in specs:
        show = shows.get(sid)
        if not show:
            print(f"skip {sid}: not in shows.toml")
            continue
        # video = solo-narrator defaults (matches the tested `polyglot video` path);
        # podcasts keep the multi-host defaults.
        if show.source_type == "youtube":
            settings.num_speakers, settings.tts_speed = 1, 1.0
        else:
            settings.num_speakers, settings.tts_speed = orig_speakers, orig_speed

        job = build_job(sid, settings, list(shows.values()))
        eps = feeds.list_episodes(job, limit=count, max_minutes=settings.max_video_minutes)
        eps = [e for e in eps if not state.is_done(settings.state_path, sid, e.guid)][:count]
        print(f"\n=== {sid}: dubbing {len(eps)} item(s) ===", flush=True)
        for ep in eps:
            ep_id = pipeline._safe_id(ep.guid)
            t0 = time.time()
            try:
                if show.source_type == "youtube":
                    res = pipeline.process_video(job, ep, settings)
                    kind = "video"
                else:
                    res = pipeline.process_episode(job, ep, settings)
                    kind = "audio"
                if not res.get("ok"):
                    print(f"  FAILED {ep.title!r}: {res.get('error')}", flush=True)
                    continue
                media = res.get("media") or [res.get("mp4") or res.get("mp3")]
                files = library.publish_to_library(kind, show.title, ep.title, media,
                                                   settings, ep_id=ep_id, lrc_src=res.get("lrc"))
                files += res.get("files", [])
                state.mark_done(settings.state_path, sid, ep.guid, kind, files, ep.title,
                                ts=ep.published_ts)
                done += 1
                print(f"  published: {ep.title!r}  ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  ERROR {ep.title!r}: {e}", flush=True)
            finally:
                storage.cleanup_episode_cache(settings, sid, ep_id)
    print(f"\npopulated {done} item(s) into {settings.library_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
