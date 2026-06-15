"""Supervised end-to-end validation of the watch loop on REAL PTI audio.

Runs into throwaway temp dirs (60s clips) so it proves the mechanics without
polluting ~/PolyglotLibrary. Validates: real download->dub->publish->ledger,
idempotency on a second pass, and retention deleting real files while keeping
the item "seen". Prints wall-clock + peak RSS.
"""
import resource
import sys
import tempfile
import time
from pathlib import Path

from polyglot import library, state, watch
from polyglot.config import load_settings, load_shows


def peak_rss_gb() -> float:
    # macOS ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="polyglot-validate-"))
    s = load_settings()
    s.clip_seconds = 60          # short clips -> fast
    s.retention_keep = 2
    s.cache_dir = tmp / "cache"
    s.output_dir = tmp / "output"
    s.library_path = tmp / "lib"
    s.state_path = tmp / "state" / "processed.json"
    shows = [sh for sh in load_shows() if sh.id == "pti-fr"]
    pod = s.library_path / "Podcasts" / library.safe_name(shows[0].title)
    print(f"workdir: {tmp}\nshows: {[sh.id for sh in shows]}\nlibrary dir: {pod}")

    t0 = time.time()
    r1 = watch.run_watch(s, shows)
    n = r1["published"]
    print(f"\nPASS 1: {r1}  ({time.time()-t0:.0f}s, peak {peak_rss_gb():.1f} GB)")
    assert n >= 1 and r1["failed"] == 0, r1

    mp3s = sorted(pod.glob("*.mp3")); mp4s = sorted(pod.glob("*.mp4")); srts = sorted(pod.glob("*.srt"))
    print(f"library: {len(mp3s)} mp3, {len(mp4s)} mp4, {len(srts)} srt")
    assert len(mp3s) == n and len(mp4s) == n and len(srts) == n, (mp3s, mp4s, srts)
    live = state.published(s.state_path, "pti-fr")
    assert len(live) == n, live
    seen_guids = [i["guid"] for i in live]

    t1 = time.time()
    r2 = watch.run_watch(s, shows)
    print(f"\nPASS 2 (idempotent): {r2}  ({time.time()-t1:.0f}s)")
    assert r2["published"] == 0, r2          # nothing re-dubbed

    # retention: drop keep below n -> the oldest live item(s) are purged (files deleted) but stay seen
    if n >= 2:
        s.retention_keep = n - 1
        r3 = watch.run_watch(s, shows)
        print(f"PASS 3 (retention keep={n-1}): {r3}")
        live_after = state.published(s.state_path, "pti-fr")
        print(f"live after retention: {len(live_after)}")
        assert len(live_after) == n - 1, live_after
        assert len(list(pod.glob('*.mp3'))) == n - 1, "evicted podcast file should be deleted"
        for g in seen_guids:                 # every processed item stays seen -> never re-dubbed
            assert state.is_done(s.state_path, "pti-fr", g) is True, g

    print(f"\nALL CHECKS PASSED  (total {time.time()-t0:.0f}s, peak {peak_rss_gb():.1f} GB)")
    print(f"sample library tree:\n  {pod}")
    for p in sorted(pod.iterdir()):
        print(f"    {p.name}  ({p.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
