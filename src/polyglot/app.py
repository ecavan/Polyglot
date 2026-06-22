"""Polyglot control panel — `uv run polyglot app`.

Queue videos (paste a URL) and podcasts (latest episode), run the queue overnight (one dub at
a time), and browse/delete the library to free space. The actual dubbing runs in a detached
worker process, so this tab can be closed while it works.
"""
import time
from pathlib import Path

import streamlit as st

from polyglot import download, jobs
from polyglot.config import load_settings, load_shows

st.set_page_config(page_title="Polyglot", page_icon="🎬", layout="wide")
settings = load_settings()
shows = load_shows()

st.title("🎬 Polyglot — dub control")

left, right = st.columns(2)

with left:
    st.subheader("➕ Add a video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    c_sp, c_dom = st.columns(2)
    speakers = c_sp.number_input("Speakers", min_value=1, max_value=6, value=1, step=1,
                                 help="1 = solo narrator (chess); 2–3 for multi-speaker (poker)")
    domain = c_dom.selectbox("Topic",
                             ["general", "chess", "poker", "news", "finance", "sports", "history"],
                             help="tunes the transcription/translation vocabulary")
    if st.button("Queue video", type="primary", disabled=not url.strip()):
        try:
            with st.spinner("Reading video info…"):
                m = download.video_metadata(url.strip(), settings.ytdlp_cookies_browser,
                                            settings.ytdlp_cookies_file)
            jobs.add_video(settings, url.strip(), title=m["title"], channel=m["channel"],
                           video_id=m["video_id"], duration=m["duration"],
                           published_ts=m.get("published_ts"), speakers=int(speakers), domain=domain)
            mins = m["duration"] / 60
            st.success(f"Queued: {m['title']} — {m['channel']} ({mins:.0f} min)")
            if mins > settings.max_video_minutes:
                st.warning(f"That's {mins:.0f} min, over the {settings.max_video_minutes}-min "
                           "cap — the worker will skip it. Raise max_video_minutes to allow it.")
        except Exception as e:
            st.error(f"Couldn't read that URL: {e}")

    st.subheader("➕ Add a podcast (latest episode)")
    pod = [s for s in shows if s.source_type == "rss"]
    if pod:
        names = [s.title for s in pod]
        choice = st.selectbox("Show", names)
        if st.button("Queue latest episode"):
            show = pod[names.index(choice)]
            jobs.add_podcast(settings, show.id, title=f"{show.title} — latest")
            st.success(f"Queued latest: {show.title}")
    else:
        st.caption("No podcast shows configured (config/shows.toml).")

with right:
    st.subheader("⏳ Queue")
    all_jobs = jobs.list_jobs(settings)
    active = [j for j in all_jobs if j["status"] in ("pending", "running")]
    running = jobs.worker_running(settings)
    st.caption(("🟢 worker running" if running else "⚪ worker idle") + f"  ·  {len(active)} queued")

    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Run queue", disabled=(not active or running)):
        jobs.start_worker(settings)
        st.toast("Worker started — dubbing one at a time. Safe to close this tab.")
        time.sleep(1)
        st.rerun()
    if c2.button("↻ Refresh"):
        st.rerun()
    if c3.button("🧹 Clear finished"):
        jobs.clear_finished(settings)
        st.rerun()

    for j in active:
        icon = "▶️" if j["status"] == "running" else "•"
        st.write(f"{icon} **{j['status']}** · {j['type']} · {j.get('title', '')[:60]}")

    done = [j for j in all_jobs if j["status"] in ("done", "failed")]
    if done:
        with st.expander(f"Finished ({len(done)})"):
            for j in reversed(done[-25:]):
                mark = "✅" if j["status"] == "done" else "❌"
                err = f" · {j['error']}" if j.get("error") else ""
                st.write(f"{mark} {j.get('title', '')[:70]}{err}")

st.divider()
st.subheader("📚 Library")
items = jobs.library_items(settings)
total_gb = sum(i["size_mb"] for i in items) / 1000
st.caption(f"{len(items)} items · {total_gb:.2f} GB on disk")
for idx, it in enumerate(sorted(items, key=lambda x: x.get("published_at", 0), reverse=True)):
    a, b, c = st.columns([6, 1, 1])
    a.write(f"**{it.get('title', '')[:80]}**  ·  _{it.get('kind', '')}_")
    b.write(f"{it['size_mb']:.0f} MB")
    if c.button("🗑 Delete", key=f"del_{idx}"):       # index key: always unique, no collisions
        jobs.delete_item(settings, it["show_id"], it["guid"])
        st.toast("Deleted — space freed; it can be re-pulled.")
        st.rerun()

log = settings.state_path.parent / "worker.log"
if log.exists():
    with st.expander("🪵 Worker log"):
        tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
        st.code("\n".join(tail) or "(empty)")
