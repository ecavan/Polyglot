"""A/B translation comparison on a real episode.

Transcribes a short clip, then translates the same segments three ways so you can SEE the
quality gap and decide whether the paid key is worth it:
  - claude : Anthropic API (needs ANTHROPIC_API_KEY)         [the proposed default]
  - mlx    : local Qwen-7B-4bit                               [the free/offline fallback]
  - argos  : Argos Translate, a free offline Python MT lib    [optional; the "just a library" idea]

Usage:
  uv run python scripts/compare_translation.py [--url URL] [--clip-seconds 90] [--n 12]
  # to include the argos column:  uv run --with argostranslate python scripts/compare_translation.py
"""
import argparse
import textwrap

from polyglot import download, feeds, transcribe
from polyglot import translate as T
from polyglot.config import build_job, load_settings, load_shows


def _copy(segs):
    return [{"text": s["text"], "index": i} for i, s in enumerate(segs)]


def _argos_translate(texts):
    """Best-effort free offline MT. Returns list[str] or None (with a printed reason)."""
    try:
        import argostranslate.package as pkg
        import argostranslate.translate as tr
    except ImportError:
        print("  [argos skipped: not installed — `uv run --with argostranslate python ...`]")
        return None
    try:
        if not any(l.code == "fr" for l in tr.get_installed_languages()):
            pkg.update_package_index()
            p = next(p for p in pkg.get_available_packages()
                     if p.from_code == "en" and p.to_code == "fr")
            pkg.install_from_path(p.download())
        langs = {l.code: l for l in tr.get_installed_languages()}
        en, fr = langs["en"], langs["fr"]
        return [en.get_translation(fr).translate(t) for t in texts]
    except Exception as e:
        print(f"  [argos skipped: {e}]")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="podcast mp3 / youtube url (default: latest PTI)")
    ap.add_argument("--clip-seconds", type=int, default=90)
    ap.add_argument("--n", type=int, default=12, help="how many segments to show")
    args = ap.parse_args()

    settings = load_settings()
    settings.clip_seconds = args.clip_seconds
    job = build_job("pti-fr", settings, load_shows())
    system = job.prompt_path.read_text(encoding="utf-8")

    url = args.url
    if not url:
        url = feeds.list_episodes(job, limit=1)[0].media_url
    print(f"source: {url}\nclip: {args.clip_seconds}s\n")

    work = settings.cache_dir / "_compare"
    src = download.fetch_audio(url, work, settings.clip_seconds)
    wav16 = download.to_16k_mono(src, work)
    segs = transcribe.transcribe(wav16, settings)[: args.n]
    print(f"transcribed {len(segs)} segments\n")

    # claude (only if key present)
    import os
    claude = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        c = _copy(segs)
        T._translate_claude(c, system, settings)
        claude = [s["translation"] for s in c]
    else:
        print("  [claude skipped: no ANTHROPIC_API_KEY]")

    # local mlx
    m = _copy(segs)
    T._translate_local(m, system, settings)
    mlx = [s["translation"] for s in m]

    # argos (free offline lib)
    argos = _argos_translate([s["text"] for s in segs])

    def block(label, text):
        wrapped = textwrap.fill(text or "—", width=78,
                                initial_indent=f"  {label:6}| ", subsequent_indent=" " * 9 + "| ")
        return wrapped

    print("\n" + "=" * 80)
    for i, s in enumerate(segs):
        print(f"[{i}] EN: {s['text']}")
        if claude:
            print(block("claude", claude[i]))
        print(block("mlx", mlx[i]))
        if argos:
            print(block("argos", argos[i]))
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
