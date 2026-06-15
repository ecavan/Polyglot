import gc
import json
import os
import time
from typing import Callable

from polyglot.config import JobSpec, Settings

# Appended to the fr.txt system prompt for the batched Claude path.
_BATCH_INSTRUCTION = (
    "\n\nTu traduis une LISTE de répliques numérotées. Réponds UNIQUEMENT avec un tableau JSON "
    "de chaînes de caractères : une traduction par réplique, dans le même ordre, exactement le "
    "même nombre d'éléments que de répliques. Aucune numérotation, aucun commentaire, rien "
    "en dehors du tableau JSON."
)


def build_messages(system: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def translate_with(
    segments: list[dict],
    system: str,
    generate: Callable[[list[dict]], str],
) -> list[dict]:
    # Per-segment translation (local fallback path). An earlier version fed the previous
    # English line as in-band context, but a 7B model intermittently echoed that context
    # into its output (garbled audio + subtitles), so the local path stays isolated.
    for seg in segments:
        msgs = build_messages(system, seg["text"])
        seg["translation"] = generate(msgs).strip()
    return segments


def _mlx_generator(settings: Settings):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(settings.mlx_llm_repo)
    sampler = make_sampler(temp=settings.temperature)

    def gen(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            model, tokenizer, prompt=prompt,
            max_tokens=settings.max_tokens, sampler=sampler, verbose=False,
        )

    def release():
        nonlocal model, tokenizer
        del model
        del tokenizer
        gc.collect()

    return gen, release


def _translate_local(segments: list[dict], system: str, settings: Settings) -> list[dict]:
    gen, release = _mlx_generator(settings)
    try:
        return translate_with(segments, system, gen)
    finally:
        release()


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def _parse_json_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of the model's reply (tolerates code fences / prose)."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array in response")
    arr = json.loads(text[start:end + 1])
    if not isinstance(arr, list):
        raise ValueError("response JSON is not an array")
    return [str(x) for x in arr]


def _context_block(prev: list[dict]) -> str:
    """The previous few source+target lines, shown read-only so terminology/pronouns stay
    consistent across chunks."""
    lines = [f"  EN: {s['text']}\n  FR: {s['translation']}"
             for s in prev if s.get("translation")]
    if not lines:
        return ""
    return ("Contexte précédent (NE PAS traduire, continuité seulement):\n"
            + "\n".join(lines) + "\n\n")


def _build_user(chunk: list[dict], ctx: str) -> str:
    numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(chunk))
    return (f"{ctx}Traduis ces {len(chunk)} répliques anglaises en français québécois. "
            f"Retourne un tableau JSON d'exactement {len(chunk)} chaînes, dans l'ordre.\n\n{numbered}")


def _gemini_caller(settings: Settings) -> Callable[[str, str, int], list[str]]:
    import requests

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{settings.gemini_model}:generateContent")

    def call(system: str, user: str, n: int) -> list[str]:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",     # force a clean JSON array back
                "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}},
                "temperature": settings.temperature,
                "maxOutputTokens": max(4096, n * 150),
            },
        }
        last = None
        for attempt in range(settings.translate_max_retries):
            try:
                r = requests.post(url, params={"key": key},
                                  headers={"Content-Type": "application/json"},
                                  json=body, timeout=180)
                r.raise_for_status()
                cand = r.json()["candidates"][0]
                text = "".join(p.get("text", "") for p in cand["content"]["parts"])
                arr = _parse_json_array(text)
                if len(arr) != n:
                    raise ValueError(f"expected {n} translations, got {len(arr)}")
                return arr
            except (requests.RequestException, ValueError, KeyError,
                    IndexError, json.JSONDecodeError) as e:
                last = e
                if attempt < settings.translate_max_retries - 1:
                    time.sleep(2 ** attempt)
        raise last

    return call


def _claude_caller(settings: Settings) -> Callable[[str, str, int], list[str]]:
    import anthropic

    client = anthropic.Anthropic()

    def call(system: str, user: str, n: int) -> list[str]:
        last = None
        for attempt in range(settings.translate_max_retries):
            try:
                resp = client.messages.create(
                    model=settings.anthropic_model, max_tokens=max(1024, n * 120),
                    system=system, messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                arr = _parse_json_array(text)
                if len(arr) != n:
                    raise ValueError(f"expected {n} translations, got {len(arr)}")
                return arr
            except (anthropic.APIError, ValueError, json.JSONDecodeError) as e:
                last = e
                if attempt < settings.translate_max_retries - 1:
                    time.sleep(2 ** attempt)
        raise last

    return call


def remote_translate(segments: list[dict], system: str, settings: Settings,
                     call: Callable[[str, str, int], list[str]], label: str) -> list[dict]:
    """Whole-episode translation in indexed chunks with cross-chunk context, via a remote LLM.
    Any chunk that can't be translated (API down, malformed/misaligned reply) falls back to
    the local model for just those segments, so the unattended loop never wedges."""
    batch_system = system + _BATCH_INSTRUCTION
    failed: list[dict] = []
    for start, chunk in _chunks(segments, settings.translate_chunk_size):
        ctx = _context_block(segments[max(0, start - settings.translate_context_lines):start])
        try:
            for seg, fr in zip(chunk, call(batch_system, _build_user(chunk, ctx), len(chunk))):
                seg["translation"] = fr.strip()
        except Exception as e:
            print(f"  {label} chunk @{start} failed ({e}); translating {len(chunk)} segs locally")
            failed.extend(chunk)
    if failed:
        _translate_local(failed, system, settings)
    return segments


# backend -> (env var that must be present, function building the per-chunk caller)
_REMOTE = {
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), _gemini_caller),
    "claude": (("ANTHROPIC_API_KEY",), _claude_caller),
}


def translate(segments: list[dict], job: JobSpec, settings: Settings) -> list[dict]:
    system = job.prompt_path.read_text(encoding="utf-8")
    backend = settings.translate_backend
    if backend in _REMOTE:
        env_keys, builder = _REMOTE[backend]
        if any(os.environ.get(k) for k in env_keys):
            return remote_translate(segments, system, settings, builder(settings), backend)
        print(f"  no {env_keys[0]} set; using local translation")
        backend = "mlx"
    if backend == "mlx":
        return _translate_local(segments, system, settings)
    raise ValueError(f"unknown translate_backend: {settings.translate_backend}")
