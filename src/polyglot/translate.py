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


def _claude_chunk(client, model: str, system: str, chunk: list[dict],
                  ctx: str, max_retries: int) -> list[str]:
    import anthropic

    numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(chunk))
    user = (f"{ctx}Traduis ces {len(chunk)} répliques anglaises en français québécois. "
            f"Retourne un tableau JSON d'exactement {len(chunk)} chaînes, dans l'ordre.\n\n{numbered}")
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max(1024, len(chunk) * 120),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            arr = _parse_json_array(text)
            if len(arr) != len(chunk):
                raise ValueError(f"expected {len(chunk)} translations, got {len(arr)}")
            return arr
        except (anthropic.APIError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)         # backoff on transient API / format errors
    raise last_err


def _translate_claude(segments: list[dict], system: str, settings: Settings) -> list[dict]:
    """Whole-episode translation in indexed chunks with cross-chunk context. Any chunk that
    can't be translated (API down, malformed/misaligned reply) falls back to the local model
    for just those segments, so the unattended loop never wedges."""
    import anthropic

    client = anthropic.Anthropic()
    batch_system = system + _BATCH_INSTRUCTION
    failed: list[dict] = []
    for start, chunk in _chunks(segments, settings.translate_chunk_size):
        ctx = _context_block(segments[max(0, start - settings.translate_context_lines):start])
        try:
            for seg, fr in zip(chunk, _claude_chunk(client, settings.anthropic_model,
                                                    batch_system, chunk, ctx,
                                                    settings.translate_max_retries)):
                seg["translation"] = fr.strip()
        except Exception as e:
            print(f"  claude chunk @{start} failed ({e}); translating {len(chunk)} segs locally")
            failed.extend(chunk)
    if failed:
        _translate_local(failed, system, settings)
    return segments


def translate(segments: list[dict], job: JobSpec, settings: Settings) -> list[dict]:
    system = job.prompt_path.read_text(encoding="utf-8")
    backend = settings.translate_backend
    if backend == "claude":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return _translate_claude(segments, system, settings)
        print("  no ANTHROPIC_API_KEY set; using local translation")
        backend = "mlx"
    if backend == "mlx":
        return _translate_local(segments, system, settings)
    raise ValueError(f"unknown translate_backend: {settings.translate_backend}")
