import gc
from typing import Callable

from polyglot.config import JobSpec, Settings


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
    # Translate each segment independently. An earlier version fed the previous
    # English line as in-band context, but a 7B model intermittently echoed that
    # context into its output (garbled audio + subtitles), so it was removed.
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


def _ollama_generator(settings: Settings):
    import requests

    def gen(messages: list[dict]) -> str:
        r = requests.post(settings.ollama_url, json={
            "model": settings.ollama_model, "stream": False,
            "options": {"temperature": settings.temperature},
            "messages": messages,
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    return gen, (lambda: None)


def translate(segments: list[dict], job: JobSpec, settings: Settings) -> list[dict]:
    system = job.prompt_path.read_text(encoding="utf-8")
    if settings.translate_backend == "mlx":
        gen, release = _mlx_generator(settings)
    elif settings.translate_backend == "ollama":
        gen, release = _ollama_generator(settings)
    else:
        raise ValueError(f"unknown translate_backend: {settings.translate_backend}")
    try:
        return translate_with(segments, system, gen)
    finally:
        release()
