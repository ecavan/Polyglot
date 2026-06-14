import re
from typing import Callable

import numpy as np

from polyglot.config import Settings

SNAC_REPO = "hubertsiuzdak/snac_24khz"
SR = 24000
_CUSTOM_RE = re.compile(r"<custom_token_(\d+)>")


def tokens_to_codes(text: str) -> list[int]:
    """Parse the Orpheus '<custom_token_N>' stream into SNAC code ids.

    Mirrors Orpheus-FastAPI's turn_token_into_id: each token maps to
    int(N) - 10 - ((index % 7) * 4096); invalid (<=0) tokens are dropped and the
    7-cycle index advances ONLY on valid tokens (leading marker tokens must be
    skipped or every frame misaligns -> garbled audio). Truncated to whole frames.
    """
    buffer: list[int] = []
    count = 0
    for m in _CUSTOM_RE.finditer(text):
        tid = int(m.group(1)) - 10 - ((count % 7) * 4096)
        if tid > 0:
            buffer.append(tid)
            count += 1
    n = (len(buffer) // 7) * 7
    return buffer[:n]


def redistribute(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """7-token frames -> SNAC's 3 hierarchical code layers."""
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        b = codes[7 * i:7 * i + 7]
        l1.append(b[0])
        l2 += [b[1], b[4]]
        l3 += [b[2], b[3], b[5], b[6]]
    return l1, l2, l3


def build_synth(settings: Settings) -> Callable[[str, str], np.ndarray]:
    """Load Orpheus (llama.cpp GGUF, Metal) + SNAC once; return synth(text, voice)->24k wav."""
    import torch
    from llama_cpp import Llama
    from snac import SNAC

    llm = Llama(model_path=str(settings.orpheus_gguf), n_gpu_layers=-1, n_ctx=4096, verbose=False)
    # SNAC decode runs on CPU: torch-MPS conflicts with MLX (whisper/Qwen) in the same
    # process ("Unknown device for graph fuser"). SNAC is tiny, so CPU is plenty fast.
    device = "cpu"
    snac = SNAC.from_pretrained(SNAC_REPO).eval().to(device)

    def synth(text: str, voice: str) -> np.ndarray:
        out = llm(
            f"<|audio|>{voice}: {text}<|eot_id|>",
            max_tokens=settings.orpheus_max_tokens,
            temperature=settings.orpheus_temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        codes = tokens_to_codes(out["choices"][0]["text"])
        if not codes:
            return np.zeros(0, dtype=np.float32)
        l1, l2, l3 = redistribute(codes)
        # Clamp to SNAC's codebook range [0, 4095]; higher temperatures occasionally emit
        # out-of-range codes which would crash the embedding lookup.
        ct = [
            torch.tensor([l1], device=device).clamp_(0, 4095),
            torch.tensor([l2], device=device).clamp_(0, 4095),
            torch.tensor([l3], device=device).clamp_(0, 4095),
        ]
        with torch.inference_mode():
            audio = snac.decode(ct).squeeze().detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    return synth
