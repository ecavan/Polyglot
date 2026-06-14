from pathlib import Path

import numpy as np

from polyglot.config import Settings

# Token-free, non-gated speaker embedding model (SpeechBrain ECAPA-TDNN, VoxCeleb).
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path.home() / ".cache" / "polyglot" / "ecapa"


def cluster_embeddings(embeddings: list, num_speakers: int = 2, threshold: float = 0.5) -> list:
    """Cluster speaker embeddings into speakers (agglomerative, cosine, average linkage).

    - num_speakers >= 2: force exactly that many clusters (capped at n). Reliable for
      shows with a known host count; per-segment embeddings are too noisy for stable
      auto-detection.
    - num_speakers == 1: single speaker (no diarization).
    - num_speakers == 0: auto — split by `threshold` cosine distance (lower => more).
    """
    n = len(embeddings)
    if n <= 1 or num_speakers == 1:
        return [0] * n
    X = np.vstack(embeddings)
    from sklearn.cluster import AgglomerativeClustering

    if num_speakers and num_speakers >= 2:
        labels = AgglomerativeClustering(
            n_clusters=min(num_speakers, n), metric="cosine", linkage="average"
        ).fit_predict(X)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
        ).fit_predict(X)
    return [int(x) for x in labels]


def label_segments(segments: list[dict], labels: list[int]) -> list[dict]:
    for seg, lab in zip(segments, labels):
        seg["speaker"] = f"SPEAKER_{lab:02d}"
    return segments


def count_speakers(segments: list[dict]) -> int:
    return len({s["speaker"] for s in segments if s.get("speaker")})


def _slice(audio: np.ndarray, sr: int, start: float, end: float, min_dur: float = 3.0) -> np.ndarray:
    """Audio slice for a segment, widened symmetrically to min_dur for stable embeddings."""
    s, e = int(start * sr), int(end * sr)
    if (end - start) < min_dur:
        pad = int((min_dur - (end - start)) * sr / 2)
        s, e = max(0, s - pad), min(len(audio), e + pad)
    return audio[s:e]


def _embed_segments(wav_path: Path, segments: list[dict], settings: Settings) -> list[np.ndarray]:
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    clf = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE,
        savedir=str(MODEL_DIR),
        run_opts={"device": settings.tts_device},
    )
    embs: list[np.ndarray] = []
    for seg in segments:
        sl = np.ascontiguousarray(_slice(audio, sr, seg["start"], seg["end"]))
        t = torch.from_numpy(sl).float().unsqueeze(0)
        emb = clf.encode_batch(t).squeeze().detach().cpu().numpy()
        embs.append(emb)
    return embs


def diarize(wav_path: Path, segments: list[dict], settings: Settings) -> list[dict]:
    if not segments:
        return segments
    embs = _embed_segments(wav_path, segments, settings)
    labels = cluster_embeddings(
        embs, num_speakers=settings.num_speakers, threshold=settings.diarize_threshold
    )
    return label_segments(segments, labels)
