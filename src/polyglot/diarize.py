from pathlib import Path

import numpy as np

from polyglot.config import Settings

# Token-free, non-gated speaker embedding model (SpeechBrain ECAPA-TDNN, VoxCeleb).
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path.home() / ".cache" / "polyglot" / "ecapa"


def cluster_embeddings(embeddings: list, max_speakers: int = 4, min_silhouette: float = 0.15) -> list:
    """Cluster speaker embeddings into speakers.

    Tries k = 2..max_speakers (agglomerative, cosine) and keeps the k with the best
    silhouette score. If even the best separation is weak (< min_silhouette) — i.e.
    it's really one speaker — everything collapses to a single speaker.
    """
    n = len(embeddings)
    if n <= 1:
        return [0] * n
    X = np.vstack(embeddings)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    best_labels = [0] * n
    best_score = -1.0
    for k in range(2, min(max_speakers, n - 1) + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_score, best_labels = score, list(labels)
    if best_score < min_silhouette:
        return [0] * n
    return [int(x) for x in best_labels]


def label_segments(segments: list[dict], labels: list[int]) -> list[dict]:
    for seg, lab in zip(segments, labels):
        seg["speaker"] = f"SPEAKER_{lab:02d}"
    return segments


def count_speakers(segments: list[dict]) -> int:
    return len({s["speaker"] for s in segments if s.get("speaker")})


def _slice(audio: np.ndarray, sr: int, start: float, end: float, min_dur: float = 1.5) -> np.ndarray:
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
    labels = cluster_embeddings(embs, min_silhouette=settings.diarize_threshold)
    return label_segments(segments, labels)
