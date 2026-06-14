import numpy as np

from polyglot.diarize import cluster_embeddings, label_segments, count_speakers
from polyglot.segments import new_segment


def test_cluster_two_well_separated_groups():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    labels = cluster_embeddings([a, a, b, b], min_silhouette=0.15)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_weak_separation_collapses_to_one():
    # four near-identical vectors; with a high silhouette bar, treat as one speaker
    base = np.array([1.0, 0.0, 0.0])
    embs = [base, base + 0.01, base + 0.02, base + 0.015]
    labels = cluster_embeddings(embs, min_silhouette=0.9)
    assert len(set(labels)) == 1


def test_cluster_single_point():
    assert cluster_embeddings([np.array([1.0, 0.0, 0.0])]) == [0]


def test_label_segments_and_count():
    segs = [new_segment(0, 0, 1, "a"), new_segment(1, 1, 2, "b"), new_segment(2, 2, 3, "c")]
    label_segments(segs, [0, 1, 0])
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[1]["speaker"] == "SPEAKER_01"
    assert segs[2]["speaker"] == "SPEAKER_00"
    assert count_speakers(segs) == 2
