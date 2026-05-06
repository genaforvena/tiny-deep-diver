"""
Local extractive summarization using sentence embeddings.

Algorithm:
  1. Embed every segment with a small sentence-transformer model (~80MB).
  2. Compute the document centroid (mean of all embeddings).
  3. Score each segment by cosine similarity to the centroid — segments
     most representative of the whole are ranked highest.
  4. Greedily pick top-scored segments until target duration is reached.
  5. Re-sort selected segments by original order (preserve chronology).

No GPU needed. Runs in ~1-2s on CPU for a typical 10-30 min video.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading local model '{_MODEL_NAME}'…")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def select_segments(
    segments: list[dict],
    target_duration: float,
) -> list[dict]:
    """
    Return the subset of segments that best covers the video's content,
    totalling approximately target_duration seconds.
    """
    if not segments:
        return []

    texts = [s["text"] for s in segments]
    durations = [s["end"] - s["start"] for s in segments]

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    scores = embeddings @ centroid  # cosine similarity (embeddings already normalized)

    # greedy selection: highest score first, stop when duration target is met
    ranked_indices = np.argsort(scores)[::-1].tolist()
    selected_indices = []
    accumulated = 0.0
    for idx in ranked_indices:
        if accumulated >= target_duration:
            break
        selected_indices.append(idx)
        accumulated += durations[idx]

    # restore chronological order
    selected_indices.sort()
    return [segments[i] for i in selected_indices]
