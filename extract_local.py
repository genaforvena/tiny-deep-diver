"""
Local extractive summarization using sentence embeddings (MMR-based).

Algorithm:
  1. Embed every segment with a sentence-transformer model.
  2. Score by Maximal Marginal Relevance (MMR): balances relevance to the
     document centroid against redundancy with already-selected segments.
     lambda_=1.0 is pure centroid similarity; 0.0 is pure diversity.
  3. Greedily pick until target duration is reached.
  4. Re-sort selected segments chronologically.

Model: BAAI/bge-small-en-v1.5 (~130MB, consistently outperforms MiniLM on
retrieval/reranking benchmarks while staying CPU-fast).

No GPU needed. Runs in ~2-3s on CPU for a typical 10-30 min video.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: SentenceTransformer | None = None

_MMR_LAMBDA = 0.7   # 0=max diversity, 1=max relevance; 0.7 works well in practice


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading local model '{_MODEL_NAME}'...")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def select_segments(
    segments: list[dict],
    target_duration: float,
) -> list[dict]:
    """
    Return the subset of segments that best covers the video's content,
    totalling approximately target_duration seconds.
    Uses MMR to balance relevance and diversity.
    """
    if not segments:
        return []

    texts = [s["text"] for s in segments]
    durations = [s["end"] - s["start"] for s in segments]

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    relevance = embeddings @ centroid   # shape (n,)

    # MMR greedy selection
    selected_indices: list[int] = []
    selected_embs: list[np.ndarray] = []
    accumulated = 0.0
    remaining = list(range(len(segments)))

    while remaining and accumulated < target_duration:
        if not selected_embs:
            # first pick: pure relevance
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_matrix = np.stack(selected_embs)   # (k, d)
            def _mmr(i: int) -> float:
                max_sim = float((embeddings[i] @ sel_matrix.T).max())
                return _MMR_LAMBDA * relevance[i] - (1 - _MMR_LAMBDA) * max_sim
            best = max(remaining, key=_mmr)

        selected_indices.append(best)
        selected_embs.append(embeddings[best])
        accumulated += durations[best]
        remaining.remove(best)

    selected_indices.sort()
    return [segments[i] for i in selected_indices]
