"""
Match selected primary segments to their closest equivalents in a secondary video.

Algorithm:
  1. Embed all segments from both videos using the shared sentence-transformer.
  2. For each primary segment, find the highest-cosine-similarity secondary segment,
     subject to a monotonic ordering constraint (next secondary match must come
     after the previous one -- preserves narrative order).
  3. If the best score is below `threshold`, keep the primary segment as-is.

Returns a list parallel to `primary_segments`, where each item is either:
  - a secondary segment dict (tagged _source='secondary') if a good match was found
  - the original primary segment dict (tagged _source='primary') as fallback
"""

from __future__ import annotations

import numpy as np
from extract_local import _get_model

_DEFAULT_THRESHOLD = 0.45   # cosine similarity floor; tune down for looser matching


def find_matches(
    primary_segments: list[dict],
    secondary_segments: list[dict],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    For each primary segment return the best-matching secondary segment (or the
    primary itself if no secondary segment clears the similarity threshold).
    """
    if not primary_segments:
        return []
    if not secondary_segments:
        return [dict(s, _source="primary") for s in primary_segments]

    model = _get_model()

    p_embs = model.encode(
        [s["text"] for s in primary_segments],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    s_embs = model.encode(
        [s["text"] for s in secondary_segments],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # (n_primary x n_secondary) cosine similarity matrix
    scores = p_embs @ s_embs.T

    results: list[dict] = []
    min_sec_idx = 0   # monotonic lower bound -- never go backwards in secondary

    for i, primary_seg in enumerate(primary_segments):
        if min_sec_idx >= len(secondary_segments):
            results.append(dict(primary_seg, _source="primary"))
            continue

        # only look at secondary segments at or after min_sec_idx
        row = scores[i, min_sec_idx:]
        best_local = int(np.argmax(row))
        best_score = float(row[best_local])
        best_sec_idx = min_sec_idx + best_local

        if best_score >= threshold:
            results.append(dict(secondary_segments[best_sec_idx], _source="secondary"))
            min_sec_idx = best_sec_idx + 1
        else:
            results.append(dict(primary_seg, _source="primary"))

    return results


def match_summary(results: list[dict]) -> str:
    n_sec = sum(1 for r in results if r.get("_source") == "secondary")
    n_pri = len(results) - n_sec
    return f"{n_sec} segments from secondary, {n_pri} kept from primary"
