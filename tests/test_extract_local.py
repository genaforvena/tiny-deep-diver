"""Tests for extract_local.py — embedding-based segment selection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_local import select_segments


SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "Welcome everyone to today's lecture on machine learning."},
    {"start": 5.0,  "end": 10.0, "text": "We will cover neural networks and deep learning fundamentals."},
    {"start": 10.0, "end": 15.0, "text": "Um, so yeah, uh, anyway..."},
    {"start": 15.0, "end": 20.0, "text": "The backpropagation algorithm is central to training neural networks."},
    {"start": 20.0, "end": 25.0, "text": "Gradient descent optimizes the loss function iteratively."},
    {"start": 25.0, "end": 30.0, "text": "Thanks, see you next time, bye bye."},
]
TOTAL = 30.0


class TestSelectSegmentsLocal:
    def test_returns_list_of_dicts(self):
        result = select_segments(SEGMENTS, target_duration=15.0)
        assert isinstance(result, list)
        assert all(isinstance(s, dict) for s in result)

    def test_duration_approximately_met(self):
        target = 15.0
        result = select_segments(SEGMENTS, target_duration=target)
        kept = sum(s["end"] - s["start"] for s in result)
        # greedy: may overshoot by one segment (5s) but should not undershoot by more
        assert kept >= target * 0.8

    def test_chronological_order_preserved(self):
        result = select_segments(SEGMENTS, target_duration=20.0)
        starts = [s["start"] for s in result]
        assert starts == sorted(starts)

    def test_subset_of_original(self):
        result = select_segments(SEGMENTS, target_duration=15.0)
        original_texts = {s["text"] for s in SEGMENTS}
        for seg in result:
            assert seg["text"] in original_texts

    def test_empty_input(self):
        assert select_segments([], target_duration=10.0) == []

    def test_full_duration_returns_all_segments(self):
        result = select_segments(SEGMENTS, target_duration=TOTAL)
        assert len(result) == len(SEGMENTS)

    def test_very_short_target_returns_at_least_one(self):
        result = select_segments(SEGMENTS, target_duration=1.0)
        assert len(result) >= 1

    def test_filler_segments_ranked_lower(self):
        # "Um, so yeah" is semantically distant from the topic centroid —
        # it should be excluded when summarising to 50% of content
        result = select_segments(SEGMENTS, target_duration=15.0)
        texts = [s["text"] for s in result]
        assert "Um, so yeah, uh, anyway..." not in texts
