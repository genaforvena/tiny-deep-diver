"""Tests for preprocess.py -- filler pruning, sentence grouping, padding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import prune_fillers, group_sentences, apply_padding, PAD_SECS


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


class TestPruneFillers:
    def test_removes_short_segments(self):
        segs = [_seg(0, 0.5, "hi"), _seg(1, 3, "real content")]
        result = prune_fillers(segs)
        assert len(result) == 1
        assert result[0]["text"] == "real content"

    def test_removes_music_notation(self):
        segs = [_seg(0, 2, "[music]"), _seg(3, 6, "good content")]
        result = prune_fillers(segs)
        assert all(r["text"] != "[music]" for r in result)

    def test_removes_pure_uh_um(self):
        segs = [_seg(0, 2, "uh"), _seg(2, 4, "um"), _seg(4, 8, "the point is")]
        result = prune_fillers(segs)
        texts = [r["text"] for r in result]
        assert "uh" not in texts
        assert "um" not in texts
        assert "the point is" in texts

    def test_removes_subscribe_promo_in_window(self):
        segs = [
            _seg(0, 3, "please subscribe and hit the bell"),
            _seg(10, 15, "today we discuss algorithms"),
        ]
        result = prune_fillers(segs, total_duration=60)
        texts = [r["text"] for r in result]
        assert not any("subscribe" in t for t in texts)
        assert any("algorithms" in t for t in texts)

    def test_keeps_subscribe_mention_in_middle(self):
        segs = [_seg(100, 105, "subscribe to the idea of clean code")]
        result = prune_fillers(segs, total_duration=300)
        assert len(result) == 1

    def test_keeps_normal_content(self):
        segs = [_seg(0, 5, "the key insight is that recursion works here")]
        result = prune_fillers(segs)
        assert result == segs


class TestGroupSentences:
    def test_merges_consecutive_short_chunks(self):
        segs = [
            _seg(0, 1, "The quick"),
            _seg(1, 2, "brown fox"),
            _seg(2, 3, "jumps over the lazy dog."),
        ]
        result = group_sentences(segs)
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["end"] == 3
        assert "quick" in result[0]["text"]
        assert "dog" in result[0]["text"]

    def test_splits_on_large_gap(self):
        segs = [
            _seg(0, 2, "First sentence ends here."),
            _seg(10, 12, "New topic after gap."),
        ]
        result = group_sentences(segs)
        assert len(result) == 2

    def test_single_segment_passthrough(self):
        segs = [_seg(5, 10, "Just one segment.")]
        result = group_sentences(segs)
        assert result == segs

    def test_empty_passthrough(self):
        assert group_sentences([]) == []

    def test_preserves_timestamps(self):
        segs = [_seg(3.5, 5.0, "Part one."), _seg(5.0, 6.5, "Part two.")]
        result = group_sentences(segs)
        assert result[0]["start"] == 3.5
        assert result[0]["end"] == 6.5


class TestApplyPadding:
    def test_adds_padding_both_sides(self):
        segs = [_seg(5.0, 10.0, "content")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["start"] == 5.0 - PAD_SECS
        assert result[0]["end"] == 10.0 + PAD_SECS

    def test_clamps_to_zero(self):
        segs = [_seg(0.05, 5.0, "content")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["start"] == 0.0

    def test_clamps_to_total_duration(self):
        segs = [_seg(55.0, 59.9, "content")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["end"] == 60.0

    def test_preserves_text(self):
        segs = [_seg(5.0, 10.0, "hello")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["text"] == "hello"

    def test_empty_passthrough(self):
        assert apply_padding([], total_duration=60.0) == []

    def test_clamps_overlap_with_neighbor(self):
        # gap of 0.1s < 2*PAD_SECS would overlap if padded independently
        segs = [_seg(0.0, 10.0, "a"), _seg(10.1, 20.0, "b")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["end"] <= result[1]["start"], (
            f"adjacent padded ranges overlap: "
            f"{result[0]['end']} > {result[1]['start']}"
        )

    def test_clamps_at_zero_gap(self):
        # touching segments — padding must not create an overlap
        segs = [_seg(0.0, 10.0, "a"), _seg(10.0, 20.0, "b")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["end"] <= result[1]["start"]

    def test_non_adjacent_padding_unaffected(self):
        # large gap — full padding on both sides should remain
        segs = [_seg(0.0, 10.0, "a"), _seg(30.0, 40.0, "b")]
        result = apply_padding(segs, total_duration=60.0)
        assert result[0]["end"] == 10.0 + PAD_SECS
        assert result[1]["start"] == 30.0 - PAD_SECS
