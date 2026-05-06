"""Tests for the multi-pass _iterative_select logic in summarizer.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizer import _iterative_select, _CONVERGENCE_TOL


def _make_segments(n: int, seg_duration: float = 5.0) -> list[dict]:
    return [
        {"start": i * seg_duration, "end": (i + 1) * seg_duration, "text": f"segment {i}"}
        for i in range(n)
    ]


def _trim_to(target: float):
    """A mock select_fn that keeps segments up to exactly target seconds."""
    def _fn(segments, t):
        out, acc = [], 0.0
        for s in segments:
            dur = s["end"] - s["start"]
            if acc + dur > t * (1 + _CONVERGENCE_TOL * 2):
                break
            out.append(s)
            acc += dur
        return out or segments[:1]
    return _fn


class TestIterativeSelect:
    def test_single_pass_converges(self):
        segs = _make_segments(20)          # 100s total
        target = 50.0                       # keep half
        result = _iterative_select(_trim_to(target), segs, target, max_passes=3, method="test")
        kept = sum(s["end"] - s["start"] for s in result)
        assert abs(kept - target) / target <= _CONVERGENCE_TOL * 2

    def test_stops_when_pool_already_within_tolerance(self):
        segs = _make_segments(4)           # 20s total
        target = 18.0                       # pool (20s) is within 12% of 18s
        calls = []
        def counting_fn(segments, t):
            calls.append(len(segments))
            return segments[:3]
        _iterative_select(counting_fn, segs, target, max_passes=5, method="test")
        # pool is already within tolerance so select_fn should not be called
        assert len(calls) == 0

    def test_respects_max_passes(self):
        segs = _make_segments(100)         # 500s total
        target = 10.0                       # very aggressive — won't converge in 2 passes
        calls = []
        def counting_fn(segments, t):
            calls.append(len(segments))
            # return 80% of input each time (never converges)
            return segments[:max(1, int(len(segments) * 0.8))]
        _iterative_select(counting_fn, segs, target, max_passes=2, method="test")
        assert len(calls) == 2

    def test_auto_mode_stops_on_convergence(self):
        segs = _make_segments(20)          # 100s total
        target = 50.0
        calls = []
        def converging_fn(segments, t):
            calls.append(len(segments))
            return [s for s in segments if (s["end"] - s["start"]) and len(calls) >= 1][:10]
        _iterative_select(converging_fn, segs, target, max_passes=0, method="test")
        # should stop early, not run all 6 auto passes
        assert len(calls) < 6

    def test_skips_selection_when_pool_already_below_target(self):
        segs = _make_segments(10)          # 50s total
        target = 100.0                      # target larger than pool — nothing to cut
        calls = []
        def fn(segments, t):
            calls.append(True)
            return segments
        result = _iterative_select(fn, segs, target, max_passes=5, method="test")
        # pool (50s) < target (100s), so early-exit triggers: select_fn never called
        assert len(calls) == 0
        assert result == segs              # full pool returned unchanged

    def test_stops_if_select_undershoots(self):
        segs = _make_segments(20)          # 100s total
        target = 80.0
        call_count = [0]
        def fn(segments, t):
            call_count[0] += 1
            # first call: return 70s (undershoot target), can't go lower
            return segments[:14]
        _iterative_select(fn, segs, target, max_passes=5, method="test")
        # undershoots on first call, so second pass should not run
        assert call_count[0] == 1

    def test_empty_segments_returns_empty(self):
        result = _iterative_select(lambda s, t: [], [], 10.0, max_passes=3, method="test")
        assert result == []

    def test_output_is_subset_of_input(self):
        segs = _make_segments(20)
        target = 30.0
        result = _iterative_select(_trim_to(target), segs, target, max_passes=3, method="test")
        original_texts = {s["text"] for s in segs}
        for seg in result:
            assert seg["text"] in original_texts
