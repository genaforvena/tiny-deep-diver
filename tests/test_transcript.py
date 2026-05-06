"""Tests for transcript.py — parsing, merging, and helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcript import _extract_video_id, _merge_short, _parse_vtt, _ts_to_sec, total_duration


# ── _extract_video_id ─────────────────────────────────────────────────────────

class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=ViS4nf9j9b8&t=10s") == "ViS4nf9j9b8"

    def test_invalid_url_raises(self):
        import pytest
        with pytest.raises(ValueError):
            _extract_video_id("https://vimeo.com/123456")


# ── _ts_to_sec ────────────────────────────────────────────────────────────────

class TestTsToSec:
    def test_zero(self):
        assert _ts_to_sec("00:00:00.000") == 0.0

    def test_seconds_only(self):
        assert _ts_to_sec("00:00:05.500") == 5.5

    def test_minutes(self):
        assert _ts_to_sec("00:01:30.000") == 90.0

    def test_hours(self):
        assert _ts_to_sec("01:00:00.000") == 3600.0

    def test_comma_separator(self):
        # SRT uses comma, VTT uses dot — both should work
        assert _ts_to_sec("00:00:10,250") == 10.25

    def test_mixed(self):
        assert _ts_to_sec("01:23:45.678") == pytest.approx(5025.678, rel=1e-4)


import pytest  # noqa: E402


# ── _merge_short ──────────────────────────────────────────────────────────────

class TestMergeShort:
    def _seg(self, start, end, text):
        return {"start": start, "end": end, "text": text}

    def test_no_merge_when_long_enough(self):
        segs = [self._seg(0, 3, "hello"), self._seg(3, 6, "world")]
        result = _merge_short(segs, min_duration=1.0)
        assert len(result) == 2

    def test_merges_short_into_next(self):
        segs = [
            self._seg(0, 0.5, "hi"),    # short
            self._seg(0.5, 4, "there"),  # immediately follows
        ]
        result = _merge_short(segs, min_duration=1.0)
        assert len(result) == 1
        assert result[0]["text"] == "hi there"
        assert result[0]["end"] == 4

    def test_no_merge_when_gap_too_large(self):
        segs = [
            self._seg(0, 0.5, "hi"),
            self._seg(2.0, 4, "there"),  # 1.5s gap — should not merge
        ]
        result = _merge_short(segs, min_duration=1.0)
        assert len(result) == 2

    def test_empty_input(self):
        assert _merge_short([]) == []

    def test_single_segment(self):
        segs = [self._seg(0, 2, "only one")]
        assert _merge_short(segs) == segs


# ── _parse_vtt ────────────────────────────────────────────────────────────────

SAMPLE_VTT = """\
WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world, this is a test.

00:00:04.500 --> 00:00:08.000
This is the second caption line.

00:00:08.100 --> 00:00:08.100
This is the second caption line.

00:00:09.000 --> 00:00:12.000
And here is a third segment.
"""


class TestParseVtt:
    def test_parses_three_distinct_segments(self, tmp_path):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(SAMPLE_VTT, encoding="utf-8")
        segs = _parse_vtt(vtt_file)
        # duplicate line is merged, so we get 3 not 4
        texts = [s["text"] for s in segs]
        assert "Hello world, this is a test." in texts
        assert "And here is a third segment." in texts

    def test_timestamps_parsed_correctly(self, tmp_path):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(SAMPLE_VTT, encoding="utf-8")
        segs = _parse_vtt(vtt_file)
        first = segs[0]
        assert first["start"] == pytest.approx(1.0)
        assert first["end"] == pytest.approx(4.0)

    def test_strips_html_tags(self, tmp_path):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c>Hello</c> <b>world</b>\n"
        vtt_file = tmp_path / "tags.vtt"
        vtt_file.write_text(vtt, encoding="utf-8")
        segs = _parse_vtt(vtt_file)
        assert segs[0]["text"] == "Hello world"

    def test_empty_vtt(self, tmp_path):
        vtt_file = tmp_path / "empty.vtt"
        vtt_file.write_text("WEBVTT\n\n", encoding="utf-8")
        assert _parse_vtt(vtt_file) == []


# ── total_duration ────────────────────────────────────────────────────────────

class TestTotalDuration:
    def test_empty(self):
        assert total_duration([]) == 0.0

    def test_returns_last_end(self):
        segs = [
            {"start": 0, "end": 5, "text": "a"},
            {"start": 5, "end": 12, "text": "b"},
        ]
        assert total_duration(segs) == 12.0
