"""Tests for cutter.py — concat list generation and error handling."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cutter import _concat, _extract_parts, cut_and_join


SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "intro"},
    {"start": 10.0, "end": 15.0, "text": "main point"},
    {"start": 25.0, "end": 30.0, "text": "conclusion"},
]


class TestCutAndJoin:
    def test_raises_on_empty_segments(self):
        with pytest.raises(ValueError, match="No segments"):
            cut_and_join("video.mp4", [], "out.mp4")

    def test_calls_ffmpeg_for_each_segment(self, tmp_path):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # create the output file so _concat can find it
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        # one call per segment + one concat call
        assert len(calls) == len(SEGMENTS) + 1

    def test_segment_timestamps_passed_to_ffmpeg(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        extraction_calls = [c for c in captured if "-ss" in c]
        starts = [c[c.index("-ss") + 1] for c in extraction_calls]
        assert "0.0" in starts
        assert "10.0" in starts
        assert "25.0" in starts

    def test_reencode_flag_uses_libx264(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"), reencode=True)

        all_args = [arg for cmd in captured for arg in cmd]
        assert "libx264" in all_args

    def test_stream_copy_used_by_default(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        all_args = [arg for cmd in captured for arg in cmd]
        assert "copy" in all_args
        assert "libx264" not in all_args
