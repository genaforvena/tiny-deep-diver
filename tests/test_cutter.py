"""Tests for cutter.py -- segment extraction and concatenation."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cutter import cut_and_join


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
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        # one extraction call per segment + one concat call
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

    def test_uses_duration_not_absolute_end(self, tmp_path):
        """Extraction should use -t (duration) not -to (absolute), for reliable fast-seek."""
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        extraction_calls = [c for c in captured if "-ss" in c]
        for cmd in extraction_calls:
            assert "-t" in cmd
            assert "-to" not in cmd

    def test_default_copies_video_reencodes_audio(self, tmp_path):
        """Default: stream-copy video, re-encode audio to fix sync at boundaries."""
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        extraction_calls = [c for c in captured if "-ss" in c]
        for cmd in extraction_calls:
            assert "copy" in cmd           # -c:v copy
            assert "aac" in cmd            # -c:a aac
            assert "libx264" not in cmd    # no full re-encode

    def test_reencode_flag_uses_libx264(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"), reencode=True)

        extraction_calls = [c for c in captured if "-ss" in c]
        for cmd in extraction_calls:
            assert "libx264" in cmd
            assert "aac" in cmd

    def test_concat_uses_stream_copy(self, tmp_path):
        """Concat step should always stream-copy -- parts already have clean audio."""
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join("video.mp4", SEGMENTS, str(tmp_path / "out.mp4"))

        concat_call = next(c for c in captured if "concat" in c)
        assert "-c" in concat_call
        assert concat_call[concat_call.index("-c") + 1] == "copy"
