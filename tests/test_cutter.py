"""Tests for cutter.py -- segment extraction and concatenation."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cutter import cut_and_join

VIDEO_A = "video_a.mp4"
VIDEO_B = "video_b.mp4"

SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "intro"},
    {"start": 10.0, "end": 15.0, "text": "main point"},
    {"start": 25.0, "end": 30.0, "text": "conclusion"},
]

CLIPS = [(VIDEO_A, s) for s in SEGMENTS]


class TestCutAndJoin:
    def test_raises_on_empty_clips(self):
        with pytest.raises(ValueError, match="No clips"):
            cut_and_join([], "out.mp4")

    def test_calls_ffmpeg_for_each_clip(self, tmp_path):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"))

        assert len(calls) == len(CLIPS) + 1   # one per clip + concat

    def test_segment_timestamps_passed_to_ffmpeg(self, tmp_path):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"))

        extraction_calls = [c for c in captured if "-ss" in c]
        starts = [c[c.index("-ss") + 1] for c in extraction_calls]
        assert "0.0" in starts
        assert "10.0" in starts
        assert "25.0" in starts

    def test_uses_duration_not_absolute_end(self, tmp_path):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"))

        for cmd in [c for c in captured if "-ss" in c]:
            assert "-t" in cmd
            assert "-to" not in cmd

    def test_default_copies_video_reencodes_audio(self, tmp_path):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"))

        for cmd in [c for c in captured if "-ss" in c]:
            assert "copy" in cmd
            assert "aac" in cmd
            assert "libx264" not in cmd

    def test_reencode_flag_uses_libx264(self, tmp_path):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"), reencode=True)

        for cmd in [c for c in captured if "-ss" in c]:
            assert "libx264" in cmd
            assert "aac" in cmd

    def test_concat_uses_stream_copy(self, tmp_path):
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(CLIPS, str(tmp_path / "out.mp4"))

        concat_call = next(c for c in captured if "concat" in c)
        assert concat_call[concat_call.index("-c") + 1] == "copy"

    def test_mixed_source_videos(self, tmp_path):
        """Clips from different source files should each use their own -i path."""
        mixed_clips = [
            (VIDEO_A, SEGMENTS[0]),
            (VIDEO_B, SEGMENTS[1]),
            (VIDEO_A, SEGMENTS[2]),
        ]
        captured = []

        def fake_run(cmd, **kw):
            captured.append(cmd)
            for arg in cmd:
                if str(arg).endswith(".mp4") and "part_" in str(arg):
                    Path(arg).touch()

        with patch("cutter._run", side_effect=fake_run):
            cut_and_join(mixed_clips, str(tmp_path / "out.mp4"))

        extraction_calls = [c for c in captured if "-ss" in c]
        input_paths = [c[c.index("-i") + 1] for c in extraction_calls]
        assert input_paths[0] == VIDEO_A
        assert input_paths[1] == VIDEO_B
        assert input_paths[2] == VIDEO_A
