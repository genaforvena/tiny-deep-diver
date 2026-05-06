"""Tests for extract.py — Claude response parsing."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import _parse_indices, select_segments


# ── _parse_indices ────────────────────────────────────────────────────────────

class TestParseIndices:
    def test_clean_json_array(self):
        assert _parse_indices("[0, 2, 5, 11]") == [0, 2, 5, 11]

    def test_json_with_whitespace(self):
        assert _parse_indices("  [1, 3, 7]  ") == [1, 3, 7]

    def test_markdown_code_fence(self):
        assert _parse_indices("```json\n[0, 1, 2]\n```") == [0, 1, 2]

    def test_plain_code_fence(self):
        assert _parse_indices("```\n[4, 5, 6]\n```") == [4, 5, 6]

    def test_fallback_extracts_numbers(self):
        # Claude sometimes adds explanation; numbers should still be extracted
        result = _parse_indices("The key segments are 0, 3, and 7.")
        assert 0 in result
        assert 3 in result
        assert 7 in result

    def test_empty_array(self):
        assert _parse_indices("[]") == []

    def test_single_element(self):
        assert _parse_indices("[42]") == [42]


# ── select_segments (mocked Claude) ──────────────────────────────────────────

SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "Welcome to the show."},
    {"start": 5.0,  "end": 10.0, "text": "Today we discuss AI."},
    {"start": 10.0, "end": 15.0, "text": "First, let me tell you about transformers."},
    {"start": 15.0, "end": 20.0, "text": "Um, so, uh, you know..."},
    {"start": 20.0, "end": 25.0, "text": "The key insight is attention is all you need."},
    {"start": 25.0, "end": 30.0, "text": "Thanks for watching!"},
]


class TestSelectSegments:
    def _make_response(self, indices: list[int]):
        import json
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(indices))]
        return msg

    def test_returns_correct_segments(self):
        with patch("extract.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._make_response([0, 2, 4])
            result = select_segments(SEGMENTS, target_duration=15.0)
        assert len(result) == 3
        assert result[0]["text"] == "Welcome to the show."
        assert result[1]["text"] == "First, let me tell you about transformers."
        assert result[2]["text"] == "The key insight is attention is all you need."

    def test_out_of_range_indices_ignored(self):
        with patch("extract.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._make_response([0, 99, 2])
            result = select_segments(SEGMENTS, target_duration=10.0)
        assert len(result) == 2

    def test_empty_segments_returns_empty(self):
        result = select_segments([], target_duration=10.0)
        assert result == []

    def test_prompt_includes_target_duration(self):
        with patch("extract.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._make_response([0])
            select_segments(SEGMENTS, target_duration=9.0)
            call_kwargs = mock_cls.return_value.messages.create.call_args
            user_content = call_kwargs.kwargs["messages"][0]["content"]
            assert "9" in user_content
