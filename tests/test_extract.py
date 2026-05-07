"""Tests for extract.py — LLM CLI invocation and response parsing."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import _parse_indices, select_segments, _run_llm


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

    def test_extracts_array_from_preamble(self):
        # CLIs (esp. gemini) often print banners before the JSON
        text = "Loaded cached credentials.\n[0, 3, 7]\n"
        assert _parse_indices(text) == [0, 3, 7]

    def test_fallback_extracts_numbers(self):
        result = _parse_indices("The key segments are 0, 3, and 7.")
        assert 0 in result
        assert 3 in result
        assert 7 in result

    def test_empty_array(self):
        assert _parse_indices("[]") == []

    def test_single_element(self):
        assert _parse_indices("[42]") == [42]


# ── _run_llm ──────────────────────────────────────────────────────────────────

class TestRunLlm:
    def test_pipes_prompt_to_stdin(self):
        with patch("extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="[0]", stderr="")
            _run_llm("gemini", "hello prompt")
            kwargs = mock_run.call_args.kwargs
            assert kwargs["input"] == "hello prompt"

    def test_splits_command_with_args(self):
        with patch("extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="[0]", stderr="")
            _run_llm("claude -p --json", "x")
            argv = mock_run.call_args.args[0]
            assert argv == ["claude", "-p", "--json"]

    def test_raises_on_nonzero_exit(self):
        with patch("extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="boom")
            with pytest.raises(RuntimeError, match="failed"):
                _run_llm("gemini", "x")

    def test_raises_helpful_error_when_cli_missing(self):
        with patch("extract.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="not found"):
                _run_llm("nonexistent-cli", "x")

    def test_empty_command_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _run_llm("", "x")


# ── select_segments (mocked CLI) ──────────────────────────────────────────────

SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "Welcome to the show."},
    {"start": 5.0,  "end": 10.0, "text": "Today we discuss AI."},
    {"start": 10.0, "end": 15.0, "text": "First, let me tell you about transformers."},
    {"start": 15.0, "end": 20.0, "text": "Um, so, uh, you know..."},
    {"start": 20.0, "end": 25.0, "text": "The key insight is attention is all you need."},
    {"start": 25.0, "end": 30.0, "text": "Thanks for watching!"},
]


def _mock_cli(stdout: str):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


class TestSelectSegments:
    def test_returns_correct_segments(self):
        with patch("extract.subprocess.run", return_value=_mock_cli("[0, 2, 4]")):
            result = select_segments(SEGMENTS, target_duration=15.0)
        assert len(result) == 3
        assert result[0]["text"] == "Welcome to the show."
        assert result[1]["text"] == "First, let me tell you about transformers."
        assert result[2]["text"] == "The key insight is attention is all you need."

    def test_out_of_range_indices_ignored(self):
        with patch("extract.subprocess.run", return_value=_mock_cli("[0, 99, 2]")):
            result = select_segments(SEGMENTS, target_duration=10.0)
        assert len(result) == 2

    def test_empty_segments_returns_empty(self):
        result = select_segments([], target_duration=10.0)
        assert result == []

    def test_prompt_includes_target_duration(self):
        with patch("extract.subprocess.run", return_value=_mock_cli("[0]")) as mock_run:
            select_segments(SEGMENTS, target_duration=9.0)
            prompt = mock_run.call_args.kwargs["input"]
            assert "9" in prompt
            assert "Target output duration" in prompt

    def test_uses_custom_llm_cmd(self):
        with patch("extract.subprocess.run", return_value=_mock_cli("[0]")) as mock_run:
            select_segments(SEGMENTS, target_duration=5.0, llm_cmd="claude -p")
            argv = mock_run.call_args.args[0]
            assert argv == ["claude", "-p"]

    def test_default_command_is_gemini(self):
        with patch("extract.subprocess.run", return_value=_mock_cli("[0]")) as mock_run:
            select_segments(SEGMENTS, target_duration=5.0)
            argv = mock_run.call_args.args[0]
            assert argv == ["gemini"]

    def test_dedups_and_sorts_indices(self):
        # LLM returns duplicates and out-of-order indices — output must be
        # a chronologically ordered subset with each segment appearing once
        with patch("extract.subprocess.run", return_value=_mock_cli("[2, 0, 0, 1, 2]")):
            result = select_segments(SEGMENTS, target_duration=15.0)
        texts = [r["text"] for r in result]
        assert texts == [
            SEGMENTS[0]["text"],
            SEGMENTS[1]["text"],
            SEGMENTS[2]["text"],
        ]
