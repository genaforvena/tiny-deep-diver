"""
Select the most important transcript segments using Claude.

Strategy: pass Claude a numbered list of segments and ask it to return
the indices to keep so total duration ≈ target_duration. No fuzzy
text matching required — just index lookups.
"""

import json
import re

import anthropic

MODEL = "claude-opus-4-7"

_SYSTEM = """\
You are an expert video editor specialising in concise, informative cuts.
You will receive a numbered transcript of a video and a target duration.
Your job: select which segments to keep so the total kept duration is as
close as possible to the target, while preserving the core argument,
key facts, and natural conclusions. Prefer coherent runs of segments
over isolated fragments. Discard filler, repetition, and tangents.
Return ONLY a valid JSON array of integer indices, nothing else.
Example: [0, 2, 3, 7, 11]"""


def select_segments(
    segments: list[dict],
    target_duration: float,
) -> list[dict]:
    """
    Return the subset of segments Claude deems most important,
    totalling approximately target_duration seconds.
    """
    if not segments:
        return []

    numbered = "\n".join(
        f"[{i}] ({s['start']:.1f}s–{s['end']:.1f}s, "
        f"{s['end'] - s['start']:.1f}s) {s['text']}"
        for i, s in enumerate(segments)
    )
    total = segments[-1]["end"]
    user_msg = (
        f"Total video duration: {total:.1f}s\n"
        f"Target output duration: {target_duration:.1f}s "
        f"({target_duration / total * 100:.0f}% of original)\n\n"
        f"Transcript segments:\n{numbered}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    indices = _parse_indices(response.content[0].text)
    selected = [segments[i] for i in indices if 0 <= i < len(segments)]
    return selected


def _parse_indices(text: str) -> list[int]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [int(n) for n in re.findall(r"\d+", text)]
