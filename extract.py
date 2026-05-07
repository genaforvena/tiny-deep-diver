"""
Select the most important transcript segments via a user-defined LLM CLI.

The CLI is invoked as a subprocess: prompt is piped to stdin, JSON
response is read from stdout. Default is `gemini` (Google's CLI), but
any tool that reads stdin and writes a response works:

    --llm-cmd "gemini"
    --llm-cmd "claude -p"
    --llm-cmd "ollama run llama3"
    --llm-cmd "llm -m gpt-4"

No API key handled here — the CLI manages its own auth.
"""

import json
import re
import shlex
import subprocess

DEFAULT_LLM_CMD = "gemini"

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
    llm_cmd: str = DEFAULT_LLM_CMD,
) -> list[dict]:
    """
    Return the subset of segments the LLM deems most important,
    totalling approximately target_duration seconds.
    """
    if not segments:
        return []

    numbered = "\n".join(
        f"[{i}] ({s['start']:.1f}s-{s['end']:.1f}s, "
        f"{s['end'] - s['start']:.1f}s) {s['text']}"
        for i, s in enumerate(segments)
    )
    total = segments[-1]["end"]
    prompt = (
        f"{_SYSTEM}\n\n"
        f"Total video duration: {total:.1f}s\n"
        f"Target output duration: {target_duration:.1f}s "
        f"({target_duration / total * 100:.0f}% of original)\n\n"
        f"Transcript segments:\n{numbered}"
    )

    response_text = _run_llm(llm_cmd, prompt)
    indices = _parse_indices(response_text)

    # dedup (preserve first occurrence) + drop out-of-range, then sort
    # chronologically — guarantees no segment is repeated in the output
    seen: set[int] = set()
    deduped = [i for i in indices if 0 <= i < len(segments) and not (i in seen or seen.add(i))]
    deduped.sort()
    return [segments[i] for i in deduped]


def _run_llm(llm_cmd: str, prompt: str) -> str:
    """Run the user's LLM CLI, sending prompt on stdin, returning stdout."""
    argv = shlex.split(llm_cmd)
    if not argv:
        raise ValueError("--llm-cmd cannot be empty")
    try:
        result = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"LLM CLI not found: '{argv[0]}'. Install it or pass --llm-cmd "
            f"with a different command (e.g. --llm-cmd 'claude -p')."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"LLM CLI '{llm_cmd}' failed (exit {result.returncode}):\n"
            f"{result.stderr[-1000:]}"
        )
    return result.stdout


def _parse_indices(text: str) -> list[int]:
    """Extract a list of integer indices from the LLM's response."""
    text = text.strip()
    # try to locate a JSON array anywhere in the output (CLIs sometimes
    # prepend banners like "Loaded cached credentials")
    match = re.search(r"\[[\s\d,]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    # fenced code block
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    # last resort: any standalone integers
    return [int(n) for n in re.findall(r"\b\d+\b", text)]
