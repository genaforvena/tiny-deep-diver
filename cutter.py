"""
Cut selected segments from a video file and concatenate them into one output.

Uses ffmpeg stream-copy (-c copy) for speed and lossless quality.
Falls back to re-encoding if --reencode is passed or if concat fails.
"""

import subprocess
import tempfile
from pathlib import Path


def cut_and_join(
    video_path: str,
    segments: list[dict],
    output_path: str,
    reencode: bool = False,
) -> None:
    """
    Extract each segment from video_path and concatenate into output_path.
    segments: list of {start: float, end: float, ...} in seconds.
    """
    if not segments:
        raise ValueError("No segments to cut")

    with tempfile.TemporaryDirectory() as tmp:
        part_paths = _extract_parts(video_path, segments, tmp, reencode)
        _concat(part_paths, output_path, reencode)


# ── extraction ────────────────────────────────────────────────────────────────

def _extract_parts(
    video_path: str,
    segments: list[dict],
    tmp: str,
    reencode: bool,
) -> list[Path]:
    parts = []
    for i, seg in enumerate(segments):
        out = Path(tmp) / f"part_{i:04d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-i", video_path,
        ]
        if reencode:
            cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast"]
        else:
            cmd += ["-c", "copy"]
        cmd += ["-avoid_negative_ts", "make_zero", str(out)]
        _run(cmd)
        parts.append(out)
    return parts


# ── concatenation ─────────────────────────────────────────────────────────────

def _concat(parts: list[Path], output_path: str, reencode: bool) -> None:
    concat_list = parts[0].parent / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p}'" for p in parts),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast"]
    else:
        cmd += ["-c", "copy"]
    cmd += [output_path]
    _run(cmd)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error:\n{result.stderr[-2000:]}"
        )
