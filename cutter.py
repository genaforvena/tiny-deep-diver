"""
Cut selected segments from a video file and concatenate them into one output.

Default mode: copy video stream (lossless, fast) + re-encode audio as AAC
(fast -- audio is ~2% of file size). This eliminates the A/V sync drift and
boundary clicks that occur when stream-copying audio after a keyframe seek.

--reencode: full libx264 + aac re-encode (slower, use for codec issues).
"""

import subprocess
import tempfile
from pathlib import Path

_AUDIO_BITRATE = "192k"


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
        _concat(part_paths, output_path)


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
        duration = seg["end"] - seg["start"]
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),   # fast input seek to nearest keyframe
            "-i", video_path,
            "-t", str(duration),        # duration from seek point (more reliable than -to)
        ]
        if reencode:
            cmd += ["-c:v", "libx264", "-c:a", "aac", "-b:a", _AUDIO_BITRATE, "-preset", "fast"]
        else:
            # copy video (lossless, instant) + re-encode audio (fixes sync at boundaries)
            cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", _AUDIO_BITRATE]
        cmd += [str(out)]
        _run(cmd)
        parts.append(out)
    return parts


# ── concatenation ─────────────────────────────────────────────────────────────

def _concat(parts: list[Path], output_path: str) -> None:
    concat_list = parts[0].parent / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p}'" for p in parts),
        encoding="utf-8",
    )
    # parts already have clean audio; stream-copy the whole concat
    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        output_path,
    ])


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error:\n{result.stderr[-2000:]}"
        )
