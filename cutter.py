"""
Cut clips from one or more source videos and concatenate into one output.

Each clip is a (video_path, segment) pair so clips can come from different
source files (used by the --secondary feature).

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
    clips: list[tuple[str, dict]],
    output_path: str,
    reencode: bool = False,
) -> None:
    """
    clips: list of (video_path, segment) where segment = {start, end, ...}.
    Extracts each clip and concatenates into output_path.
    """
    if not clips:
        raise ValueError("No clips to cut")

    _assert_no_overlap_per_source(clips)

    with tempfile.TemporaryDirectory() as tmp:
        part_paths = _extract_parts(clips, tmp, reencode)
        _concat(part_paths, output_path)


# ── invariants ────────────────────────────────────────────────────────────────

def _assert_no_overlap_per_source(clips: list[tuple[str, dict]]) -> None:
    """
    Per-source subset invariant: for each source video, clips drawn from it
    must be in chronological order with no overlap. The output must never
    contain the same source frame twice.
    """
    last_end: dict[str, float] = {}
    last_index: dict[str, int] = {}
    for i, (path, seg) in enumerate(clips):
        prev_end = last_end.get(path)
        if prev_end is not None and seg["start"] < prev_end:
            raise ValueError(
                f"clips for {path} overlap or are out of order at index {i} "
                f"(prev clip #{last_index[path]} ends at {prev_end}, "
                f"this clip starts at {seg['start']})"
            )
        last_end[path] = seg["end"]
        last_index[path] = i


# ── extraction ────────────────────────────────────────────────────────────────

def _extract_parts(
    clips: list[tuple[str, dict]],
    tmp: str,
    reencode: bool,
) -> list[Path]:
    parts = []
    for i, (video_path, seg) in enumerate(clips):
        out = Path(tmp) / f"part_{i:04d}.mp4"
        duration = seg["end"] - seg["start"]
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),   # fast input seek to nearest keyframe
            "-i", video_path,
            "-t", str(duration),        # duration from seek point (reliable with fast seek)
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
    # parts already have clean audio from extraction; stream-copy the whole concat
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
