"""
tiny-deep-diver -- video-to-video summarizer

Usage:
    python summarizer.py <url> --ratio 0.3
    python summarizer.py <url> --duration 180 --local
    python summarizer.py <url> --ratio 1.0 --secondary <url2>
        Select all segments from url, replace matching ones with clips from url2.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from cutter import cut_and_join
from transcript import get_transcript, total_duration

_CONVERGENCE_TOL = 0.12
_MAX_AUTO_PASSES = 6


def main() -> None:
    args = _parse_args()

    print("-> Fetching primary transcript...")
    primary_segments = get_transcript(args.url)
    if not primary_segments:
        sys.exit("Error: could not retrieve transcript for primary video.")

    orig_duration = total_duration(primary_segments)
    target = _resolve_target(args, orig_duration)
    print(
        f"  {len(primary_segments)} segments, {orig_duration:.0f}s original "
        f"-> {target:.0f}s target ({target / orig_duration * 100:.0f}%)"
    )

    if args.local:
        from extract_local import select_segments
        method = "local embeddings"
    else:
        from extract import select_segments
        method = "Claude"

    selected = _iterative_select(
        select_segments, primary_segments, target,
        max_passes=args.passes, method=method,
    )
    if not selected:
        sys.exit("Error: no segments were selected.")

    kept = sum(s["end"] - s["start"] for s in selected)

    # ── secondary video substitution ──────────────────────────────────────────
    secondary_url = args.secondary
    if secondary_url:
        print("-> Fetching secondary transcript...")
        secondary_segments = get_transcript(secondary_url)
        print(
            f"  {len(secondary_segments)} segments, "
            f"{total_duration(secondary_segments):.0f}s"
        )
        from match import find_matches, match_summary
        print("-> Matching segments to secondary video...")
        matched = find_matches(selected, secondary_segments)
        print(f"  {match_summary(matched)}")
    else:
        matched = [dict(s, _source="primary") for s in selected]

    with tempfile.TemporaryDirectory() as tmp:
        primary_path = _download_video(args.url, tmp, label="primary")
        if secondary_url:
            secondary_path = _download_video(secondary_url, tmp, label="secondary")
        else:
            secondary_path = primary_path

        clips = [
            (secondary_path if seg["_source"] == "secondary" else primary_path, seg)
            for seg in matched
        ]

        print("-> Cutting and joining...")
        cut_and_join(clips, args.output, reencode=args.reencode)

    print(f"\nDone: {args.output}")
    print(
        f"  {orig_duration:.0f}s -> {kept:.0f}s "
        f"({kept / orig_duration * 100:.0f}% of original)"
    )


# ── multi-pass compression ────────────────────────────────────────────────────

def _iterative_select(
    select_fn,
    segments: list[dict],
    target: float,
    max_passes: int,
    method: str,
) -> list[dict]:
    """
    Repeatedly apply select_fn, feeding each pass's output as the next
    pass's input, until the result is within _CONVERGENCE_TOL of target
    or max_passes is reached.
    """
    pool = segments
    selected = segments
    auto = max_passes == 0

    limit = _MAX_AUTO_PASSES if auto else max_passes

    for pass_num in range(1, limit + 1):
        kept_before = sum(s["end"] - s["start"] for s in pool)
        if kept_before <= target * (1 + _CONVERGENCE_TOL):
            selected = pool
            break

        print(f"-> Pass {pass_num} ({method}, pool={len(pool)} segs, {kept_before:.0f}s)...")
        selected = select_fn(pool, target)
        kept = sum(s["end"] - s["start"] for s in selected)
        print(f"   {len(selected)} segments kept, {kept:.0f}s")

        if not selected:
            break

        within_tol = abs(kept - target) / target <= _CONVERGENCE_TOL
        if within_tol:
            print(f"   Converged (within {_CONVERGENCE_TOL*100:.0f}% of target).")
            break

        if kept <= target:
            break

        pool = selected

    return selected


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize a YouTube video into a shorter video using key segments."
    )
    p.add_argument("url", help="Primary YouTube video URL")
    p.add_argument(
        "--secondary", default=None, metavar="URL",
        help="Optional second YouTube URL; matching segments replace primary clips"
    )
    p.add_argument(
        "--ratio", type=float, default=None,
        help="Keep this fraction of original duration (0.0-1.0). Use 1.0 for no compression."
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Target output duration in seconds"
    )
    p.add_argument(
        "--output", default="summary.mp4",
        help="Output file path (default: summary.mp4)"
    )
    p.add_argument(
        "--passes", type=int, default=0,
        help="Number of compression passes (default: 0 = auto until converged)"
    )
    p.add_argument(
        "--reencode", action="store_true",
        help="Re-encode output (slower but fixes A/V sync issues on some videos)"
    )
    p.add_argument(
        "--local", action="store_true",
        help="Use local sentence embeddings instead of Claude (no API key needed)"
    )
    p.add_argument(
        "--match-threshold", type=float, default=0.45, dest="match_threshold",
        help="Cosine similarity threshold for secondary matching (default: 0.45)"
    )
    args = p.parse_args()
    if args.ratio is None and args.duration is None:
        p.error("One of --ratio or --duration is required.")
    if args.ratio is not None and not (0.0 < args.ratio <= 1.0):
        p.error("--ratio must be between 0.0 and 1.0.")
    if args.passes < 0:
        p.error("--passes must be >= 0.")
    return args


def _resolve_target(args: argparse.Namespace, orig_duration: float) -> float:
    if args.duration is not None:
        return float(args.duration)
    return args.ratio * orig_duration


def _download_video(url: str, tmp: str, label: str = "") -> str:
    tag = f" ({label})" if label else ""
    print(f"-> Downloading video{tag}...")
    out_template = str(Path(tmp) / f"video_{label}.%(ext)s")
    result = subprocess.run(
        [
            "python", "-m", "yt_dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", out_template,
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error downloading video{tag}:\n{result.stderr[-1000:]}")

    mp4_files = list(Path(tmp).glob(f"video_{label}*.mp4"))
    if not mp4_files:
        sys.exit(f"Error: yt-dlp did not produce an .mp4 file for {label}.")
    return str(mp4_files[0])


if __name__ == "__main__":
    main()
