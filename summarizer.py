"""
tiny-deep-diver — video-to-video summarizer

Usage:
    python summarizer.py <youtube_url> --ratio 0.3
    python summarizer.py <youtube_url> --duration 120
    python summarizer.py <youtube_url> --ratio 0.4 --output highlights.mp4 --reencode
    python summarizer.py <youtube_url> --ratio 0.3 --local   # no API key needed
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from cutter import cut_and_join
from transcript import get_transcript, total_duration


def main() -> None:
    args = _parse_args()

    print("-> Fetching transcript…")
    segments = get_transcript(args.url)
    if not segments:
        sys.exit("Error: could not retrieve transcript for this video.")

    orig_duration = total_duration(segments)
    target = _resolve_target(args, orig_duration)
    print(
        f"  {len(segments)} segments · {orig_duration:.0f}s original "
        f"-> {target:.0f}s target ({target / orig_duration * 100:.0f}%)"
    )

    if args.local:
        from extract_local import select_segments
        print("-> Selecting key segments (local embeddings)…")
    else:
        from extract import select_segments
        print("-> Selecting key segments with Claude…")

    selected = select_segments(segments, target)
    if not selected:
        sys.exit("Error: no segments were selected.")

    kept = sum(s["end"] - s["start"] for s in selected)
    print(f"  {len(selected)} segments selected · {kept:.0f}s total")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = _download_video(args.url, tmp)

        print("-> Cutting and joining…")
        cut_and_join(video_path, selected, args.output, reencode=args.reencode)

    print(f"\nDone: {args.output}")
    print(
        f"  {orig_duration:.0f}s -> {kept:.0f}s "
        f"({kept / orig_duration * 100:.0f}% of original)"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize a YouTube video into a shorter video using key segments."
    )
    p.add_argument("url", help="YouTube video URL")
    p.add_argument(
        "--ratio", type=float, default=None,
        help="Keep this fraction of original duration (0.0–1.0)"
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
        "--reencode", action="store_true",
        help="Re-encode output (slower but fixes A/V sync issues on some videos)"
    )
    p.add_argument(
        "--local", action="store_true",
        help="Use local sentence embeddings instead of Claude (no API key needed)"
    )
    args = p.parse_args()
    if args.ratio is None and args.duration is None:
        p.error("One of --ratio or --duration is required.")
    if args.ratio is not None and not (0.0 < args.ratio <= 1.0):
        p.error("--ratio must be between 0.0 and 1.0 (exclusive).")
    return args


def _resolve_target(args: argparse.Namespace, orig_duration: float) -> float:
    if args.duration is not None:
        return float(args.duration)
    return args.ratio * orig_duration


def _download_video(url: str, tmp: str) -> str:
    out_template = str(Path(tmp) / "video.%(ext)s")
    print("-> Downloading video…")
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
        sys.exit(f"Error downloading video:\n{result.stderr[-1000:]}")

    mp4_files = list(Path(tmp).glob("*.mp4"))
    if not mp4_files:
        sys.exit("Error: yt-dlp did not produce an .mp4 file.")
    return str(mp4_files[0])


if __name__ == "__main__":
    main()
