"""
Fetch and parse transcript segments from a YouTube video.

Returns a list of dicts: [{start: float, end: float, text: str}, ...]
where start/end are in seconds.

Fallback chain:
  1. youtube-transcript-api (fast, no download)
  2. yt-dlp --write-auto-subs (downloads .vtt alongside video)
"""

import re
import subprocess
import tempfile
from pathlib import Path


def get_transcript(url: str) -> list[dict]:
    """Return timed segments for the given YouTube URL."""
    try:
        return _from_transcript_api(url)
    except Exception:
        pass
    return _from_yt_dlp(url)


def total_duration(segments: list[dict]) -> float:
    if not segments:
        return 0.0
    return segments[-1]["end"]


# ── youtube-transcript-api ────────────────────────────────────────────────────

def _from_transcript_api(url: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _extract_video_id(url)
    # v1.x uses an instance; fetch() returns an iterable of FetchedTranscriptSnippet
    raw = YouTubeTranscriptApi().fetch(video_id)
    segments = [
        {"start": e.start, "end": e.start + e.duration, "text": e.text.strip()}
        for e in raw
        if e.text.strip()
    ]
    return _merge_short(segments)


# ── yt-dlp subtitle fallback ──────────────────────────────────────────────────

def _from_yt_dlp(url: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "python", "-m", "yt_dlp",
                "--write-auto-subs",
                "--skip-download",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--output", str(Path(tmp) / "subs"),
                url,
            ],
            check=True,
            capture_output=True,
        )
        vtt_files = list(Path(tmp).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError("yt-dlp found no subtitles for this video")
        return _parse_vtt(vtt_files[0])


def _parse_vtt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n([\s\S]*?)(?=\n\n|\Z)"
    )
    segments = []
    for m in pattern.finditer(text):
        start = _ts_to_sec(m.group(1))
        end = _ts_to_sec(m.group(2))
        body = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        if body:
            segments.append({"start": start, "end": end, "text": body})

    # deduplicate consecutive identical lines (VTT repeats lines during karaoke)
    deduped = []
    for seg in segments:
        if deduped and deduped[-1]["text"] == seg["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)

    return _merge_short(deduped)


# ── helpers ───────────────────────────────────────────────────────────────────

def _merge_short(segments: list[dict], min_duration: float = 1.0) -> list[dict]:
    """Merge segments shorter than min_duration into the next one."""
    result = []
    for seg in segments:
        if result and (seg["start"] - result[-1]["end"] < 0.1) and \
                (result[-1]["end"] - result[-1]["start"] < min_duration):
            result[-1]["end"] = seg["end"]
            result[-1]["text"] += " " + seg["text"]
        else:
            result.append(dict(seg))
    return result


def _ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from URL: {url}")
