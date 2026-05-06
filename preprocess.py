"""
Segment preprocessing: filler pruning, sentence-aware grouping, boundary padding.

Pipeline (applied in order):
  1. prune_fillers    -- remove sub-1s chunks, stutter patterns, channel intros
  2. group_sentences  -- merge caption chunks into whole sentences (~50–200 chars)
  3. add_padding      -- extend each segment by PAD_SECS on each side (clamped)
"""

from __future__ import annotations

import re

# seconds of silence to add before/after each selected segment on playback
PAD_SECS = 0.15

_MIN_SEGMENT_DURATION = 0.8   # seconds; shorter than this is usually a caption artifact

# Patterns that indicate filler/meta content; matched against lowercased text
_FILLER_PATTERNS = [
    r"^\[.{1,30}\]$",               # [music] [applause] [laughter] etc.
    r"^uh+\s*$",                    # "uh" / "uhh"
    r"^um+\s*$",
    r"^like\s*$",
    r"^so\s*$",
    r"^you know\s*$",
    r"^i mean\s*$",
    r"^okay\s*$",
    r"^right\s*$",
    r"^alright\s*$",
    r"^\W*$",                       # punctuation / whitespace only
    # channel meta — first/last N seconds handled separately
    r"subscribe",
    r"hit the bell",
    r"like and subscribe",
    r"smash the like",
    r"join the channel",
    r"patreon",
    r"support the channel",
    r"check the description",
]
_FILLER_RE = re.compile("|".join(f"(?:{p})" for p in _FILLER_PATTERNS))

# Channel intro/outro: prune segments in first/last window if they match promo text
_PROMO_WINDOW_SECS = 60

# Sentence-grouping: merge chunks until accumulated text looks like a sentence end
_TARGET_GROUP_CHARS = 120       # aim for ~120 chars per group
_MAX_GROUP_CHARS = 280          # hard cap to avoid huge groups
_SENTENCE_END_RE = re.compile(r"[.!?]['\"]?\s*$")


def preprocess(segments: list[dict]) -> list[dict]:
    """Full preprocessing pipeline. Returns cleaned, grouped, padded segments."""
    if not segments:
        return segments
    total = segments[-1]["end"]
    segs = prune_fillers(segments, total_duration=total)
    segs = group_sentences(segs)
    return segs   # padding applied post-selection in cut stage, not here


def prune_fillers(segments: list[dict], total_duration: float = 0) -> list[dict]:
    """Remove sub-duration, pure-filler, and promo segments."""
    result = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < _MIN_SEGMENT_DURATION:
            continue
        text = seg["text"].strip().lower()
        if _FILLER_RE.search(text):
            # in promo windows apply promo-specific patterns only; elsewhere skip all matches
            in_promo = (
                total_duration > 0 and (
                    seg["start"] < _PROMO_WINDOW_SECS or
                    seg["end"] > total_duration - _PROMO_WINDOW_SECS
                )
            )
            if in_promo or not any(
                re.search(p, text)
                for p in [r"subscribe", r"hit the bell", r"like and subscribe",
                           r"smash the like", r"join the channel", r"patreon",
                           r"support the channel", r"check the description"]
            ):
                continue
        result.append(seg)
    return result


def group_sentences(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive caption chunks into whole sentences.

    Groups end when:
    - accumulated text ends with sentence-ending punctuation AND exceeds target length, OR
    - accumulated text exceeds hard cap.
    Adjacent segments are merged only if there is no gap >1s between them.
    """
    if not segments:
        return segments

    result: list[dict] = []
    group_start = segments[0]["start"]
    group_end = segments[0]["end"]
    group_texts: list[str] = [segments[0]["text"].strip()]

    def _flush():
        if group_texts:
            result.append({
                "start": group_start,
                "end": group_end,
                "text": " ".join(group_texts),
            })

    for seg in segments[1:]:
        gap = seg["start"] - group_end
        accumulated = " ".join(group_texts + [seg["text"].strip()])

        # break the group if gap too large or hard cap exceeded
        if gap > 1.0 or len(accumulated) > _MAX_GROUP_CHARS:
            _flush()
            group_start = seg["start"]
            group_end = seg["end"]
            group_texts = [seg["text"].strip()]
            continue

        # extend the group
        group_end = seg["end"]
        group_texts.append(seg["text"].strip())

        # flush when we reach a natural sentence boundary
        if len(accumulated) >= _TARGET_GROUP_CHARS and _SENTENCE_END_RE.search(accumulated):
            _flush()
            group_start = seg["end"]   # will be overwritten by next seg
            group_end = seg["end"]
            group_texts = []

    _flush()
    return [r for r in result if r["text"].strip()]


def apply_padding(segments: list[dict], total_duration: float, pad: float = PAD_SECS) -> list[dict]:
    """
    Expand each segment by `pad` seconds on each side, clamped to [0, total_duration].
    Call this on the *selected* segments before passing to cutter.
    """
    result = []
    for seg in segments:
        result.append({
            **seg,
            "start": max(0.0, seg["start"] - pad),
            "end": min(total_duration, seg["end"] + pad),
        })
    return result
