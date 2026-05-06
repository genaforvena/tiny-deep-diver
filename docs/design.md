# Design Notes

## Why extractive, not abstractive summarization?

Abstractive summarizers (e.g. "summarize this transcript") produce paraphrased text. To build a video from that you'd need to fuzzy-match the paraphrase back to the source — error-prone and often fails on exact quotes.

**Extractive summarization** selects a subset of the original segments unchanged. We pass Claude a numbered list:

```
[0] (0.0s–4.2s) "Welcome to the channel, today we're going to talk about..."
[1] (4.2s–9.1s) "The main thing most people get wrong is..."
...
```

And ask it to return `[1, 3, 7, 12, ...]` — just indices. Zero matching required.

## Why segment-level, not word-level timestamps?

YouTube's auto-captions give reliable **segment-level** timestamps (2–5 second chunks). Word-level accuracy from Whisper varies by ±100–500ms and requires running a local model. For this use case, segment-level is precise enough — a 2-second padding is barely noticeable in the output video.

## ffmpeg strategy

Segments are extracted with `-c copy` (stream copy — no re-encode). This is:
- Fast (~real-time extraction regardless of video length)
- Lossless quality
- Risk: keyframe alignment. ffmpeg seeks to the nearest keyframe before the cut point, so a segment starting at 10.0s might actually start at 9.7s. This is a known trade-off and acceptable for this use case.

Concatenation uses the ffmpeg **concat demuxer** (a file listing segments) rather than the concat filter, because demuxer works with stream copy and doesn't require all streams to match exactly.

## Claude model choice

Uses `claude-opus-4-7` for summarization quality. The transcript is passed in a single user message; the response is a JSON array of integers. Low token usage (transcripts are short) so cost is negligible even for long videos.

## Transcript fallback chain

1. `youtube-transcript-api` — fastest, pure HTTP, no video download needed
2. `yt-dlp --write-auto-subs` — downloads `.vtt` caption file alongside the video
3. *(roadmap)* Whisper on the audio track — for videos with no captions at all
