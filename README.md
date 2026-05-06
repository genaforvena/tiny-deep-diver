# tiny-deep-diver

**Video-to-video summarizer.** Give it a YouTube URL, a compression target, and it hands you back a shorter video built from the most important segments of the original — no re-encoding, no voice-over, just the real footage.

```
python summarizer.py "https://youtube.com/watch?v=..." --ratio 0.3
# → summary.mp4  (30% of original duration, key segments only)
```

---

## How it works

```
YouTube URL
    │
    ▼
 yt-dlp          download video + fetch auto-captions
    │
    ▼
 Transcript       parse captions into timed segments [{start, end, text}]
    │              (falls back to yt-dlp subtitle download if API unavailable)
    │
    ▼
 Claude API       extractive summarization — selects which segment indices
    │              to keep so total duration ≈ your target
    │
    ▼
 ffmpeg           cuts those segments, concatenates into output .mp4
```

The key insight: instead of asking the LLM to paraphrase and then trying to match text back to timestamps, we pass Claude a **numbered list of segments** and ask it to return **indices**. No fuzzy matching, no timestamp alignment risk.

---

## Requirements

- Python 3.10+
- `ffmpeg` on your PATH ([download](https://ffmpeg.org/download.html))
- An Anthropic API key

Install Python deps:
```bash
pip install -r requirements.txt
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Linux/macOS
$env:ANTHROPIC_API_KEY="sk-ant-..."   # PowerShell
```

---

## Usage

```
python summarizer.py <youtube_url> [options]

Options:
  --ratio FLOAT      Keep this fraction of original duration (0.0–1.0)
                     e.g. --ratio 0.3  keeps the 30% most important content
  --duration INT     Target output length in seconds
                     e.g. --duration 120  makes a ~2 minute summary
  --output PATH      Output file (default: summary.mp4)
  --local            Use local sentence embeddings instead of Claude
                     (no API key needed; downloads ~80MB model on first run)
  --reencode         Re-encode with libx264/aac (fixes A/V sync on some videos)

One of --ratio or --duration is required.
```

### Examples

```bash
# Keep the 25% most important content (uses Claude)
python summarizer.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --ratio 0.25

# Same but fully local — no API key needed
python summarizer.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --ratio 0.25 --local

# Make a 90-second highlight reel
python summarizer.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --duration 90

# Custom output name
python summarizer.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --ratio 0.4 --output highlights.mp4
```

### Local vs Claude

| | `--local` | Claude (default) |
|---|---|---|
| API key needed | No | Yes |
| Model download | ~80MB (once) | None |
| Speed | ~1-2s on CPU | ~2-5s API call |
| Quality | Good (semantic similarity ranking) | Best (editorial judgement) |
| Works offline | Yes | No |

The local mode uses `all-MiniLM-L6-v2` via `sentence-transformers`. It embeds every segment, ranks them by cosine similarity to the document centroid, and greedily picks the highest-scoring ones until the duration target is met.

---

## Project structure

```
tiny-deep-diver/
├── summarizer.py      CLI entry point
├── transcript.py      fetch & parse transcript segments from YouTube
├── extract.py         Claude-based extractive segment selection
├── cutter.py          ffmpeg segment cutting + concatenation
├── requirements.txt
└── docs/
    └── design.md      Architecture decisions and trade-offs
```

---

## Limitations

- Requires a video with auto-generated or manual captions on YouTube. Videos with no captions at all will fail (Whisper fallback is on the roadmap).
- Stream-copy (`-c copy`) is used for speed — if the output has A/V sync issues on certain videos, re-encoding mode can be enabled with `--reencode`.
- Very short videos (<60s) or very aggressive ratios (<0.1) may produce choppy results.

---

## Roadmap

- [ ] Whisper fallback for videos without captions
- [ ] `--reencode` flag for problematic codecs
- [ ] Chapter-aware summarization
- [ ] Web UI
