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
 LLM CLI          extractive summarization — selects which segment indices
    │              to keep so total duration ≈ your target
    │              (gemini by default; any stdin-reading CLI works)
    │
    ▼
 ffmpeg           cuts those segments, concatenates into output .mp4
```

The key insight: instead of asking the LLM to paraphrase and then trying to match text back to timestamps, we pass it a **numbered list of segments** and ask it to return **indices**. No fuzzy matching, no timestamp alignment risk.

---

## Requirements

- Python 3.10+
- `ffmpeg` on your PATH ([download](https://ffmpeg.org/download.html))
- An LLM CLI of your choice (default: [`gemini`](https://github.com/google-gemini/gemini-cli)).
  Or use `--local` for fully offline mode — no CLI needed.

Install Python deps:
```bash
pip install -r requirements.txt
```

The LLM CLI manages its own auth — no API keys handled by this tool.
Common choices:
- `gemini` — free tier, just `npm install -g @google/gemini-cli` and sign in
- `claude -p` — set up with Claude Code
- `ollama run llama3` — fully local, no cloud
- `llm -m gpt-4` — Simon Willison's `llm` tool, supports many providers

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
  --llm-cmd CMD      LLM CLI to invoke (prompt piped to stdin).
                     Default: 'gemini'. Try 'claude -p', 'ollama run llama3', etc.
  --local            Use local sentence embeddings instead of an LLM
                     (no auth needed; downloads ~130MB model on first run)
  --reencode         Re-encode with libx264/aac (fixes A/V sync on some videos)

One of --ratio or --duration is required.
```

### Examples

```bash
# Keep the 25% most important content (uses gemini CLI)
python summarizer.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --ratio 0.25

# Use Claude Code instead
python summarizer.py "..." --ratio 0.25 --llm-cmd "claude -p"

# Use a fully-local Ollama model
python summarizer.py "..." --ratio 0.25 --llm-cmd "ollama run llama3"

# Skip the LLM entirely — pure local embeddings
python summarizer.py "..." --ratio 0.25 --local

# Make a 90-second highlight reel
python summarizer.py "..." --duration 90
```

### LLM CLI vs --local

| | `--local` | LLM CLI (default) |
|---|---|---|
| Auth/CLI install | None | A CLI of your choice |
| Model download | ~130MB (once) | None (CLI handles it) |
| Speed | ~2-3s on CPU | ~2-10s per call |
| Quality | Good (semantic similarity + MMR) | Best (editorial judgement) |
| Works offline | Yes | Only with offline CLIs (e.g. `ollama`) |

The local mode uses `BAAI/bge-small-en-v1.5` via `sentence-transformers`. It embeds every segment and picks the highest-scoring ones via MMR (balances relevance to the document centroid against redundancy with already-selected segments).

---

## Project structure

```
tiny-deep-diver/
├── summarizer.py      CLI entry point
├── transcript.py      fetch & parse transcript segments from YouTube
├── extract.py         LLM-CLI-based extractive segment selection
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
