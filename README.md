# Vibe Summarizer

**LLM-powered session and transcript summarizer** — provider-agnostic, works with any OpenAI-compatible API.

## Quick Start

```bash
pip install git+https://github.com/jmars/vibe-summarizer
```

### Configuration

Set these environment variables to point at your LLM provider:

```bash
export SUMMARIZER_API_URL="https://api.openai.com/v1"   # or your provider
export SUMMARIZER_API_KEY="sk-..."                        # your API key
export SUMMARIZER_MODEL="gpt-4o-mini"                     # model name
```

Works with any service that speaks the OpenAI `/v1/chat/completions` protocol:
OpenAI, DeepSeek, Groq, Together, Ollama, LM Studio, and more.

### Summarize a Coding Session

```bash
# Summarize the latest session
session-summarizer

# Summarize a specific session
session-summarizer session_2026-07-29

# Batch process all incomplete sessions
session-summarizer --batch --max 10

# Preview the prompt without calling the LLM
session-summarizer --dry-run session_2026-07-29
```

By default, session data is read from `$XDG_DATA_HOME/vibe/sessions` or
`~/.local/share/vibe/sessions`. Override with `SUMMARIZER_SESSION_ROOT`.

### Summarize a Meeting Transcript

```bash
# Summarize a specific transcript
transcript-summarizer standup-2026-07-29.txt

# Batch process all new transcripts
transcript-summarizer --batch --max 5

# Preview the prompt
transcript-summarizer --dry-run meeting.txt
```

By default, transcripts are read from `~/Dropbox/Apps/Tactiq.io`. Override with
`SUMMARIZER_TRANSCRIPT_DIR`.

## Output Format

### Session summaries (`summary.json`)

```json
{
  "goal": "Fix the login crash on Safari",
  "outcome": "Root cause was a null pointer in the auth middleware. Fixed and tested.",
  "key_decisions": ["Refactored auth middleware to use Option types"],
  "tags": ["bugfix", "auth", "rust"],
  "status": "completed",
  "files_touched": ["src/auth.rs", "tests/auth_test.rs"],
  "generated_at": "2026-07-29T12:00:00Z",
  "model": "gpt-4o-mini",
  "tokens_in": 450,
  "tokens_out": 120
}
```

### Transcript summaries (`.summary.json`)

```json
{
  "topic": "Q3 roadmap planning for the platform team",
  "key_points": ["Auth service needs a v2 API", "CI pipeline migration to GitHub Actions"],
  "decisions": ["Prioritize auth v2 over new features"],
  "action_items": [{"who": "Alice", "what": "Draft auth v2 spec by Friday"}],
  "tags": ["planning", "roadmap", "platform"],
  "generated_at": "2026-07-29T12:00:00Z",
  "model": "gpt-4o-mini",
  "tokens_in": 320,
  "tokens_out": 150
}
```

## Library Usage

```python
from pathlib import Path
from vibe_summarizer.session import summarize, extract_digest, build_prompt
from vibe_summarizer.llm import call

# Extract a digest from a session
digest = extract_digest(Path("/path/to/session"))
prompt = build_prompt(digest, meta={"title": "My Session"})

# Call the LLM directly
summary = call(system_prompt="You are a summarizer.", user_prompt=prompt)
print(summary)
```

## How It Works

1. **Extract** — parse session `messages.jsonl` or transcript `.txt`/`.docx` files
2. **Build prompt** — construct a compact prompt from the conversation structure
3. **Call LLM** — send to any OpenAI-compatible chat completions API
4. **Write output** — structured JSON summary written alongside the source data

If the LLM is unavailable, a structural fallback summary is produced from the
data alone — no API call needed.

## Supported Transcript Formats

- **Tactiq `.txt`** — standard format with header metadata and speaker turns
- **Tactiq `.docx`** — auto-converted to the standard format before parsing

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SUMMARIZER_API_URL` | `https://api.openai.com/v1` | Base URL for chat completions API |
| `SUMMARIZER_API_KEY` | *(required)* | Bearer token for API authentication |
| `SUMMARIZER_MODEL` | `gpt-4o-mini` | Model name passed to the API |
| `SUMMARIZER_SESSION_ROOT` | `~/.local/share/vibe/sessions` | Directory containing session folders |
| `SUMMARIZER_TRANSCRIPT_DIR` | `~/Dropbox/Apps/Tactiq.io` | Directory containing transcript files |

## Security

- **API keys** — never stored or logged. Error messages redact keys before
  printing.
- **Prompt injection** — session and transcript content is included in LLM
  prompts verbatim. Only summarize files from trusted sources. Maliciously
  crafted session data could manipulate the LLM's behaviour.
- **URL validation** — only HTTPS endpoints are accepted (localhost excepted
  for local models). Raw IP addresses are blocked to prevent SSRF.
- **DOCX safety** — XML parsing uses `defusedxml` where available to prevent
  XXE attacks. ZIP path traversal is validated.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup
instructions, development workflow, and pull request guidelines.

This project is licensed under the MIT License — all contributions are accepted
under the same license.

## License

MIT
