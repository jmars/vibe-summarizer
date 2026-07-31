"""Web-archive summarizer — generate AI summaries for archived web content.

Reads web-archive JSONL files (from web-archive-mcp), extracts the content,
calls the LLM for a structured summary, and writes .summary.json alongside.

Usage:
    web-archive-summarizer --batch --max 5
    web-archive-summarizer <file.jsonl>
    web-archive-summarizer --dry-run <file.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .llm import call as llm_call

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARCHIVE_ROOT = Path(
    os.environ.get(
        "SUMMARIZER_WEB_ARCHIVE_DIR",
        str(Path.home() / ".local" / "share" / "web-archive"),
    )
)

SYSTEM_PROMPT = """You are an archivist. Summarize web content concisely.

Output a JSON object with these fields:
- "topic": what the page/search is about (1 line)
- "key_points": 3-5 bullet points of key information
- "source_type": "web_search" or "web_fetch"
- "relevance": "high", "medium", or "low" — how useful this is for future investigation

Return ONLY the JSON object, no markdown, no explanation."""


# ---------------------------------------------------------------------------
# Summarize one entry
# ---------------------------------------------------------------------------

def summarize(archive_path: Path, dry_run: bool = False) -> bool:
    """Summarize a single web-archive JSONL entry. Returns True on success."""
    if not archive_path.suffix == ".jsonl":
        return False

    # Read the JSONL file (each line is a JSON object)
    try:
        lines = [
            json.loads(line)
            for line in archive_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False

    if not lines:
        return False

    # Use the first entry's content for summarization
    entry = lines[0]
    content = entry.get("content", "")
    source = entry.get("source", entry.get("title", archive_path.stem))

    if not content:
        return False

    # Truncate very long content for the prompt
    if len(content) > 8000:
        content = content[:8000] + "\n\n... (truncated)"

    prompt = f"Source: {source}\n\nContent:\n{content}"

    if dry_run:
        print(f"DRY RUN for {archive_path.name}")
        print(f"Source: {source}")
        print(f"Content length: {len(content)} chars")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"\n--- PROMPT ---\n{prompt[:2000]}")
        if len(prompt) > 2000:
            print(f"... ({len(prompt) - 2000} more chars)")
        print("--- END PROMPT ---")
        return True

    print(f"  Summarizing {archive_path.name}...", file=sys.stderr)
    summary = llm_call(SYSTEM_PROMPT, prompt)

    if summary is None:
        summary = json.dumps({
            "topic": source[:120],
            "key_points": ["LLM summarization failed — content archived but not summarized"],
            "source_type": entry.get("type", "unknown"),
            "relevance": "medium",
        })
        print(f"  Wrote fallback summary for {archive_path.name}", file=sys.stderr)

    summary_path = archive_path.with_suffix(".summary.json")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def summarize_all(dry_run: bool = False, max_entries: int = 0) -> tuple[int, int]:
    """Process all archive entries that don't have summaries.

    Returns ``(generated, skipped)``.
    """
    if not ARCHIVE_ROOT.is_dir():
        print("No web-archive directory found.", file=sys.stderr)
        return 0, 0

    entries = sorted(
        [
            p
            for p in ARCHIVE_ROOT.iterdir()
            if p.suffix == ".jsonl" and not p.with_suffix(".summary.json").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    generated = 0
    skipped = 0

    for archive_path in entries:
        if max_entries > 0 and generated >= max_entries:
            break
        if summarize(archive_path, dry_run=dry_run):
            generated += 1

    return generated, skipped
