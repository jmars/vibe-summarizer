"""Session summarizer — extracts digests from coding session logs.

Reads a session directory containing ``messages.jsonl`` and ``meta.json``,
builds a prompt from the conversation structure, calls an LLM to produce a
structured summary, and writes ``summary.json``.

Usage as library::

    from vibe_summarizer.session import summarize

    summarize(Path("/path/to/session_dir"))

Configuration via environment variables: see ``vibe_summarizer.llm``.

Session directory defaults to ``$XDG_DATA_HOME/vibe/sessions`` or
``~/.local/share/vibe/sessions``. Override with ``SUMMARIZER_SESSION_ROOT``.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from vibe_summarizer.llm import call as llm_call

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_ROOT = Path(
    os.environ.get(
        "SUMMARIZER_SESSION_ROOT",
        os.path.join(
            os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")),
            "vibe",
            "sessions",
        ),
    )
)

# Prompt limits
MAX_USER_MESSAGES = 6
MAX_ASSISTANT_MESSAGES = 4
MAX_CHARS_PER_MSG = 800

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _messages_lines(session_dir: Path) -> list[str]:
    """Read non-empty lines from messages.jsonl."""
    path = session_dir / "messages.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    except OSError:
        return []


def _load_meta(session_dir: Path) -> dict | None:
    """Load meta.json from a session directory."""
    path = session_dir / "meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


def _parse_message(line: str) -> dict | None:
    """Parse a JSONL message line."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _extract_content(msg: dict) -> str:
    """Get content from a message dict. Returns empty for tool-call-only messages."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("text")]
        return " ".join(parts)
    return str(content) if content else ""


def _extract_tool_names(msg: dict) -> list[str]:
    """Extract tool call names from an assistant message."""
    tcs = msg.get("tool_calls") or []
    if not isinstance(tcs, list):
        return []
    names = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        if name:
            names.append(name)
    return names


def _extract_file_paths(msg: dict) -> list[str]:
    """Extract file paths from tool call arguments."""
    tcs = msg.get("tool_calls") or []
    if not isinstance(tcs, list):
        return []
    paths: set[str] = set()
    path_keywords = {"file_path", "filePath", "path", "outputPath"}

    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        args_str = fn.get("arguments", "")
        # Normalise: if arguments is already a dict, serialise it
        if isinstance(args_str, dict):
            args_str = json.dumps(args_str)
        if not isinstance(args_str, str):
            continue
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            for match in re.finditer(
                r'\b(?:file_path|path|outputPath)["\']?\s*:\s*["\']([^"\']+)["\']',
                args_str,
            ):
                paths.add(match.group(1))
            continue

        if isinstance(args, dict):
            for key in path_keywords:
                val = args.get(key)
                if isinstance(val, str) and val:
                    if not val.startswith(("http://", "https://")):
                        paths.add(val)

    return sorted(paths)[:20]


# ---------------------------------------------------------------------------
# Digest extraction
# ---------------------------------------------------------------------------


def extract_digest(session_dir: Path) -> dict:
    """Extract a compact conversation digest from a session directory.

    Returns a dict with:
      - ``goal_messages`` — first few user messages
      - ``decision_messages`` — assistant messages with tool calls
      - ``outcome_message`` — last meaningful assistant message
      - ``all_file_paths`` — files touched across the session
      - ``stats`` — message count & role breakdown
    """
    lines = _messages_lines(session_dir)
    if not lines:
        return {}

    all_msgs = []
    for line in lines:
        msg = _parse_message(line)
        if msg is not None:
            all_msgs.append(msg)

    if not all_msgs:
        return {}

    # Stats
    roles: dict[str, int] = {}
    for m in all_msgs:
        r = m.get("role", "?")
        roles[r] = roles.get(r, 0) + 1

    # User messages (goal)
    user_msgs = [m for m in all_msgs if m.get("role") == "user"]
    goal_msgs = user_msgs[:MAX_USER_MESSAGES]

    # Assistant messages with substance
    assistant_msgs = [m for m in all_msgs if m.get("role") == "assistant"]
    substantive = []
    for m in assistant_msgs:
        content = _extract_content(m)
        tool_names = _extract_tool_names(m)
        if content.strip() or tool_names:
            substantive.append(m)

    # Decision messages: assistant messages with tool calls
    decision_msgs = [
        m for m in substantive if _extract_tool_names(m)
    ][:MAX_ASSISTANT_MESSAGES]

    # Outcome: last assistant message with content
    outcome_msg = None
    for m in reversed(substantive):
        content = _extract_content(m).strip()
        if content:
            outcome_msg = m
            break

    # All file paths
    all_file_paths: set[str] = set()
    for m in substantive:
        all_file_paths.update(_extract_file_paths(m))

    return {
        "goal_messages": goal_msgs,
        "decision_messages": decision_msgs,
        "outcome_message": outcome_msg,
        "all_file_paths": sorted(all_file_paths),
        "stats": roles,
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(digest: dict, meta: dict | None) -> str:
    """Build a summarization prompt from a conversation digest."""
    title = (meta or {}).get("title", "untitled")
    start = (meta or {}).get("start_time", "unknown")

    parts = [f"Session: {title}", f"Started: {start}", ""]

    # Goal
    parts.append("## User's Goal (first messages):")
    for i, msg in enumerate(digest.get("goal_messages", []), 1):
        content = _extract_content(msg)[:MAX_CHARS_PER_MSG]
        if content.strip():
            parts.append(f"  [{i}] {content}")
    parts.append("")

    # Key decisions
    if digest.get("decision_messages"):
        parts.append("## Key Actions / Decisions:")
        for msg in digest["decision_messages"]:
            content = _extract_content(msg)[:MAX_CHARS_PER_MSG]
            tool_names = _extract_tool_names(msg)
            if tool_names:
                parts.append(f"  - Used tools: {', '.join(tool_names)}")
            if content.strip():
                parts.append(f"    {content}")
        parts.append("")

    # Outcome
    outcome = digest.get("outcome_message")
    if outcome:
        content = _extract_content(outcome)[:MAX_CHARS_PER_MSG]
        parts.append("## Final Outcome / Last Summary:")
        parts.append(f"  {content}")
        parts.append("")

    # Files
    files = digest.get("all_file_paths", [])
    if files:
        parts.append(f"## Files Touched ({len(files)}):")
        for f in files[:15]:
            parts.append(f"  - {f}")
        parts.append("")

    # Stats
    stats = digest.get("stats", {})
    if stats:
        parts.append(f"## Session Stats: {json.dumps(stats)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a session summarizer for a developer's AI coding assistant. "
    "Your job is to produce a concise, structured JSON summary of a coding session. "
    "Focus on: what the user wanted to achieve, what was actually done, key "
    "architectural or implementation decisions made, and the session's completion "
    "status.\n\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{"goal": "<one-sentence goal>", '
    '"outcome": "<what was achieved, 1-2 sentences>", '
    '"key_decisions": ["decision 1", "decision 2"], '
    '"tags": ["tag1", "tag2"], '
    '"status": "completed|incomplete|aborted"}\n\n'
    "Guidelines:\n"
    "- goal: distill the user's primary intent into one clear sentence\n"
    "- outcome: describe concrete accomplishments; if incomplete say what's left\n"
    "- key_decisions: technical choices, architecture decisions, tool selections "
    "(max 5)\n"
    "- tags: 2-5 lowercase tags (e.g., \"refactoring\", \"bugfix\", \"typescript\", "
    "\"testing\", \"ci-cd\")\n"
    "- status: \"completed\" if goals were met, \"incomplete\" if work started but "
    "not finished, \"aborted\" if the session ended abruptly\n"
    "- Be concise. Every field should be short and actionable.\n"
    "- Do NOT include markdown, code fences, or any text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Fallback summary (no LLM)
# ---------------------------------------------------------------------------


def fallback_summary(digest: dict, meta: dict | None) -> dict:
    """Produce a summary from structural data alone, no LLM needed."""
    goal_msgs = digest.get("goal_messages", [])
    outcome = digest.get("outcome_message")
    decision_msgs = digest.get("decision_messages", [])

    goal = _extract_content(goal_msgs[0])[:200] if goal_msgs else "unknown"
    outcome_text = _extract_content(outcome)[:300] if outcome else "unclear"

    tool_names: set[str] = set()
    for m in decision_msgs:
        tool_names.update(_extract_tool_names(m))

    decisions = []
    if tool_names:
        decisions.append(f"Used tools: {', '.join(sorted(tool_names))}")

    files = digest.get("all_file_paths", [])
    if files:
        decisions.append(f"Touched {len(files)} file(s)")

    return {
        "goal": goal[:150],
        "outcome": outcome_text[:250],
        "key_decisions": decisions,
        "tags": [],
        "status": "completed" if outcome else "incomplete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "fallback (no API)",
        "tokens_in": 0,
        "tokens_out": 0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def summarize(session_dir: Path, dry_run: bool = False) -> bool:
    """Generate summary.json for a single session.

    Returns True if a summary was generated or already exists.
    """
    summary_path = session_dir / "summary.json"

    if summary_path.exists() and not dry_run:
        return True

    meta = _load_meta(session_dir)
    if meta is None:
        print(f"  SKIP: no valid meta.json in {session_dir.name}", file=sys.stderr)
        return False

    # Only summarize completed sessions
    if not meta.get("end_time"):
        print(
            f"  SKIP: session {session_dir.name} not yet complete (no end_time)",
            file=sys.stderr,
        )
        return False

    lines = _messages_lines(session_dir)
    if len(lines) < 2:
        print(
            f"  SKIP: session {session_dir.name} too short ({len(lines)} messages)",
            file=sys.stderr,
        )
        return False

    digest = extract_digest(session_dir)
    if not digest or not digest.get("goal_messages"):
        print(
            f"  SKIP: no user messages found in {session_dir.name}",
            file=sys.stderr,
        )
        return False

    prompt = build_prompt(digest, meta)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN for {session_dir.name}")
        print(f"Title: {meta.get('title', '?')}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"\n--- PROMPT ---\n{prompt}\n--- END PROMPT ---")
        return True

    print(f"  Summarizing {session_dir.name}...", file=sys.stderr)
    summary = llm_call(SYSTEM_PROMPT, prompt)

    if summary is None:
        summary = fallback_summary(digest, meta)
        print(f"  Wrote fallback summary for {session_dir.name}", file=sys.stderr)
    else:
        print(
            f"  Wrote LLM summary for {session_dir.name} "
            f"({summary.get('tokens_in', 0)} + {summary.get('tokens_out', 0)} tokens)",
            file=sys.stderr,
        )

    summary["files_touched"] = digest.get("all_file_paths", [])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return True


def summarize_all(dry_run: bool = False, max_sessions: int = 0) -> tuple[int, int]:
    """Process all sessions that don't have summaries.

    Returns ``(generated, skipped)``.
    """
    if not SESSION_ROOT.is_dir():
        print("No session directory found.", file=sys.stderr)
        return 0, 0

    sessions = sorted(
        [
            d
            for d in SESSION_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ],
        reverse=True,
    )

    generated = 0
    skipped = 0

    for session_dir in sessions:
        if max_sessions > 0 and generated >= max_sessions:
            break
        summary_path = session_dir / "summary.json"
        if summary_path.exists():
            skipped += 1
            continue
        if summarize(session_dir, dry_run=dry_run):
            generated += 1

    return generated, skipped
