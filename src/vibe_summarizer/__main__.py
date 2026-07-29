"""CLI entry points for vibe-summarizer.

Two commands are registered via ``pyproject.toml`` ``[project.scripts]``:

    session-summarizer     — summarize coding sessions
    transcript-summarizer  — summarize meeting transcripts
"""

import argparse
import sys
from pathlib import Path


def _validate_within_root(path: Path, root: Path) -> None:
    """Ensure *path* is within *root* (prevents path traversal)."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        print(
            f"Error: path escapes configured directory: {path}",
            file=sys.stderr,
        )
        sys.exit(1)


def _session_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize coding sessions via LLM")
    p.add_argument(
        "session_dir", nargs="?", default=None,
        help="Session directory name or path to summarize",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Process all sessions without summaries",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview prompt without calling the LLM",
    )
    p.add_argument(
        "--max", type=int, default=0, dest="max_sessions",
        help="Maximum sessions to process in batch mode (default: unlimited)",
    )
    return p


def main_session() -> None:
    """CLI entry point: session-summarizer."""
    from vibe_summarizer.session import SESSION_ROOT, summarize, summarize_all

    parser = _session_cli()
    args = parser.parse_args()

    if args.batch:
        gen, skip = summarize_all(dry_run=args.dry_run, max_sessions=args.max_sessions)
        print(f"Generated: {gen}, Already summarized: {skip}")
    elif args.session_dir:
        session_path = Path(args.session_dir)
        if not session_path.is_absolute():
            session_path = SESSION_ROOT / args.session_dir
        _validate_within_root(session_path, SESSION_ROOT)
        if not session_path.is_dir():
            candidates = sorted(SESSION_ROOT.glob(f"{args.session_dir}*"))
            if candidates:
                session_path = candidates[0]
            else:
                print(f"Session not found: {args.session_dir}", file=sys.stderr)
                sys.exit(1)
        ok = summarize(session_path, dry_run=args.dry_run)
        if ok:
            print(f"Done: {session_path.name}")
        else:
            print(f"Failed to summarize {session_path.name}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: process latest session
        if not SESSION_ROOT.is_dir():
            print("No sessions found.", file=sys.stderr)
            sys.exit(1)
        sessions = sorted(
            [
                d for d in SESSION_ROOT.iterdir()
                if d.is_dir() and d.name.startswith("session_")
            ],
            reverse=True,
        )
        if not sessions:
            print("No sessions found.", file=sys.stderr)
            sys.exit(1)
        ok = summarize(sessions[0], dry_run=args.dry_run)
        if ok:
            print(f"Done: {sessions[0].name}")
        else:
            print("Failed.", file=sys.stderr)
            sys.exit(1)


def _transcript_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize meeting transcripts via LLM")
    p.add_argument(
        "transcript", nargs="?", default=None,
        help="Transcript .txt or .docx file to summarize",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Process all transcripts without summaries",
    )
    p.add_argument(
        "--max", type=int, default=0, dest="max_transcripts",
        help="Maximum transcripts to process in batch mode (default: unlimited)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview prompt without calling the LLM",
    )
    return p


def main_transcript() -> None:
    """CLI entry point: transcript-summarizer."""
    from vibe_summarizer.transcript import (
        TRANSCRIPT_DIR,
        summarize,
        summarize_all,
    )

    parser = _transcript_cli()
    args = parser.parse_args()

    if args.batch:
        gen, skip = summarize_all(
            dry_run=args.dry_run, max_transcripts=args.max_transcripts
        )
        print(f"Generated: {gen}, Already summarized: {skip}")
    elif args.transcript:
        tp = Path(args.transcript)
        if not tp.is_absolute():
            tp = TRANSCRIPT_DIR / args.transcript
        _validate_within_root(tp, TRANSCRIPT_DIR)
        if not tp.is_file():
            print(f"Transcript not found: {args.transcript}", file=sys.stderr)
            sys.exit(1)
        ok = summarize(tp, dry_run=args.dry_run)
        if ok:
            print(f"Done: {tp.name}")
        else:
            print("Failed.", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Usage: transcript-summarizer <file> | --batch [--max N] | --dry-run <file>",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    # Allow running as: python -m vibe_summarizer [session|transcript] ...
    if len(sys.argv) < 2:
        print("Usage: python -m vibe_summarizer [session|transcript] ...", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "session":
        main_session()
    elif cmd == "transcript":
        main_transcript()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
