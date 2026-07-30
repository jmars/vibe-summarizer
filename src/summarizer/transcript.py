"""Transcript summarizer — extracts digests from meeting transcripts.

Reads Tactiq-format transcript files (``.txt`` or ``.docx``), builds a compact
prompt from the discussion structure, calls an LLM to produce a structured
summary, and writes ``.summary.json`` alongside the transcript.

Usage as library::

    from summarizer.transcript import summarize

    summarize(Path("/path/to/transcript.txt"))

Configuration via environment variables: see ``summarizer.llm``.

Transcript directory defaults to ``~/Dropbox/Apps/Tactiq.io``. Override with
``SUMMARIZER_TRANSCRIPT_DIR``.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

try:
    from defusedxml.ElementTree import ParseError as SafeParseError
    from defusedxml.ElementTree import parse as safe_parse
except ImportError:  # pragma: no cover
    SafeParseError = ET.ParseError  # type: ignore[assignment,misc]
    safe_parse = ET.parse  # type: ignore[assignment]

from summarizer.llm import call as llm_call

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSCRIPT_DIR = Path(
    os.environ.get(
        "SUMMARIZER_TRANSCRIPT_DIR",
        str(Path.home() / "Dropbox" / "Apps" / "Tactiq.io"),
    )
)

TRANSCRIPT_EXTENSIONS = {".txt", ".docx"}

# Prompt sampling
MAX_SAMPLE_TURNS = 15
MAX_END_TURNS = 5
MAX_CHARS_PER_TURN = 300

# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _extract_docx_text(p: Path) -> str:
    """Extract plain text from a Tactiq-synced .docx transcript.

    Uses ``defusedxml`` for safe XML parsing when available, falling back
    to stdlib ``ElementTree``.
    """
    try:
        with ZipFile(p) as z:
            # Validate ZIP contents — no path traversal
            for name in z.namelist():
                if name.startswith("../") or name.startswith("/"):
                    return ""
            with z.open("word/document.xml") as f:
                tree = safe_parse(f)
    except (OSError, ET.ParseError, SafeParseError, KeyError, BadZipFile):
        return ""
    paras = []
    for para in tree.iterfind(".//w:p", _DOCX_NS):
        texts = []
        for t in para.iterfind(".//w:t", _DOCX_NS):
            if t.text:
                texts.append(t.text)
        paras.append("".join(texts))
    return "\n".join(paras)


def _docx_to_standard_txt(p: Path) -> str:
    """Convert a Tactiq .docx transcript to the standard .txt header+turn format."""
    raw = _extract_docx_text(p)
    if not raw:
        return ""

    lines = raw.split("\n")
    stem = p.stem
    meeting_name = stem
    meeting_date_str = ""
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})", stem)
    if m:
        meeting_date_str = f"{m.group(1)} {m.group(2)}:{m.group(3)}"
        meeting_name = re.sub(
            r"\s*-\s*\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d+$", "", stem
        ).strip()

    # Find the transcript body (after a standalone "Transcript" heading line)
    transcript_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^Transcript$", stripped, re.IGNORECASE):
            transcript_start = i + 1
            break

    if transcript_start is None:
        transcript_start = 0

    body = lines[transcript_start:]
    while body and not body[0].strip():
        body = body[1:]

    # Convert speaker format: "00:00 Speaker Name:" to "Speaker Name (00:00:00)"
    turn_re = re.compile(r"^(\d{2}:\d{2})\s+(.+?):(?:\s*(.*))?$")
    out_lines: list[str] = []
    header_written = False
    participants: set[str] = set()

    for line in body:
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        tm = turn_re.match(stripped)
        if tm:
            hhmm = tm.group(1)
            speaker = tm.group(2).strip()
            rest = tm.group(3) or ""
            participants.add(speaker)
            if not header_written:
                out_lines.append(f"Meeting: {meeting_name}")
                out_lines.append(f"Date: {meeting_date_str}")
                out_lines.append("Duration: ?")
                out_lines.append("Platform: MS_TEAMS")
                out_lines.append(f"Participants: {', '.join(sorted(participants))}")
                out_lines.append("=" * 60)
                header_written = True
            out_lines.append("")
            out_lines.append(f"{speaker} ({hhmm}:00)")
            if rest:
                out_lines.append(rest)
        else:
            out_lines.append(stripped)

    if participants:
        for i, line in enumerate(out_lines):
            if line.startswith("Participants:"):
                out_lines[i] = f"Participants: {', '.join(sorted(participants))}"
                break

    return "\n".join(out_lines)


def _read_transcript_text(p: Path) -> str:
    """Read text content from a transcript file (.txt or .docx)."""
    if p.suffix.lower() == ".docx":
        return _docx_to_standard_txt(p)
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

SPEAKER_HEADER_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<ts>\d{2}:\d{2}:\d{2})\)\s*$"
)


def parse_transcript(text: str) -> dict:
    """Parse a Tactiq TXT transcript into header metadata and speaker turns.

    Returns a dict with:
      - ``meeting`` — meeting name
      - ``meeting_date`` — datetime object or None
      - ``duration`` — duration string
      - ``platform`` — platform name
      - ``participants`` — list of participant names
      - ``turns`` — list of ``{speaker, timestamp, text}`` dicts
    """
    lines = text.split("\n")
    result: dict = {
        "meeting": None,
        "meeting_date": None,
        "duration": None,
        "platform": None,
        "participants": [],
        "turns": [],
    }

    header_end = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("==="):
            header_end = i
            break

    if header_end >= 0:
        for line in lines[:header_end]:
            line = line.strip()
            if line.startswith("Meeting:"):
                result["meeting"] = line[len("Meeting:"):].strip()
            elif line.startswith("Date:") and not result["meeting_date"]:
                ds = line[len("Date:"):].strip()
                try:
                    result["meeting_date"] = datetime.strptime(ds, "%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            elif line.startswith("Duration:"):
                result["duration"] = line[len("Duration:"):].strip()
            elif line.startswith("Platform:"):
                result["platform"] = line[len("Platform:"):].strip()
            elif line.startswith("Participants:"):
                raw = line[len("Participants:"):].strip()
                result["participants"] = [
                    p.strip() for p in raw.split(",") if p.strip()
                ]

    current_speaker = None
    current_ts = None
    current_text: list[str] = []

    for i in range(header_end + 1, len(lines)):
        line = lines[i]
        m = SPEAKER_HEADER_RE.match(line.strip())
        if m:
            if current_speaker is not None:
                result["turns"].append({
                    "speaker": current_speaker,
                    "timestamp": current_ts,
                    "text": "\n".join(current_text).strip(),
                })
            current_speaker = m.group("name")
            current_ts = m.group("ts")
            current_text = []
        else:
            if current_speaker is not None:
                current_text.append(line)

    if current_speaker is not None:
        result["turns"].append({
            "speaker": current_speaker,
            "timestamp": current_ts,
            "text": "\n".join(current_text).strip(),
        })

    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(parsed: dict) -> str:
    """Build a compact prompt from a parsed transcript."""
    meeting = parsed.get("meeting") or "Unknown"
    date = parsed.get("meeting_date")
    date_str = date.strftime("%Y-%m-%d %H:%M") if date else "Unknown"
    duration = parsed.get("duration") or "?"
    participants = parsed.get("participants", [])
    turns = parsed.get("turns", [])
    total_turns = len(turns)

    parts = [
        f"Meeting: {meeting}",
        f"Date: {date_str}",
        f"Duration: {duration}",
        f"Participants ({len(participants)}): {', '.join(participants)}",
        f"Total speaker turns: {total_turns}",
        "",
    ]

    # First N turns
    sample = turns[:MAX_SAMPLE_TURNS]
    if total_turns > MAX_SAMPLE_TURNS + MAX_END_TURNS:
        parts.append(f"## First {len(sample)} turns:")
    elif total_turns > MAX_SAMPLE_TURNS:
        # Not enough headroom for a gap — just show all turns
        parts.append("## Discussion:")
        for t in turns:
            text = t["text"][:MAX_CHARS_PER_TURN]
            if len(t["text"]) > MAX_CHARS_PER_TURN:
                text += "..."
            parts.append(f"  [{t['speaker']} {t['timestamp']}] {text}")
        return "\n".join(parts)
    else:
        parts.append("## Discussion:")

    for t in sample:
        text = t["text"][:MAX_CHARS_PER_TURN]
        if len(t["text"]) > MAX_CHARS_PER_TURN:
            text += "..."
        parts.append(f"  [{t['speaker']} {t['timestamp']}] {text}")

    # Middle gap indicator
    if total_turns > MAX_SAMPLE_TURNS + MAX_END_TURNS:
        omitted = total_turns - MAX_SAMPLE_TURNS - MAX_END_TURNS
        parts.append(f"\n  ... ({omitted} turns omitted) ...\n")

    # Last N turns (only when there is a gap — no overlap)
    if total_turns > MAX_SAMPLE_TURNS + MAX_END_TURNS:
        end_sample = turns[-MAX_END_TURNS:]
        parts.append(f"## Last {len(end_sample)} turns:")
        for t in end_sample:
            text = t["text"][:MAX_CHARS_PER_TURN]
            if len(t["text"]) > MAX_CHARS_PER_TURN:
                text += "..."
            parts.append(f"  [{t['speaker']} {t['timestamp']}] {text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meeting transcript summarizer. Your job is to produce a concise, "
    "structured JSON summary of a technical meeting transcript.\n\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{"topic": "<main topic, one sentence>", '
    '"key_points": ["point 1", "point 2"], '
    '"decisions": ["decision 1"], '
    '"action_items": [{"who": "Name", "what": "task description"}], '
    '"tags": ["tag1", "tag2"]}\n\n'
    "Guidelines:\n"
    "- topic: distill the primary subject of discussion into one sentence\n"
    "- key_points: 3-7 bullet points covering the main discussion areas\n"
    "- decisions: concrete decisions made (empty list if none clear)\n"
    "- action_items: specific tasks assigned to people (infer from context, "
    "empty if none)\n"
    "- tags: 2-5 lowercase tags (e.g. \"architecture\", \"refinement\", "
    "\"incident\", \"planning\")\n"
    "- Be concise. Every field should be short.\n"
    "- If the transcript is garbled or incoherent, note that in topic.\n"
    "- Do NOT include markdown, code fences, or any text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Fallback summary (no LLM)
# ---------------------------------------------------------------------------


def fallback_summary(parsed: dict) -> dict:
    """Produce a summary from structural data alone, no LLM needed."""
    meeting = parsed.get("meeting") or "Unknown"
    participants = parsed.get("participants", [])
    turns = parsed.get("turns", [])
    duration = parsed.get("duration", "?")

    # Simple heuristic: find repeated words as topic hints
    all_text = " ".join(t.get("text", "") for t in turns).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "will", "what", "when", "just", "like", "about", "there", "which",
        "would", "could", "should",
    }
    word_freq: dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1

    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

    return {
        "topic": f"Meeting: {meeting} ({duration})",
        "key_points": [
            f"Participants: {', '.join(participants)}"
            if participants
            else "No participants parsed"
        ],
        "decisions": [],
        "action_items": [],
        "tags": top_words,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "fallback (no API)",
        "tokens_in": 0,
        "tokens_out": 0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def summarize(transcript_path: Path, dry_run: bool = False) -> bool:
    """Generate .summary.json for a single transcript.

    Returns True if a summary was generated or already exists.
    """
    summary_path = transcript_path.parent / (transcript_path.stem + ".summary.json")

    if summary_path.exists():
        return True  # already summarized

    if not transcript_path.exists():
        print(f"  SKIP: file not found: {transcript_path}", file=sys.stderr)
        return False

    try:
        text = _read_transcript_text(transcript_path)
    except OSError as e:
        print(f"  SKIP: cannot read {transcript_path.name}: {e}", file=sys.stderr)
        return False

    parsed = parse_transcript(text)
    turns = parsed.get("turns", [])

    if not turns:
        print(
            f"  SKIP: {transcript_path.name} has no speaker turns",
            file=sys.stderr,
        )
        return False

    if not parsed.get("meeting"):
        print(
            f"  SKIP: {transcript_path.name} has no meeting header",
            file=sys.stderr,
        )
        return False

    prompt = build_prompt(parsed)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN for {transcript_path.name}")
        print(f"Meeting: {parsed.get('meeting', '?')}")
        print(f"Turns: {len(turns)}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"\n--- PROMPT ---\n{prompt[:2000]}")
        if len(prompt) > 2000:
            print(f"... ({len(prompt) - 2000} more chars)")
        print("--- END PROMPT ---")
        return True

    print(f"  Summarizing {transcript_path.name}...", file=sys.stderr)
    summary = llm_call(SYSTEM_PROMPT, prompt)

    if summary is None:
        summary = fallback_summary(parsed)
        print(
            f"  Wrote fallback summary for {transcript_path.name}",
            file=sys.stderr,
        )
    else:
        print(
            f"  Wrote LLM summary for {transcript_path.name} "
            f"({summary.get('tokens_in', 0)} + {summary.get('tokens_out', 0)} tokens)",
            file=sys.stderr,
        )

    out_path = transcript_path.parent / (transcript_path.stem + ".summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return True


def summarize_all(dry_run: bool = False, max_transcripts: int = 0) -> tuple[int, int]:
    """Process all transcripts without summaries.

    Returns ``(generated, skipped)``.
    """
    if not TRANSCRIPT_DIR.is_dir():
        print("No transcript directory found.", file=sys.stderr)
        return 0, 0

    transcripts = []
    for ext in TRANSCRIPT_EXTENSIONS:
        transcripts.extend(TRANSCRIPT_DIR.rglob(f"*{ext}"))
    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    generated = 0
    skipped = 0

    for tp in transcripts:
        if max_transcripts > 0 and generated >= max_transcripts:
            break
        summary_path = tp.parent / (tp.stem + ".summary.json")
        if summary_path.exists():
            skipped += 1
            continue
        if summarize(tp, dry_run=dry_run):
            generated += 1

    return generated, skipped
