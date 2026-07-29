"""Tests for vibe_summarizer.transcript — transcript summarizer."""

import json
import tempfile
from pathlib import Path

import pytest

from vibe_summarizer import transcript


SAMPLE_TXT = """Meeting: Standup
Date: 2026-01-15 09:30
Duration: 15 min
Platform: MS_TEAMS
Participants: Alice, Bob, Charlie
============================================================

Alice (09:30:00)
Good morning everyone. Let's go through updates.

Bob (09:31:30)
I finished the auth refactor yesterday. PR is up for review.

Charlie (09:32:00)
I'm blocked on the deployment pipeline. Need help from ops.

Alice (09:33:00)
I'll follow up with ops after this. Bob, when can we merge?

Bob (09:34:00)
Should be ready today after review passes.

Alice (09:35:00)
Great. Anything else? No? Thanks everyone.
"""


class TestParseTranscript:
    def test_parse_basic(self):
        parsed = transcript.parse_transcript(SAMPLE_TXT)
        assert parsed["meeting"] == "Standup"
        assert parsed["duration"] == "15 min"
        assert parsed["platform"] == "MS_TEAMS"
        assert parsed["participants"] == ["Alice", "Bob", "Charlie"]
        assert len(parsed["turns"]) == 6
        assert parsed["turns"][0]["speaker"] == "Alice"
        assert parsed["turns"][0]["timestamp"] == "09:30:00"
        assert "Good morning" in parsed["turns"][0]["text"]

    def test_parse_empty(self):
        parsed = transcript.parse_transcript("")
        assert parsed["meeting"] is None
        assert parsed["turns"] == []

    def test_parse_no_header(self):
        parsed = transcript.parse_transcript("just some text\nno header\n")
        assert parsed["meeting"] is None
        assert parsed["turns"] == []

    def test_parse_date(self):
        parsed = transcript.parse_transcript(SAMPLE_TXT)
        assert parsed["meeting_date"] is not None
        assert parsed["meeting_date"].month == 1
        assert parsed["meeting_date"].day == 15


class TestPromptBuilding:
    def test_build_prompt(self):
        parsed = transcript.parse_transcript(SAMPLE_TXT)
        prompt = transcript.build_prompt(parsed)
        assert "Meeting: Standup" in prompt
        assert "Participants (3): Alice, Bob, Charlie" in prompt
        assert "auth refactor" in prompt
        assert "Total speaker turns: 6" in prompt


class TestFallbackSummary:
    def test_fallback(self):
        parsed = transcript.parse_transcript(SAMPLE_TXT)
        summary = transcript.fallback_summary(parsed)
        assert "Standup" in summary["topic"]
        assert summary["model"] == "fallback (no API)"


class TestSummarize:
    def test_already_summarized(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(SAMPLE_TXT)
            txt_path = Path(f.name)

        try:
            # Create a pre-existing summary
            summary_path = txt_path.parent / (txt_path.stem + ".summary.json")
            summary_path.write_text("{}")

            assert transcript.summarize(txt_path) is True
        finally:
            txt_path.unlink(missing_ok=True)
            if summary_path.exists():
                summary_path.unlink(missing_ok=True)

    def test_missing_file(self):
        assert transcript.summarize(Path("/nonexistent/transcript.txt")) is False

    def test_no_turns(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Meeting: Empty\nDate: 2026-01-01 00:00\n====\n")
            txt_path = Path(f.name)

        try:
            assert transcript.summarize(txt_path) is False
        finally:
            txt_path.unlink(missing_ok=True)


class TestTranscriptDir:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_TRANSCRIPT_DIR", "/custom/transcripts")
        import importlib
        import vibe_summarizer.transcript as tmod
        importlib.reload(tmod)
        assert str(tmod.TRANSCRIPT_DIR) == "/custom/transcripts"


class TestDocxExtraction:
    def test_extract_empty_docx(self):
        # Non-existent file
        assert transcript._extract_docx_text(Path("/nonexistent.docx")) == ""
