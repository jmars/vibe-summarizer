"""Tests for vibe_summarizer.session — session summarizer."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vibe_summarizer import session


class TestMessageParsing:
    def test_parse_valid_json(self):
        msg = session._parse_message('{"role": "user", "content": "hello"}')
        assert msg == {"role": "user", "content": "hello"}

    def test_parse_invalid_json_returns_none(self):
        assert session._parse_message("not json") is None

    def test_extract_string_content(self):
        msg = {"role": "user", "content": "fix the bug"}
        assert session._extract_content(msg) == "fix the bug"

    def test_extract_multimodal_content(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "part 1"},
            {"type": "image", "url": "http://x"},
            {"type": "text", "text": "part 2"},
        ]}
        assert session._extract_content(msg) == "part 1 part 2"

    def test_extract_empty_content(self):
        msg = {"role": "assistant", "tool_calls": [{"function": {"name": "bash"}}]}
        assert session._extract_content(msg) == ""

    def test_extract_tool_names(self):
        msg = {"tool_calls": [
            {"function": {"name": "bash"}},
            {"function": {"name": "read_file"}},
        ]}
        names = session._extract_tool_names(msg)
        assert set(names) == {"bash", "read_file"}

    def test_extract_file_paths_from_args(self):
        msg = {"tool_calls": [
            {"function": {
                "name": "write_file",
                "arguments": json.dumps({"file_path": "/tmp/test.py"}),
            }},
        ]}
        paths = session._extract_file_paths(msg)
        assert "/tmp/test.py" in paths

    def test_extract_file_paths_skips_urls(self):
        msg = {"tool_calls": [
            {"function": {
                "name": "web_fetch",
                "arguments": json.dumps({"url": "https://example.com"}),
            }},
        ]}
        paths = session._extract_file_paths(msg)
        assert "https://example.com" not in paths


class TestDigestExtraction:
    def _make_session(self, messages: list[dict]) -> Path:
        d = Path(tempfile.mkdtemp())
        lines = [json.dumps(m) for m in messages]
        (d / "messages.jsonl").write_text("\n".join(lines))
        (d / "meta.json").write_text(json.dumps({
            "title": "Test Session",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T01:00:00Z",
        }))
        return d

    def test_extract_basic_digest(self):
        d = self._make_session([
            {"role": "user", "content": "fix the login bug"},
            {"role": "assistant", "content": "I'll look at that", "tool_calls": [
                {"function": {"name": "grep", "arguments": json.dumps({"pattern": "login"})}},
            ]},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "done — the login bug is fixed"},
        ])
        digest = session.extract_digest(d)
        assert len(digest["goal_messages"]) == 2
        assert digest["goal_messages"][0]["content"] == "fix the login bug"
        assert len(digest["decision_messages"]) == 1
        assert digest["outcome_message"]["content"] == "done — the login bug is fixed"
        assert digest["stats"]["user"] == 2
        assert digest["stats"]["assistant"] == 2

    def test_empty_session(self):
        d = Path(tempfile.mkdtemp())
        (d / "messages.jsonl").write_text("")
        (d / "meta.json").write_text(json.dumps({
            "title": "Empty",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T01:00:00Z",
        }))
        digest = session.extract_digest(d)
        assert digest == {}


class TestPromptBuilding:
    def test_build_prompt(self):
        digest = {
            "goal_messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "user", "content": "it crashes on login"},
            ],
            "decision_messages": [
                {"role": "assistant", "content": "found the issue",
                 "tool_calls": [
                     {"function": {"name": "edit", "arguments": json.dumps({"file_path": "src/main.py"})}},
                 ]},
            ],
            "outcome_message": {"role": "assistant", "content": "bug fixed"},
            "all_file_paths": ["src/main.py", "src/utils.py"],
            "stats": {"user": 2, "assistant": 2},
        }
        meta = {"title": "Test Session", "start_time": "2026-01-01"}
        prompt = session.build_prompt(digest, meta)
        assert "Session: Test Session" in prompt
        assert "fix the bug" in prompt
        assert "Used tools: edit" in prompt
        assert "src/main.py" in prompt
        assert "Files Touched (2)" in prompt


class TestFallbackSummary:
    def test_fallback_with_data(self):
        digest = {
            "goal_messages": [
                {"role": "user", "content": "deploy the app"},
            ],
            "decision_messages": [
                {"role": "assistant", "content": "",
                 "tool_calls": [
                     {"function": {"name": "bash", "arguments": json.dumps({"command": "deploy"})}},
                 ]},
            ],
            "outcome_message": {"role": "assistant", "content": "deployed successfully"},
            "all_file_paths": ["Dockerfile"],
            "stats": {},
        }
        summary = session.fallback_summary(digest, None)
        assert summary["goal"] == "deploy the app"
        assert "deployed successfully" in summary["outcome"]
        assert "bash" in summary["key_decisions"][0]
        assert summary["status"] == "completed"
        assert summary["model"] == "fallback (no API)"


class TestSummarize:
    def test_already_summarized(self):
        d = Path(tempfile.mkdtemp())
        (d / "summary.json").write_text("{}")
        (d / "meta.json").write_text(json.dumps({
            "title": "Done",
            "start_time": "X",
            "end_time": "Y",
        }))
        (d / "messages.jsonl").write_text(
            json.dumps({"role": "user", "content": "hello"}) + "\n" +
            json.dumps({"role": "assistant", "content": "hi"})
        )
        assert session.summarize(d) is True

    def test_skip_incomplete_session(self):
        d = Path(tempfile.mkdtemp())
        (d / "meta.json").write_text(json.dumps({
            "title": "In progress",
            "start_time": "X",
            # no end_time
        }))
        (d / "messages.jsonl").write_text(
            json.dumps({"role": "user", "content": "hello"})
        )
        assert session.summarize(d) is False

    def test_skip_too_short(self):
        d = Path(tempfile.mkdtemp())
        (d / "meta.json").write_text(json.dumps({
            "title": "Short",
            "start_time": "X",
            "end_time": "Y",
        }))
        (d / "messages.jsonl").write_text(json.dumps({"role": "user", "content": "hi"}))
        assert session.summarize(d) is False


class TestSessionRoot:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("SUMMARIZER_SESSION_ROOT", "/custom/sessions")
        # Re-import to pick up new env
        import importlib
        import vibe_summarizer.session as sess
        importlib.reload(sess)
        assert str(sess.SESSION_ROOT) == "/custom/sessions"


class TestMessagesLines:
    def test_read_messages(self):
        d = Path(tempfile.mkdtemp())
        (d / "messages.jsonl").write_text('{"a":1}\n\n{"b":2}\n')
        lines = session._messages_lines(d)
        assert len(lines) == 2

    def test_missing_file(self):
        d = Path(tempfile.mkdtemp())
        lines = session._messages_lines(d)
        assert lines == []
