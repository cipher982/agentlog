from __future__ import annotations

import json
from pathlib import Path

from agentlog import EventParser


def test_event_parser_codex_jsonl_messages_only(tmp_path: Path) -> None:
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "id": "sid-1",
            "timestamp": "2025-01-01T00:00:00.000Z",
            "instructions": None,
            "git": {"branch": "main", "commit_hash": "abc", "repository_url": "git@github.com:org/repo.git"},
        },
        {"record_type": "state"},
        {"type": "message", "id": None, "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "message", "id": None, "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "id": "c1", "name": "tool", "arguments": {}},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "message", "id": None, "role": "user", "content": [{"type": "input_text", "text": "bye"}]},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert [e.role for e in events] == ["user", "assistant", "user"]
    assert [e.content_text for e in events] == ["hi", "hello", "bye"]
    assert all(e.session_id == "sid-1" for e in events)


def test_event_parser_codex_response_item_message_payload(tmp_path: Path) -> None:
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "timestamp": "2025-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": "sid-3",
                "timestamp": "2025-01-01T00:00:00.000Z",
                "cwd": "/Users/davidrose/git/agentlog",
                "git": {"branch": "main", "commit_hash": "abc", "repository_url": "git@github.com:org/repo.git"},
            },
        },
        {"timestamp": "2025-01-01T00:00:01.000Z", "type": "turn_context", "payload": {"cwd": "/Users/davidrose/git/agentlog"}},
        {
            "timestamp": "2025-01-01T00:00:02.000Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        },
        {
            "timestamp": "2025-01-01T00:00:03.000Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        },
        {
            "timestamp": "2025-01-01T00:00:04.000Z",
            "type": "response_item",
            "payload": {"type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"},
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert [e.role for e in events] == ["user", "assistant"]
    assert [e.content_text for e in events] == ["hi", "hello"]
    assert all(e.session_id == "sid-3" for e in events)
    assert all(e.cwd == "/Users/davidrose/git/agentlog" for e in events)
    assert all(e.git_repo == "git@github.com:org/repo.git" for e in events)
    assert all(e.git_branch == "main" for e in events)
    assert all(e.git_commit == "abc" for e in events)


def test_event_parser_gemini_json_role_mapping(tmp_path: Path) -> None:
    p = tmp_path / "session.json"
    p.write_text(
        json.dumps(
            {
                "sessionId": "sid-2",
                "projectHash": "hash",
                "startTime": "2025-01-01T00:00:00.000Z",
                "lastUpdated": "2025-01-01T00:00:01.000Z",
                "messages": [
                    {"id": "m1", "timestamp": "2025-01-01T00:00:00.000Z", "type": "user", "content": "hello"},
                    {"id": "m2", "timestamp": "2025-01-01T00:00:01.000Z", "type": "gemini", "content": "hi"},
                ],
            }
        ),
        encoding="utf-8",
    )

    events = list(EventParser("gemini").parse_session_file(p))
    assert [e.role for e in events] == ["user", "assistant"]
    assert [e.content_text for e in events] == ["hello", "hi"]
    assert all(e.session_id == "sid-2" for e in events)


def test_event_parser_claude_jsonl_text_and_tool_fields(tmp_path: Path) -> None:
    p = tmp_path / "session.jsonl"
    lines = [
        "not json",
        json.dumps(
            {
                "type": "user",
                "timestamp": "2025-01-01T00:00:00.000Z",
                "sessionId": "sid-4",
                "cwd": "/Users/davidrose/git/agentlog",
                "message": {"role": "user", "content": "hi"},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2025-01-01T00:00:01.000Z",
                "sessionId": "sid-4",
                "cwd": "/Users/davidrose/git/agentlog",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "tool_use", "name": "exec_command", "input": {"cmd": "ls"}},
                        {"type": "tool_result", "content": "ok"},
                    ],
                },
            }
        ),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = list(EventParser("claude").parse_session_file(p))
    assert [e.role for e in events] == ["user", "assistant"]
    assert [e.content_text for e in events] == ["hi", "hello"]
    assert all(e.session_id == "sid-4" for e in events)
    assert all(e.cwd == "/Users/davidrose/git/agentlog" for e in events)
    assert events[1].tool_name == "exec_command"
    assert events[1].tool_input_json == {"cmd": "ls"}
    assert events[1].tool_output_text == "ok"
