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
        {"type": "function_call", "id": "c1", "name": "tool", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "message", "id": None, "role": "user", "content": [{"type": "input_text", "text": "bye"}]},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    # Now includes function_call and function_call_output events
    assert len(events) == 5
    assert [e.role for e in events] == ["user", "assistant", "assistant", "tool", "user"]
    assert events[0].content_text == "hi"
    assert events[1].content_text == "hello"
    assert events[2].event_type == "tool_call"
    assert events[2].tool_name == "tool"
    assert events[3].event_type == "tool_output"
    assert events[3].tool_output_text == "ok"
    assert events[4].content_text == "bye"
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
    # Now includes function_call event
    assert len(events) == 3
    assert [e.role for e in events] == ["user", "assistant", "assistant"]
    assert events[0].content_text == "hi"
    assert events[1].content_text == "hello"
    assert events[2].event_type == "tool_call"
    assert events[2].tool_name == "exec_command"
    assert all(e.session_id == "sid-3" for e in events)
    assert all(e.cwd == "/Users/davidrose/git/agentlog" for e in events)
    assert all(e.git_repo == "git@github.com:org/repo.git" for e in events)
    assert all(e.git_branch == "main" for e in events)
    assert all(e.git_commit == "abc" for e in events)


def test_event_parser_codex_function_call_parsing(tmp_path: Path) -> None:
    """Test parsing of function_call events with various argument formats."""
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "timestamp": "2025-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "sid-fc", "cwd": "/test"},
        },
        {
            "timestamp": "2025-01-01T00:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":["bash","-lc","ls -la"],"workdir":"/test"}',
                "call_id": "call_abc123",
            },
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert len(events) == 1
    event = events[0]
    assert event.role == "assistant"
    assert event.event_type == "tool_call"
    assert event.tool_name == "shell"
    assert event.tool_input_json == {"command": ["bash", "-lc", "ls -la"], "workdir": "/test"}
    assert event.session_id == "sid-fc"


def test_event_parser_codex_function_call_output_parsing(tmp_path: Path) -> None:
    """Test parsing of function_call_output events."""
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "timestamp": "2025-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "sid-fco", "cwd": "/test"},
        },
        {
            "timestamp": "2025-01-01T00:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_abc123",
                "output": '{"output":"total 42\\ndrwxr-xr-x  5 user staff 160 Jan  1 00:00 .","metadata":{"exit_code":0}}',
            },
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert len(events) == 1
    event = events[0]
    assert event.role == "tool"
    assert event.event_type == "tool_output"
    assert event.tool_call_id == "call_abc123"  # call_id stored for linking
    assert event.tool_output_text == "total 42\ndrwxr-xr-x  5 user staff 160 Jan  1 00:00 ."
    assert event.session_id == "sid-fco"


def test_event_parser_codex_legacy_function_call(tmp_path: Path) -> None:
    """Test parsing of legacy top-level function_call events."""
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "id": "sid-legacy",
            "timestamp": "2025-01-01T00:00:00.000Z",
        },
        {
            "timestamp": "2025-01-01T00:00:01.000Z",
            "type": "function_call",
            "name": "read_file",
            "arguments": {"path": "/etc/hosts"},
        },
        {
            "timestamp": "2025-01-01T00:00:02.000Z",
            "type": "function_call_output",
            "call_id": "call_xyz",
            "output": "127.0.0.1 localhost",
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert len(events) == 2
    
    # First event: function_call
    assert events[0].role == "assistant"
    assert events[0].event_type == "tool_call"
    assert events[0].tool_name == "read_file"
    assert events[0].tool_input_json == {"path": "/etc/hosts"}
    
    # Second event: function_call_output
    assert events[1].role == "tool"
    assert events[1].event_type == "tool_output"
    assert events[1].tool_output_text == "127.0.0.1 localhost"


def test_event_parser_codex_tool_fields_populated(tmp_path: Path) -> None:
    """Verify tool fields are correctly populated for Codex events."""
    p = tmp_path / "session.jsonl"
    lines = [
        {
            "timestamp": "2025-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "sid-verify", "cwd": "/verify"},
        },
        {
            "timestamp": "2025-01-01T00:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "write_file",
                "arguments": '{"path":"/tmp/test.txt","content":"hello world"}',
                "call_id": "call_write",
            },
        },
        {
            "timestamp": "2025-01-01T00:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_write",
                "output": "File written successfully",
            },
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = list(EventParser("codex").parse_session_file(p))
    assert len(events) == 2
    
    # Verify function_call event fields
    fc_event = events[0]
    assert fc_event.tool_name == "write_file"
    assert fc_event.tool_input_json == {"path":"/tmp/test.txt","content":"hello world"}
    assert fc_event.tool_output_text is None
    assert fc_event.content_text is None
    assert fc_event.tool_call_id == "call_write"
    
    # Verify function_call_output event fields
    fco_event = events[1]
    assert fco_event.tool_call_id == "call_write"  # call_id for linking
    assert fco_event.tool_input_json is None
    assert fco_event.tool_output_text == "File written successfully"
    assert fco_event.content_text is None


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
