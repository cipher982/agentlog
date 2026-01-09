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

