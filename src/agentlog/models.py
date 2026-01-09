"""Data models for agentlog."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SessionInfo:
    """Information about an agent session."""
    
    session_id: str
    provider: str           # 'claude', 'codex', 'gemini', 'cursor'
    cwd: str
    file_path: Path
    turn_count: int = 0
    project: str | None = None
    last_activity: datetime | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvent:
    """A single event from an agent session."""
    
    timestamp: datetime
    role: str               # 'user', 'assistant', 'system'
    event_type: str | None
    content_text: str | None
    raw_json: dict[str, Any]
    tool_name: str | None = None
    tool_input_json: dict | str | None = None
    tool_output_text: str | None = None
    session_id: str | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None
