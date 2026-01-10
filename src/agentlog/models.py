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

    @property
    def last_activity_fmt(self) -> str:
        """Return formatted last activity timestamp."""
        if self.last_activity:
            return self.last_activity.strftime("%Y-%m-%d %H:%M:%S")
        return "Unknown"


@dataclass(frozen=True)
class AgentEvent:
    """A single event from an agent session."""
    
    timestamp: datetime
    role: str               # 'user', 'assistant', 'system', 'tool'
    event_type: str | None
    content_text: str | None
    raw_json: dict[str, Any]
    tool_name: str | None = None
    tool_input_json: Any | None = None
    tool_output_text: str | None = None
    tool_call_id: str | None = None
    session_id: str | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None

    @property
    def is_tool_call(self) -> bool:
        """Return True if the event is a tool call."""
        return self.event_type in ("tool_call", "tool_use") or self.tool_name is not None or self.tool_call_id is not None

    @property
    def is_tool_output(self) -> bool:
        """Return True if the event is a tool output."""
        return self.role == "tool" or self.event_type in ("tool_output", "tool_result") or self.tool_output_text is not None
