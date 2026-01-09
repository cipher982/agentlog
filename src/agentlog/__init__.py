"""agentlog - Unified API for agent session logs.

Provides a consistent interface for reading session data from
Claude Code, Codex, Gemini CLI, and Cursor.

Example:
    >>> from agentlog import SessionRegistry
    >>> registry = SessionRegistry()
    >>> for session in registry.list_sessions():
    ...     print(f"{session.provider}: {session.session_id}")
"""

from .models import SessionInfo, AgentEvent
from .registry import SessionRegistry
from .event_parser import EventParser

__all__ = ["SessionInfo", "AgentEvent", "SessionRegistry", "EventParser"]
__version__ = "0.1.0"
