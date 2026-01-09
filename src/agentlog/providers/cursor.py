"""Cursor session provider.

Cursor is a VS Code fork with AI capabilities. Session storage format
is still being investigated.

TODO: Implement once session file format is documented.
"""

from __future__ import annotations

from pathlib import Path

from ..provider import SessionInfo, SessionProvider


class CursorProvider(SessionProvider):
    """Provider for Cursor AI sessions.
    
    Note: This is a placeholder implementation. Cursor's session
    storage format needs to be documented before full implementation.
    """
    
    @property
    def name(self) -> str:
        return "cursor"
    
    @property
    def watch_patterns(self) -> list[str]:
        # Cursor likely stores data in Library/Application Support
        # This needs investigation
        return []
    
    def parse_session_file(self, path: Path) -> SessionInfo | None:
        # Not implemented yet
        return None
    
    def count_turns(self, path: Path) -> int:
        return 0
