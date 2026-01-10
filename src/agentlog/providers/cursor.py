"""Cursor session provider.

Cursor stores sessions in a SQLite database at:
~/Library/Application Support/Cursor/User/globalStorage/state.vscdb

Table: cursorDiskKV
Session data: composerData:<id> keys contain JSON with:
- createdAt: timestamp in milliseconds
- fullConversationHeadersOnly: list of bubble headers

Each bubble has: bubbleId, type (1=user, 2=assistant)
Bubbles don't have individual timestamps - use session createdAt.

Note: cwd is not available in Cursor sessions.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ..providers.base import AgentEvent, SessionInfo, SessionProvider

if TYPE_CHECKING:
    pass


# Bubble type constants
BUBBLE_TYPE_USER = 1
BUBBLE_TYPE_ASSISTANT = 2


class CursorProvider(SessionProvider):
    """Provider for Cursor AI sessions.
    
    Cursor stores data in a SQLite database, not individual session files.
    This provider reads sessions directly from the database using read-only
    connections to avoid locking issues with a running Cursor instance.
    """
    
    # Path to Cursor's SQLite database (macOS only for now)
    DB_PATH = Path("~/Library/Application Support/Cursor/User/globalStorage/state.vscdb").expanduser()
    
    def __init__(self, db_path: Path | None = None):
        """Initialize Cursor provider.
        
        Args:
            db_path: Optional override for the database path. Useful for testing.
        """
        self._db_path = db_path or self.DB_PATH
    
    @property
    def name(self) -> str:
        return "cursor"
    
    @property
    def watch_patterns(self) -> list[str]:
        # No file watching - SQLite based
        # The registry handles this by not iterating patterns
        return []
    
    def list_sessions(self) -> Iterator[SessionInfo]:
        """List all available Cursor sessions.
        
        Yields:
            SessionInfo for each composer session found in the database.
        """
        if not self._db_path.exists():
            return
        
        try:
            # Connect in read-only mode to avoid locking running Cursor
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Get all composer sessions
            cursor.execute("""
                SELECT key, value FROM cursorDiskKV
                WHERE key LIKE 'composerData:%' AND value IS NOT NULL
            """)
            
            for key, value_blob in cursor.fetchall():
                if not value_blob:
                    continue
                
                try:
                    composer_id = key.split(':', 1)[1] if ':' in key else key
                    session = self._parse_session_data(composer_id, value_blob)
                    if session is not None:
                        yield session
                except Exception:
                    # Skip malformed sessions
                    continue
            
            conn.close()
            
        except sqlite3.Error:
            # Database not accessible, return empty
            return
    
    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a session by ID.
        
        For Cursor, 'path' is treated as a session ID string, not a file path.
        This is a workaround since Cursor doesn't use files.
        
        To parse by session_id, pass Path(session_id) and it will be extracted.
        The file_path in the returned SessionInfo will be the database path.
        
        Args:
            path: Path containing the session_id as its name component,
                  or Path to the database with a session_id lookup needed.
                  
        Returns:
            SessionInfo if session found, None otherwise.
        """
        # If path points to the database, we can't parse without a session_id
        if path == self._db_path or str(path).endswith('.vscdb'):
            return None
        
        # Treat path.name as the session_id
        session_id = path.name
        if not session_id:
            return None
        
        return self._get_session_by_id(session_id)
    
    def _get_session_by_id(self, session_id: str) -> SessionInfo | None:
        """Get a specific session by composer ID.
        
        Args:
            session_id: The composer session ID.
            
        Returns:
            SessionInfo if found, None otherwise.
        """
        if not self._db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value FROM cursorDiskKV
                WHERE key = ? AND value IS NOT NULL
            """, (f"composerData:{session_id}",))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row or not row[0]:
                return None
            
            return self._parse_session_data(session_id, row[0])
            
        except sqlite3.Error:
            return None
    
    def _parse_session_data(self, session_id: str, data: str | bytes) -> SessionInfo | None:
        """Parse session data from the database.
        
        Args:
            session_id: The composer session ID.
            data: JSON blob from the database.
            
        Returns:
            SessionInfo if valid, None otherwise.
        """
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            session_data = json.loads(data)
            
            # createdAt is required (timestamp in milliseconds)
            created_at = session_data.get('createdAt')
            if not created_at:
                return None
            
            # Convert ms timestamp to datetime
            try:
                last_activity = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                last_activity = None
            
            # Get bubble list for turn counting
            bubbles = session_data.get('fullConversationHeadersOnly', [])
            turn_count = self._count_bubbles(bubbles)
            
            # Cursor doesn't provide cwd - set to empty string
            cwd = ""
            
            # Extra metadata
            extra = {}
            if bubbles:
                extra['bubble_count'] = len(bubbles)
            
            return SessionInfo(
                session_id=session_id,
                provider=self.name,
                cwd=cwd,
                file_path=self._db_path,  # Use DB path as file_path
                turn_count=turn_count,
                project=None,  # Cannot derive without cwd
                last_activity=last_activity,
                extra=extra,
            )
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    
    def _count_bubbles(self, bubbles: list) -> int:
        """Count conversational bubbles (user and assistant messages).
        
        Args:
            bubbles: List of bubble header dictionaries.
            
        Returns:
            Count of user and assistant messages.
        """
        if not isinstance(bubbles, list):
            return 0
        
        count = 0
        for bubble in bubbles:
            if not isinstance(bubble, dict):
                continue
            bubble_type = bubble.get('type')
            if bubble_type in (BUBBLE_TYPE_USER, BUBBLE_TYPE_ASSISTANT):
                count += 1
        
        return count
    
    def parse_events(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a Cursor session.
        
        For Cursor, path should be Path(session_id) where session_id is the
        composer ID. The actual database path is determined by CursorProvider.
        """
        # Determine session_id from path
        session_id = path.name
        if not session_id:
            return
        
        if not self._db_path.exists():
            return
        
        try:
            # Connect in read-only mode
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value FROM cursorDiskKV
                WHERE key = ? AND value IS NOT NULL
            """, (f"composerData:{session_id}",))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row or not row[0]:
                return
            
            data = row[0]
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            session_data = json.loads(data)
            
            # Get session timestamp
            created_at = session_data.get('createdAt')
            if created_at:
                try:
                    session_timestamp = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
                except (ValueError, OSError):
                    session_timestamp = datetime.now(timezone.utc)
            else:
                session_timestamp = datetime.now(timezone.utc)
            
            # Get bubbles
            bubbles = session_data.get('fullConversationHeadersOnly', [])
            
            for i, bubble in enumerate(bubbles):
                event = self._parse_event({
                    'bubbleId': bubble.get('bubbleId', f'unknown-{i}'),
                    'type': bubble.get('type', 0),
                    'composerId': session_id,
                    'index': i,
                    'timestamp': session_timestamp.isoformat(),
                })
                yield event
                
        except (sqlite3.Error, json.JSONDecodeError):
            return

    def _parse_event(self, raw: Mapping[str, Any]) -> AgentEvent:
        """Parse a single Cursor bubble event."""
        bubble_type = raw.get('type', 0)
        
        # Map bubble type to role
        if bubble_type == 1:
            role = "user"
        elif bubble_type == 2:
            role = "assistant"
        else:
            role = "system"
        
        # Extract timestamp
        timestamp = datetime.now()
        ts_str = raw.get('timestamp')
        if isinstance(ts_str, str) and ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type="bubble",
            content_text=None,  # Content not available in headers
            raw_json=dict(raw),
            session_id=raw.get('composerId') if isinstance(raw.get('composerId'), str) else None,
        )

    def count_turns(self, path: Path) -> int:
        """Count the number of conversational turns in a session.
        
        For Cursor, treats path.name as the session_id.
        
        Args:
            path: Path containing session_id as name, or database path.
            
        Returns:
            Number of turns (user + assistant bubbles).
        """
        # If path is the database, we can't count without a session_id
        if path == self._db_path or str(path).endswith('.vscdb'):
            return 0
        
        session_id = path.name
        if not session_id:
            return 0
        
        session = self._get_session_by_id(session_id)
        if session is None:
            return 0
        
        return session.turn_count
    
    def get_all_sessions(self) -> list[SessionInfo]:
        """Get all sessions from the database.
        
        Convenience method that returns a list instead of iterator.
        
        Returns:
            List of all SessionInfo objects.
        """
        return list(self.list_sessions())
