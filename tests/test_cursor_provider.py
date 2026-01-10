"""Tests for Cursor provider.

Uses mocks since we cannot guarantee Cursor is installed or has session data.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentlog.providers.cursor import CursorProvider, BUBBLE_TYPE_USER, BUBBLE_TYPE_ASSISTANT
from agentlog.models import SessionInfo


class TestCursorProviderInit:
    """Tests for CursorProvider initialization."""

    def test_provider_name(self) -> None:
        """Test provider name is 'cursor'."""
        provider = CursorProvider()
        assert provider.name == "cursor"

    def test_watch_patterns_empty(self) -> None:
        """Test watch_patterns is empty (SQLite-based, not file watching)."""
        provider = CursorProvider()
        assert provider.watch_patterns == []

    def test_custom_db_path(self, tmp_path: Path) -> None:
        """Test custom db_path is used."""
        custom_path = tmp_path / "test.vscdb"
        provider = CursorProvider(db_path=custom_path)
        assert provider._db_path == custom_path


class TestCursorProviderMissingDatabase:
    """Tests for handling missing database gracefully."""

    def test_list_sessions_missing_db(self, tmp_path: Path) -> None:
        """Test list_sessions returns empty when database doesn't exist."""
        provider = CursorProvider(db_path=tmp_path / "nonexistent.vscdb")
        sessions = list(provider.list_sessions())
        assert sessions == []

    def test_parse_session_file_missing_db(self, tmp_path: Path) -> None:
        """Test parse_session_file returns None when database doesn't exist."""
        provider = CursorProvider(db_path=tmp_path / "nonexistent.vscdb")
        result = provider.parse_session_file(Path("some-session-id"))
        assert result is None

    def test_count_turns_missing_db(self, tmp_path: Path) -> None:
        """Test count_turns returns 0 when database doesn't exist."""
        provider = CursorProvider(db_path=tmp_path / "nonexistent.vscdb")
        count = provider.count_turns(Path("some-session-id"))
        assert count == 0


class TestCursorProviderWithMockedDatabase:
    """Tests using a mocked SQLite database."""

    @pytest.fixture
    def mock_db(self, tmp_path: Path) -> Path:
        """Create a mock SQLite database with test data."""
        db_path = tmp_path / "state.vscdb"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create the cursorDiskKV table
        cursor.execute("""
            CREATE TABLE cursorDiskKV (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Insert a test session
        session_data = {
            "createdAt": 1704067200000,  # 2024-01-01 00:00:00 UTC
            "fullConversationHeadersOnly": [
                {"bubbleId": "bubble-1", "type": BUBBLE_TYPE_USER},
                {"bubbleId": "bubble-2", "type": BUBBLE_TYPE_ASSISTANT},
                {"bubbleId": "bubble-3", "type": BUBBLE_TYPE_USER},
                {"bubbleId": "bubble-4", "type": BUBBLE_TYPE_ASSISTANT},
            ]
        }
        cursor.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:test-session-123", json.dumps(session_data))
        )
        
        # Insert another session with minimal data
        minimal_session = {
            "createdAt": 1704153600000,  # 2024-01-02 00:00:00 UTC
            "fullConversationHeadersOnly": []
        }
        cursor.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:empty-session", json.dumps(minimal_session))
        )
        
        conn.commit()
        conn.close()
        
        return db_path

    @pytest.fixture
    def provider(self, mock_db: Path) -> CursorProvider:
        """Create provider with mock database."""
        return CursorProvider(db_path=mock_db)

    def test_list_sessions(self, provider: CursorProvider) -> None:
        """Test listing all sessions from mock database."""
        sessions = list(provider.list_sessions())
        assert len(sessions) == 2
        
        # Find the test session
        test_session = next((s for s in sessions if s.session_id == "test-session-123"), None)
        assert test_session is not None
        assert test_session.provider == "cursor"
        assert test_session.turn_count == 4

    def test_parse_session_returns_correct_structure(self, provider: CursorProvider) -> None:
        """Test parse_session_file returns correct SessionInfo structure."""
        session = provider.parse_session_file(Path("test-session-123"))
        
        assert session is not None
        assert isinstance(session, SessionInfo)
        assert session.session_id == "test-session-123"
        assert session.provider == "cursor"
        assert session.turn_count == 4
        assert session.cwd == ""  # Cursor doesn't provide cwd
        assert session.project is None  # Cannot derive without cwd
        assert session.last_activity is not None
        
        # Check timestamp conversion (2024-01-01 00:00:00 UTC)
        expected_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert session.last_activity == expected_time

    def test_parse_session_empty_session(self, provider: CursorProvider) -> None:
        """Test parse_session_file handles empty session correctly."""
        session = provider.parse_session_file(Path("empty-session"))
        
        assert session is not None
        assert session.session_id == "empty-session"
        assert session.turn_count == 0

    def test_parse_session_nonexistent(self, provider: CursorProvider) -> None:
        """Test parse_session_file returns None for nonexistent session."""
        session = provider.parse_session_file(Path("nonexistent-session"))
        assert session is None

    def test_count_turns(self, provider: CursorProvider) -> None:
        """Test count_turns returns correct count."""
        count = provider.count_turns(Path("test-session-123"))
        assert count == 4

    def test_count_turns_empty_session(self, provider: CursorProvider) -> None:
        """Test count_turns returns 0 for empty session."""
        count = provider.count_turns(Path("empty-session"))
        assert count == 0

    def test_count_turns_nonexistent(self, provider: CursorProvider) -> None:
        """Test count_turns returns 0 for nonexistent session."""
        count = provider.count_turns(Path("nonexistent"))
        assert count == 0

    def test_get_all_sessions(self, provider: CursorProvider) -> None:
        """Test get_all_sessions convenience method."""
        sessions = provider.get_all_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 2

    def test_parse_events(self, provider: CursorProvider) -> None:
        """Test parsing events for a session ID."""
        events = list(provider.parse_events(Path("test-session-123")))
        assert len(events) == 4
        assert all(e.session_id == "test-session-123" for e in events)
        assert events[0].role == "user"
        assert events[1].role == "assistant"

    def test_parse_session(self, provider: CursorProvider) -> None:
        """Test parsing events from a SessionInfo object."""
        session = provider.parse_session_file(Path("test-session-123"))
        assert session is not None
        
        events = list(provider.parse_session(session))
        assert len(events) == 4
        assert all(e.session_id == "test-session-123" for e in events)


class TestCursorProviderEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def corrupt_db(self, tmp_path: Path) -> Path:
        """Create a database with corrupt/malformed data."""
        db_path = tmp_path / "corrupt.vscdb"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE cursorDiskKV (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Insert malformed JSON
        cursor.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:malformed", "not valid json{{{")
        )
        
        # Insert valid JSON but missing required fields
        cursor.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:missing-fields", json.dumps({"foo": "bar"}))
        )
        
        # Insert NULL value
        cursor.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:null-value", None)
        )
        
        conn.commit()
        conn.close()
        
        return db_path

    def test_handles_malformed_json(self, corrupt_db: Path) -> None:
        """Test provider handles malformed JSON gracefully."""
        provider = CursorProvider(db_path=corrupt_db)
        sessions = list(provider.list_sessions())
        # Should skip malformed entries
        assert len(sessions) == 0

    def test_handles_missing_created_at(self, corrupt_db: Path) -> None:
        """Test provider handles missing createdAt field."""
        provider = CursorProvider(db_path=corrupt_db)
        session = provider.parse_session_file(Path("missing-fields"))
        # Should return None for session without createdAt
        assert session is None

    def test_handles_null_value(self, corrupt_db: Path) -> None:
        """Test provider handles NULL values in database."""
        provider = CursorProvider(db_path=corrupt_db)
        session = provider.parse_session_file(Path("null-value"))
        assert session is None

    def test_parse_database_path_returns_none(self, tmp_path: Path) -> None:
        """Test parse_session_file returns None when given database path."""
        db_path = tmp_path / "state.vscdb"
        db_path.write_text("")  # Create empty file
        
        provider = CursorProvider(db_path=db_path)
        result = provider.parse_session_file(db_path)
        assert result is None

    def test_count_turns_database_path_returns_zero(self, tmp_path: Path) -> None:
        """Test count_turns returns 0 when given database path."""
        db_path = tmp_path / "state.vscdb"
        db_path.write_text("")
        
        provider = CursorProvider(db_path=db_path)
        count = provider.count_turns(db_path)
        assert count == 0


class TestBubbleCounting:
    """Tests for bubble counting logic."""

    def test_count_user_bubbles(self) -> None:
        """Test counting user bubbles (type=1)."""
        provider = CursorProvider()
        bubbles = [
            {"bubbleId": "1", "type": BUBBLE_TYPE_USER},
            {"bubbleId": "2", "type": BUBBLE_TYPE_USER},
        ]
        assert provider._count_bubbles(bubbles) == 2

    def test_count_assistant_bubbles(self) -> None:
        """Test counting assistant bubbles (type=2)."""
        provider = CursorProvider()
        bubbles = [
            {"bubbleId": "1", "type": BUBBLE_TYPE_ASSISTANT},
            {"bubbleId": "2", "type": BUBBLE_TYPE_ASSISTANT},
        ]
        assert provider._count_bubbles(bubbles) == 2

    def test_count_mixed_bubbles(self) -> None:
        """Test counting mixed user/assistant bubbles."""
        provider = CursorProvider()
        bubbles = [
            {"bubbleId": "1", "type": BUBBLE_TYPE_USER},
            {"bubbleId": "2", "type": BUBBLE_TYPE_ASSISTANT},
            {"bubbleId": "3", "type": BUBBLE_TYPE_USER},
        ]
        assert provider._count_bubbles(bubbles) == 3

    def test_skip_unknown_bubble_types(self) -> None:
        """Test that unknown bubble types are skipped."""
        provider = CursorProvider()
        bubbles = [
            {"bubbleId": "1", "type": BUBBLE_TYPE_USER},
            {"bubbleId": "2", "type": 99},  # Unknown type
            {"bubbleId": "3", "type": 0},   # Unknown type
            {"bubbleId": "4", "type": BUBBLE_TYPE_ASSISTANT},
        ]
        assert provider._count_bubbles(bubbles) == 2

    def test_handle_invalid_bubble_format(self) -> None:
        """Test handling invalid bubble formats."""
        provider = CursorProvider()
        bubbles = [
            {"bubbleId": "1", "type": BUBBLE_TYPE_USER},
            "not a dict",  # Invalid
            None,          # Invalid
            {"bubbleId": "2", "type": BUBBLE_TYPE_ASSISTANT},
        ]
        assert provider._count_bubbles(bubbles) == 2

    def test_handle_non_list_input(self) -> None:
        """Test handling non-list input."""
        provider = CursorProvider()
        assert provider._count_bubbles(None) == 0
        assert provider._count_bubbles("not a list") == 0
        assert provider._count_bubbles({"type": BUBBLE_TYPE_USER}) == 0
