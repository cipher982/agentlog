"""Comprehensive tests for all provider implementations.

Tests each provider with real session fixtures to ensure:
- Fixture files can be parsed
- Turn counting returns > 0
- SessionInfo has required fields
- Empty/malformed files don't crash
"""

from pathlib import Path

import pytest

from agentlog.providers.claude import ClaudeProvider
from agentlog.providers.codex import CodexProvider
from agentlog.providers.gemini import GeminiProvider
from agentlog.models import SessionInfo


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestClaudeProvider:
    """Tests for Claude provider with real session fixtures."""

    @pytest.fixture
    def provider(self) -> ClaudeProvider:
        return ClaudeProvider()

    @pytest.fixture
    def fixture_path(self) -> Path:
        return FIXTURES_DIR / "claude_session.jsonl"

    def test_fixture_exists(self, fixture_path: Path) -> None:
        """Verify fixture file exists."""
        assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

    def test_parse_session_file(self, provider: ClaudeProvider, fixture_path: Path) -> None:
        """Test parsing the fixture returns a valid SessionInfo."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None, "parse_session_file returned None"
        assert isinstance(session, SessionInfo)

    def test_session_has_required_fields(self, provider: ClaudeProvider, fixture_path: Path) -> None:
        """Test SessionInfo has all required fields populated."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None

        assert session.session_id, "session_id should be non-empty"
        assert session.provider == "claude", f"provider should be 'claude', got {session.provider}"
        # cwd may be empty for some sessions, but file_path must be set
        assert session.file_path == fixture_path, "file_path should match input"

    def test_turn_count_positive(self, provider: ClaudeProvider, fixture_path: Path) -> None:
        """Test turn count returns > 0 for a real session."""
        count = provider.count_turns(fixture_path)
        assert count > 0, f"Turn count should be > 0, got {count}"

    def test_parse_empty_file(self, provider: ClaudeProvider, tmp_path: Path) -> None:
        """Test empty file doesn't crash, returns None or 0 turns."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        session = provider.parse_session_file(empty_file)
        # Empty file should return None (no session metadata)
        assert session is None

        count = provider.count_turns(empty_file)
        assert count == 0

    def test_parse_malformed_file(self, provider: ClaudeProvider, tmp_path: Path) -> None:
        """Test malformed JSON doesn't crash."""
        malformed_file = tmp_path / "malformed.jsonl"
        malformed_file.write_text("not valid json\n{incomplete\n")

        session = provider.parse_session_file(malformed_file)
        assert session is None

        count = provider.count_turns(malformed_file)
        assert count == 0


class TestCodexProvider:
    """Tests for Codex provider with real session fixtures."""

    @pytest.fixture
    def provider(self) -> CodexProvider:
        return CodexProvider()

    @pytest.fixture
    def fixture_path(self) -> Path:
        return FIXTURES_DIR / "codex_session.jsonl"

    def test_fixture_exists(self, fixture_path: Path) -> None:
        """Verify fixture file exists."""
        assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

    def test_parse_session_file(self, provider: CodexProvider, fixture_path: Path) -> None:
        """Test parsing the fixture returns a valid SessionInfo."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None, "parse_session_file returned None"
        assert isinstance(session, SessionInfo)

    def test_session_has_required_fields(self, provider: CodexProvider, fixture_path: Path) -> None:
        """Test SessionInfo has all required fields populated."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None

        assert session.session_id, "session_id should be non-empty"
        assert session.provider == "codex", f"provider should be 'codex', got {session.provider}"
        assert session.file_path == fixture_path, "file_path should match input"

    def test_turn_count_positive(self, provider: CodexProvider, fixture_path: Path) -> None:
        """Test turn count returns > 0 for a real session."""
        count = provider.count_turns(fixture_path)
        assert count > 0, f"Turn count should be > 0, got {count}"

    def test_parse_empty_file(self, provider: CodexProvider, tmp_path: Path) -> None:
        """Test empty file doesn't crash, returns None or 0 turns."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        session = provider.parse_session_file(empty_file)
        assert session is None

        count = provider.count_turns(empty_file)
        assert count == 0

    def test_parse_malformed_file(self, provider: CodexProvider, tmp_path: Path) -> None:
        """Test malformed JSON doesn't crash."""
        malformed_file = tmp_path / "malformed.jsonl"
        malformed_file.write_text("not valid json\n{incomplete\n")

        session = provider.parse_session_file(malformed_file)
        assert session is None

        count = provider.count_turns(malformed_file)
        assert count == 0


class TestGeminiProvider:
    """Tests for Gemini provider with real session fixtures."""

    @pytest.fixture
    def provider(self) -> GeminiProvider:
        return GeminiProvider()

    @pytest.fixture
    def fixture_path(self) -> Path:
        return FIXTURES_DIR / "gemini_session.json"

    def test_fixture_exists(self, fixture_path: Path) -> None:
        """Verify fixture file exists."""
        assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

    def test_parse_session_file(self, provider: GeminiProvider, fixture_path: Path) -> None:
        """Test parsing the fixture returns a valid SessionInfo."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None, "parse_session_file returned None"
        assert isinstance(session, SessionInfo)

    def test_session_has_required_fields(self, provider: GeminiProvider, fixture_path: Path) -> None:
        """Test SessionInfo has all required fields populated."""
        session = provider.parse_session_file(fixture_path)
        assert session is not None

        assert session.session_id, "session_id should be non-empty"
        assert session.provider == "gemini", f"provider should be 'gemini', got {session.provider}"
        assert session.file_path == fixture_path, "file_path should match input"

    def test_turn_count_positive(self, provider: GeminiProvider, fixture_path: Path) -> None:
        """Test turn count returns > 0 for a real session."""
        count = provider.count_turns(fixture_path)
        assert count > 0, f"Turn count should be > 0, got {count}"

    def test_parse_empty_file(self, provider: GeminiProvider, tmp_path: Path) -> None:
        """Test empty file doesn't crash, returns None or 0 turns."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        session = provider.parse_session_file(empty_file)
        assert session is None

        count = provider.count_turns(empty_file)
        assert count == 0

    def test_parse_malformed_file(self, provider: GeminiProvider, tmp_path: Path) -> None:
        """Test malformed JSON doesn't crash."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{not valid json")

        session = provider.parse_session_file(malformed_file)
        assert session is None

        count = provider.count_turns(malformed_file)
        assert count == 0

    def test_parse_valid_but_empty_session(self, provider: GeminiProvider, tmp_path: Path) -> None:
        """Test valid JSON with empty messages array."""
        empty_session = tmp_path / "empty_messages.json"
        empty_session.write_text('{"sessionId": "test-123", "messages": []}')

        session = provider.parse_session_file(empty_session)
        assert session is not None
        assert session.session_id == "test-123"

        count = provider.count_turns(empty_session)
        assert count == 0


class TestDeriveProjectFromCwd:
    """Tests for derive_project_from_cwd method."""

    @pytest.fixture
    def provider(self) -> ClaudeProvider:
        # Use any provider - they all inherit from SessionProvider
        return ClaudeProvider()

    def test_direct_git_project(self, provider: ClaudeProvider, tmp_path: Path) -> None:
        """Test finding project root via .git directory."""
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        
        subdir = project_root / "src" / "deep"
        subdir.mkdir(parents=True)
        
        result = provider.derive_project_from_cwd(str(subdir))
        assert result == "my-project"

    def test_fallback_to_basename(self, provider: ClaudeProvider, tmp_path: Path) -> None:
        """Test fallback to basename when no .git found."""
        random_path = tmp_path / "foo" / "bar" / "baz"
        random_path.mkdir(parents=True)
        
        result = provider.derive_project_from_cwd(str(random_path))
        assert result == "baz"

    def test_empty_cwd(self, provider: ClaudeProvider) -> None:
        """Test empty cwd returns None."""
        result = provider.derive_project_from_cwd("")
        assert result is None

    def test_none_cwd(self, provider: ClaudeProvider) -> None:
        """Test None cwd returns None."""
        result = provider.derive_project_from_cwd(None)
        assert result is None

    def test_whitespace_only_cwd(self, provider: ClaudeProvider) -> None:
        """Test whitespace-only cwd returns None."""
        result = provider.derive_project_from_cwd("   ")
        assert result is None

    def test_root_path(self, provider: ClaudeProvider) -> None:
        """Test root path returns empty string or None."""
        result = provider.derive_project_from_cwd("/")
        # Root has no name component
        assert result is None or result == ""
