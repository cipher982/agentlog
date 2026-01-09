"""Gemini CLI session provider.

Gemini stores sessions in ~/.gemini/tmp/<projectHash>/chats/session-*.json where:
- projectHash is SHA256 hash of the absolute cwd
- Each file is a single JSON object (not JSONL) with the entire session
- Structure:
  {
    "sessionId": "uuid",
    "projectHash": "sha256-of-cwd",
    "startTime": "ISO8601",
    "lastUpdated": "ISO8601",
    "messages": [
      {"id": "uuid", "timestamp": "ISO8601", "type": "user|gemini", "content": "..."},
      ...
    ]
  }

Turn counting rules:
- Count: messages with type="user" or type="gemini"
- Note: Gemini uses "gemini" not "assistant" for AI responses
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..provider import SessionInfo, SessionProvider


def _hash_cwd(cwd: str) -> str:
    """Compute SHA256 hash of cwd for Gemini's projectHash.

    Normalizes the path first to handle symlinks and trailing slashes.

    Args:
        cwd: Absolute path to working directory.

    Returns:
        SHA256 hex digest of the normalized cwd string.
    """
    resolved = str(Path(cwd).resolve())
    # Strip trailing slash but preserve root "/"
    normalized = resolved.rstrip("/") or "/"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_turn_message(msg: dict) -> bool:
    """Check if a message object counts as a conversational turn.

    Gemini messages that count as turns:
    - type="user" (human messages)
    - type="gemini" (AI responses - Gemini uses "gemini" not "assistant")

    Args:
        msg: Message dictionary from the messages array.

    Returns:
        True if the message counts as a turn, False otherwise.
    """
    if not isinstance(msg, dict):
        return False

    msg_type = msg.get("type")
    return msg_type in ("user", "gemini")


@dataclass
class TurnCountState:
    """Caching state for efficient turn count updates.

    Unlike Claude/Codex which use append-only JSONL, Gemini rewrites
    the entire JSON file on each update. So we cache based on message
    count rather than byte offset.

    Falls back to full recount if message count doesn't match.
    """

    mtime_ns: int = 0
    size: int = 0
    message_count: int = 0
    turn_count: int = 0


class GeminiProvider(SessionProvider):
    """Session provider for Gemini CLI."""

    def __init__(self, home: Path | None = None):
        """Initialize Gemini provider.

        Args:
            home: Home directory path. Defaults to Path.home().
        """
        self._home = home or Path.home()
        self._turn_cache: dict[Path, TurnCountState] = {}
        self._cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def watch_patterns(self) -> list[str]:
        return [".gemini/tmp/*/chats/session*.json"]

    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a Gemini session JSON file."""
        if not path.exists():
            return None

        # Read and parse the JSON file
        data = self._read_session_data(path)
        if not data:
            return None

        # Extract session ID
        session_id = data.get("sessionId")
        if not session_id:
            return None

        # Extract projectHash (cannot reverse to cwd, but store it)
        project_hash = data.get("projectHash", "")

        # Try to derive cwd from path - Gemini encodes hash in path
        cwd = self.derive_cwd_from_path(path)
        # If we can't derive it (hash is not reversible), use empty string
        # The cwd would need to be looked up via find_session_for_cwd
        if not cwd:
            cwd = ""

        # Get turn count
        turn_count = self.count_turns(path)

        # Get last activity from lastUpdated field or file mtime
        last_activity = None
        last_updated = data.get("lastUpdated")
        if last_updated:
            try:
                last_activity = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        if not last_activity:
            try:
                mtime = path.stat().st_mtime
                last_activity = datetime.fromtimestamp(mtime)
            except OSError:
                pass

        # Extract project from cwd if we have it
        project = self.derive_project_from_cwd(cwd) if cwd else None

        # Extra metadata
        extra = {"project_hash": project_hash}
        start_time = data.get("startTime")
        if start_time:
            extra["start_time"] = start_time

        return SessionInfo(
            session_id=session_id,
            provider=self.name,
            cwd=cwd,
            file_path=path,
            turn_count=turn_count,
            project=project,
            last_activity=last_activity,
            extra=extra,
        )

    def count_turns(self, path: Path) -> int:
        """Count conversational turns with caching."""
        with self._cache_lock:
            cached = self._turn_cache.get(path)
            if cached:
                updated = self._update_turn_count(path, cached)
                self._turn_cache[path] = updated
                return updated.turn_count

            # Full count for new file
            state = self._count_turns_full(path)
            self._turn_cache[path] = state
            return state.turn_count

    def derive_cwd_from_path(self, path: Path) -> str | None:
        """Extract projectHash from path but cannot reverse to cwd.

        Gemini path format: ~/.gemini/tmp/<projectHash>/chats/session-*.json

        The projectHash is a SHA256 hash of the cwd, which cannot be reversed.
        This method extracts the hash but returns None since we can't derive cwd.
        Use find_session_for_cwd() to find sessions for a known cwd.

        Args:
            path: Absolute path to the session file.

        Returns:
            Always None - hash cannot be reversed to cwd.
        """
        # Path is like: ~/.gemini/tmp/<hash>/chats/session-*.json
        # We can extract the hash but cannot reverse it
        return None

    def find_session_for_cwd(self, cwd: str) -> SessionInfo | None:
        """Find the most recent session for a given working directory.

        Computes the SHA256 hash of the cwd and looks for a matching
        projectHash directory in ~/.gemini/tmp/.

        Args:
            cwd: Absolute path to the working directory.

        Returns:
            SessionInfo for the most recent matching session, or None.
        """
        if not cwd:
            return None

        # Hash the cwd to find the projectHash directory
        project_hash = _hash_cwd(cwd)
        project_dir = self._home / ".gemini" / "tmp" / project_hash / "chats"

        if not project_dir.exists():
            return None

        # Find most recently modified session file
        try:
            candidates = list(project_dir.glob("session*.json"))
        except OSError:
            return None

        if not candidates:
            return None

        try:
            latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        except OSError:
            return None

        # Parse and return with cwd populated (since we know it)
        info = self.parse_session_file(latest)
        if info and not info.cwd:
            # Update with known cwd
            return SessionInfo(
                session_id=info.session_id,
                provider=info.provider,
                cwd=cwd,
                file_path=info.file_path,
                turn_count=info.turn_count,
                project=self.derive_project_from_cwd(cwd),
                last_activity=info.last_activity,
                extra=info.extra,
            )
        return info

    def get_project_hash_from_path(self, path: Path) -> str | None:
        """Extract projectHash from session file path.

        Args:
            path: Absolute path to a session file.

        Returns:
            The projectHash string, or None if not extractable.
        """
        # Path format: ~/.gemini/tmp/<projectHash>/chats/session-*.json
        # parent = chats, parent.parent = <projectHash>, parent.parent.parent = tmp
        try:
            if path.parent.name == "chats" and path.parent.parent.parent.name == "tmp":
                return path.parent.parent.name
        except (AttributeError, IndexError):
            pass
        return None

    def _read_session_data(self, path: Path) -> dict | None:
        """Read and parse a Gemini session JSON file.

        Args:
            path: Path to the session file.

        Returns:
            Parsed JSON dict, or None if unreadable/invalid.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _count_turns_full(self, path: Path) -> TurnCountState:
        """Full turn count by parsing the entire JSON file."""
        try:
            stat = path.stat()
        except OSError:
            return TurnCountState()

        data = self._read_session_data(path)
        if not data:
            return TurnCountState()

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return TurnCountState()

        turns = sum(1 for msg in messages if _is_turn_message(msg))

        return TurnCountState(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            message_count=len(messages),
            turn_count=turns,
        )

    def _update_turn_count(self, path: Path, state: TurnCountState) -> TurnCountState:
        """Check if cached state is still valid, otherwise recount.

        Unlike JSONL providers that can do incremental reads, Gemini
        rewrites the entire file. We recount if mtime or size changed.
        """
        try:
            stat = path.stat()
        except OSError:
            return state

        # If unchanged, return cached state
        if stat.st_size == state.size and stat.st_mtime_ns == state.mtime_ns:
            return state

        # File changed - do full recount
        return self._count_turns_full(path)
