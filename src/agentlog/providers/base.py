"""Base class for session providers."""

from __future__ import annotations

import json
import threading
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from ..models import AgentEvent, SessionInfo


@dataclass
class TurnCountState:
    """Incremental turn counting state for efficient updates.

    Tracks file position to only parse new content on updates.
    Falls back to full recount if file shrinks (truncated/replaced).
    """

    pos: int = 0
    size: int = 0
    mtime_ns: int = 0
    remainder: bytes = b""
    turn_count: int = 0


class SessionProvider(ABC):
    """Abstract base class for agent session providers.
    
    Each provider knows how to discover and parse session files
    for a specific agent tool (Claude, Codex, Gemini, Cursor).
    """
    
    @property
    def name(self) -> str:
        """Provider name (e.g., 'claude', 'codex', 'gemini')."""
        return "base"
    
    @property
    def watch_patterns(self) -> list[str]:
        """Glob patterns for watching session files.
        
        Used by file watchers to detect new/changed sessions.
        Example: ['~/.claude/projects/**/*.jsonl']
        """
        return []
    
    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a session file and return session info.
        
        Args:
            path: Path to the session file
            
        Returns:
            SessionInfo if the file is a valid session, None otherwise
        """
        raise NotImplementedError
    
    def parse_events(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a session file.
        
        Args:
            path: Path to the session file
            
        Yields:
            AgentEvent for each event in the file
        """
        raise NotImplementedError
    
    def parse_session(self, session: SessionInfo) -> Iterator[AgentEvent]:
        """Parse all events for a given session.
        
        Defaults to calling parse_events(session.file_path).
        Providers like Cursor may override this to use session_id.
        
        Args:
            session: The session info to parse events for
            
        Yields:
            AgentEvent for each event in the session
        """
        return self.parse_events(session.file_path)
    
    def count_turns(self, path: Path) -> int:
        """Count the number of conversational turns in a session.
        
        A turn typically includes a user message and assistant response.
        Tool use/results are not counted as separate turns.
        
        Args:
            path: Path to the session file
            
        Returns:
            Number of turns in the session
        """
        raise NotImplementedError
    
    def derive_cwd_from_path(self, path: Path) -> str | None:
        """Derive the working directory from the session file path.
        
        Some providers encode the cwd in the file path or directory structure.
        
        Args:
            path: Path to the session file
            
        Returns:
            The working directory, or None if it cannot be derived
        """
        return None
    
    def find_session_for_cwd(self, cwd: str) -> SessionInfo | None:
        """Find the most recent session for a given working directory.

        Args:
            cwd: Working directory to find session for
            
        Returns:
            SessionInfo for the most recent session, or None if not found
        """
        return None

    def derive_project_from_cwd(self, cwd: str) -> str | None:
        """Derive a human-friendly project name from a working directory.

        Attempts to find the git root by looking for a .git directory.
        If found, uses the name of the root directory.
        Otherwise, falls back to the basename of the directory.
        """
        cwd = (cwd or "").strip()
        if not cwd:
            return None

        try:
            resolved = Path(cwd).resolve()
        except OSError:
            resolved = Path(cwd)

        # Walk up the tree to find a .git directory
        current = resolved
        while current.parent != current:
            if (current / ".git").exists():
                return current.name
            current = current.parent

        # Fallback to the basename of the directory
        name = resolved.name.strip()
        return name or None


class JSONLProvider(SessionProvider):
    """Base class for providers that use JSONL session files."""

    def __init__(self):
        self._turn_cache: dict[Path, TurnCountState] = {}
        self._cache_lock = threading.Lock()

    def is_turn_event(self, obj: dict[str, Any]) -> bool:
        """Check if a JSONL event counts as a conversational turn."""
        raise NotImplementedError

    def count_turns(self, path: Path) -> int:
        """Count conversational turns with incremental caching."""
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

    def _count_turns_full(self, path: Path) -> TurnCountState:
        """Full turn count by streaming the entire file."""
        try:
            stat = path.stat()
        except OSError:
            return TurnCountState()

        turns = 0
        remainder = b""
        pos = 0

        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    pos += len(chunk)
                    buf = remainder + chunk
                    parts = buf.split(b"\n")
                    remainder = b""
                    if buf and not buf.endswith(b"\n"):
                        remainder = parts.pop() if parts else buf

                    for raw in parts:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw.decode("utf-8"))
                        except Exception:
                            continue
                        if self.is_turn_event(obj):
                            turns += 1
        except OSError:
            return TurnCountState()

        return TurnCountState(
            pos=stat.st_size,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            remainder=remainder,
            turn_count=turns,
        )

    def _update_turn_count(self, path: Path, state: TurnCountState) -> TurnCountState:
        """Incrementally update turn count from newly appended bytes."""
        try:
            stat = path.stat()
        except OSError:
            return state

        # If file shrank or our position is past EOF, do full recount
        if stat.st_size < state.size or state.pos > stat.st_size:
            return self._count_turns_full(path)

        # If unchanged, return cached state
        if stat.st_size == state.size and stat.st_mtime_ns == state.mtime_ns:
            return state

        # Read only the new bytes
        try:
            with path.open("rb") as f:
                f.seek(state.pos)
                data = f.read()
        except OSError:
            return self._count_turns_full(path)

        buf = state.remainder + (data or b"")
        lines = buf.split(b"\n")
        remainder = b""
        if buf and not buf.endswith(b"\n"):
            remainder = lines.pop() if lines else buf

        turns = state.turn_count
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if self.is_turn_event(obj):
                turns += 1

        return TurnCountState(
            pos=stat.st_size,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            remainder=remainder,
            turn_count=turns,
        )
