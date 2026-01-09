"""Base class for session providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import SessionInfo

# Organizational directories under /git/ that are not projects themselves
# These are category directories that contain actual project repos
_GIT_CATEGORY_DIRS = frozenset({"me", "zeta"})

# Secondary category directories under /git/me/
_GIT_ME_SUBDIRS = frozenset({"mytech", "myhealth", "myfinance", "mypersonal"})


class SessionProvider(ABC):
    """Abstract base class for agent session providers.
    
    Each provider knows how to discover and parse session files
    for a specific agent tool (Claude, Codex, Gemini, Cursor).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'claude', 'codex', 'gemini')."""
        ...
    
    @property
    @abstractmethod
    def watch_patterns(self) -> list[str]:
        """Glob patterns for watching session files.
        
        Used by file watchers to detect new/changed sessions.
        Example: ['~/.claude/projects/**/*.jsonl']
        """
        ...
    
    @abstractmethod
    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a session file and return session info.
        
        Args:
            path: Path to the session file
            
        Returns:
            SessionInfo if the file is a valid session, None otherwise
        """
        ...
    
    @abstractmethod
    def count_turns(self, path: Path) -> int:
        """Count the number of conversational turns in a session.
        
        A turn typically includes a user message and assistant response.
        Tool use/results are not counted as separate turns.
        
        Args:
            path: Path to the session file
            
        Returns:
            Number of turns in the session
        """
        ...
    
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

        Uses /git/<project>/ anchoring to find the repo root:
        - /Users/davidrose/git/agentlog/ -> "agentlog"
        - /Users/davidrose/git/me/mytech/life-hub/src -> "life-hub"
        - /Users/davidrose/git/zerg/workers/ -> "zerg"
        
        Falls back to basename if no /git/ pattern found.
        """
        cwd = (cwd or "").strip()
        if not cwd:
            return None

        try:
            resolved = Path(cwd).resolve()
        except OSError:
            resolved = Path(cwd)

        # Try to find /git/ in the path and extract project name
        parts = resolved.parts
        try:
            git_idx = parts.index("git")
        except ValueError:
            # No /git/ found, fall back to basename
            name = resolved.name.strip()
            return name or None

        # Walk forward from /git/ to find the project root
        # Skip organizational directories
        project_name = None
        for i in range(git_idx + 1, len(parts)):
            part = parts[i]
            # Skip top-level category dirs (me, zeta)
            if part in _GIT_CATEGORY_DIRS:
                continue
            # Skip secondary category dirs under /git/me/ (mytech, myhealth, etc.)
            if part in _GIT_ME_SUBDIRS:
                continue
            # First non-category directory is the project
            project_name = part
            break

        if project_name:
            return project_name.strip() or None

        # Edge case: path ends at a category dir (e.g., /git/me/)
        # Fall back to basename
        name = resolved.name.strip()
        return name or None
