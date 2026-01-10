"""Claude Code session provider.

Claude stores sessions in ~/.claude/projects/<slug>/*.jsonl where:
- slug is the cwd with '/' replaced by '-' (e.g., /Users/david/git/watchman → -Users-david-git-watchman)
- Each JSONL line is an event with type, sessionId, timestamp, etc.

Turn counting rules:
- Count: type="user" (non-meta) and type="assistant"
- Skip: toolUseResult events (tool results have type="user")
- Skip: isSidechain events (internal subtask work)
- Skip: isMeta events (system messages)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..providers.base import JSONLProvider, SessionInfo, AgentEvent


def _claude_project_slug(cwd: str) -> str:
    """Convert cwd to Claude's project slug format.

    Claude encodes paths by replacing '/' with '-'.
    Example: /Users/david/git/watchman → -Users-david-git-watchman
    """
    cwd = (cwd or "").strip()
    if not cwd:
        return ""
    if cwd.startswith("/"):
        return cwd.replace("/", "-")
    return "-" + cwd.replace("/", "-")


def _slug_to_cwd(slug: str) -> str:
    """Reverse Claude's project slug to original cwd.

    Example: -Users-david-git-watchman → /Users/david/git/watchman
    """
    if not slug:
        return ""
    # Remove leading dash and convert remaining dashes back to slashes
    if slug.startswith("-"):
        slug = slug[1:]
    return "/" + slug.replace("-", "/")


def _is_turn_event(obj: dict) -> bool:
    """Check if a JSONL event counts as a conversational turn."""
    if not isinstance(obj, dict):
        return False

    # Skip tool results - they have type="user" but aren't conversational turns
    if "toolUseResult" in obj:
        return False

    # Skip sidechain/subtask events - internal agent work, not user conversation
    if obj.get("isSidechain"):
        return False

    event_type = obj.get("type")

    if event_type == "user":
        # Skip meta messages (system prompts, etc.)
        return not bool(obj.get("isMeta"))

    if event_type == "assistant":
        return True

    # Handle older format with nested message
    message = obj.get("message")
    if isinstance(message, dict):
        role = message.get("role")
        if role in {"user", "assistant"}:
            return True

    return False
    
class ClaudeProvider(JSONLProvider):
    """Session provider for Claude Code CLI."""

    def __init__(self, home: Path | None = None):
        """Initialize Claude provider.

        Args:
            home: Home directory path. Defaults to Path.home().
        """
        super().__init__()
        self._home = home or Path.home()

    @property
    def name(self) -> str:
        return "claude"

    def is_turn_event(self, obj: dict) -> bool:
        """Check if a JSONL event counts as a conversational turn."""
        return _is_turn_event(obj)

    @property
    def watch_patterns(self) -> list[str]:
        return [".claude/projects/*/*.jsonl"]

    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a Claude session JSONL file."""
        if not path.exists():
            return None

        # Extract session ID from file content
        session_id = self._read_session_id(path)
        if not session_id:
            return None

        # Read cwd from file content (more reliable than slug which breaks on hyphens)
        cwd = self._read_cwd(path)
        if not cwd:
            # Fallback to slug derivation for older files without cwd field
            cwd = self.derive_cwd_from_path(path)
        if not cwd:
            return None

        # Get turn count
        turn_count = self.count_turns(path)

        # Get last modified time as last activity
        try:
            mtime = path.stat().st_mtime
            last_activity = datetime.fromtimestamp(mtime)
        except OSError:
            last_activity = None

        # Extract project from cwd
        project = self.derive_project_from_cwd(cwd)

        # Try to get git branch from file content
        extra = {}
        git_branch = self._read_git_branch(path)
        if git_branch:
            extra["git_branch"] = git_branch

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

    def parse_events(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a Claude session file."""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    yield self._parse_event(raw)
                except (json.JSONDecodeError, KeyError):
                    continue

    def _parse_event(self, raw: dict[str, Any]) -> AgentEvent:
        """Parse a single Claude event."""
        event_type = raw.get("type")
        role = event_type if event_type in ("user", "assistant", "system") else "system"

        # Extract timestamp
        ts_str = raw.get("timestamp")
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Extract content
        content_text = None
        message = raw.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", [])
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content_text = "\n".join(text_parts) if text_parts else None
            elif isinstance(content, str):
                content_text = content

        # Extract tool info
        tool_name = None
        tool_input = None
        tool_output = None

        if isinstance(message, dict):
            content = message.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "tool_use":
                            tool_name = part.get("name")
                            tool_input = part.get("input")
                        elif part.get("type") == "tool_result":
                            tool_output = part.get("content")

        tool_output_text = None
        if tool_output is not None:
            tool_output_text = (
                tool_output
                if isinstance(tool_output, str)
                else json.dumps(tool_output, ensure_ascii=False)
            )

        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=event_type,
            content_text=content_text,
            raw_json=dict(raw),
            tool_name=tool_name,
            tool_input_json=tool_input,
            tool_output_text=tool_output_text,
            session_id=raw.get("sessionId")
            if isinstance(raw.get("sessionId"), str)
            else None,
            cwd=raw.get("cwd") if isinstance(raw.get("cwd"), str) else None,
        )

    def derive_cwd_from_path(self, path: Path) -> str | None:
        """Extract cwd from Claude's project directory slug."""
        # Path format: ~/.claude/projects/<slug>/<session>.jsonl
        # Parent is the slug directory
        slug = path.parent.name
        if not slug or slug == "projects":
            return None
        return _slug_to_cwd(slug)

    def find_session_for_cwd(self, cwd: str) -> SessionInfo | None:
        """Find the most recent session for a given working directory."""
        slug = _claude_project_slug(cwd)
        if not slug:
            return None

        project_dir = self._home / ".claude" / "projects" / slug
        if not project_dir.exists():
            return None

        # Find most recently modified JSONL
        try:
            candidates = list(project_dir.glob("*.jsonl"))
        except OSError:
            return None

        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        return self.parse_session_file(latest)

    def _read_session_id(self, path: Path, max_lines: int = 200) -> str | None:
        """Extract sessionId from the beginning of a JSONL file."""
        try:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    session_id = obj.get("sessionId")
                    if isinstance(session_id, str) and session_id:
                        return session_id
        except OSError:
            pass
        return None

    def _read_cwd(self, path: Path, max_lines: int = 50) -> str | None:
        """Extract cwd from the beginning of a JSONL file.

        Claude stores the working directory in the file content, which is more
        reliable than deriving it from the slug (which breaks on hyphenated paths).
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cwd = obj.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
        except OSError:
            pass
        return None

    def _read_git_branch(self, path: Path, max_lines: int = 50) -> str | None:
        """Extract git branch from the beginning of a JSONL file."""
        try:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    git_branch = obj.get("gitBranch")
                    if isinstance(git_branch, str) and git_branch:
                        return git_branch
        except OSError:
            pass
        return None
                    
