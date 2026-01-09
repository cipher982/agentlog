"""Codex CLI session provider.

Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/*.jsonl where:
- Each file is JSONL (JSON Lines) format.
- First line: {"type":"session_meta","payload":{"id":"uuid","cwd":"/path",...}}
- Subsequent lines: {"type":"response_item","payload":{"type":"message","role":"user|assistant",...}}

Turn counting rules:
- Count: type="response_item" with payload.type="message" and payload.role in ("user", "assistant")
- Skip: type="session_meta" (metadata, not conversation)
- Skip: type="response_item" with payload.type in ("function_call", "function_call_output", "reasoning")
- Skip: type="turn_context", "event_msg" (metadata events)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..provider import SessionInfo, SessionProvider


def _is_turn_event(obj: dict) -> bool:
    """Check if a JSONL event counts as a conversational turn.

    Codex events that count as turns:
    - type="response_item" with payload.type="message" and payload.role in ("user", "assistant")
    """
    if not isinstance(obj, dict):
        return False

    # Must be a response_item
    if obj.get("type") != "response_item":
        return False

    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return False

    # Must be a message type
    if payload.get("type") != "message":
        return False

    # Must have user or assistant role
    role = payload.get("role")
    return role in ("user", "assistant")


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


class CodexProvider(SessionProvider):
    """Session provider for Codex CLI (OpenAI)."""

    def __init__(self, home: Path | None = None):
        """Initialize Codex provider.

        Args:
            home: Home directory path. Defaults to Path.home().
        """
        self._home = home or Path.home()
        self._turn_cache: dict[Path, TurnCountState] = {}
        self._cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "codex"

    @property
    def watch_patterns(self) -> list[str]:
        return [".codex/sessions/*/*/*/*.jsonl"]

    def parse_session_file(self, path: Path) -> SessionInfo | None:
        """Parse a Codex session JSONL file."""
        if not path.exists():
            return None

        meta = self._read_session_meta(path)
        if not meta:
            return None

        session_id = meta.get("id")
        if not isinstance(session_id, str) or not session_id:
            return None

        # Codex stores cwd in session_meta payload
        cwd = meta.get("cwd", "")
        if not isinstance(cwd, str):
            cwd = ""

        # Get turn count
        turn_count = self.count_turns(path)

        # Use file mtime as last activity (timestamp in meta is start time).
        last_activity = None
        try:
            last_activity = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            pass

        # Derive project from cwd or git info
        project = self.derive_project_from_cwd(cwd) if cwd else None
        git_info = meta.get("git")
        if not project and isinstance(git_info, dict):
            project = self._derive_project_from_git(git_info)

        extra: dict[str, object] = {}
        if isinstance(git_info, dict):
            repo = git_info.get("repository_url")
            if isinstance(repo, str) and repo:
                extra["git_repo"] = repo
            branch = git_info.get("branch")
            if isinstance(branch, str) and branch:
                extra["git_branch"] = branch
            commit = git_info.get("commit_hash")
            if isinstance(commit, str) and commit:
                extra["git_commit"] = commit

        cli_version = meta.get("cli_version")
        if isinstance(cli_version, str) and cli_version:
            extra["cli_version"] = cli_version

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

    def find_session_for_cwd(self, cwd: str) -> SessionInfo | None:
        """Find the most recent session for a given working directory.

        Searches through session files to find one matching the given cwd.
        """
        if not cwd:
            return None

        sessions_dir = self._home / ".codex" / "sessions"
        if not sessions_dir.exists():
            return None

        # Search through all session files (most recent first by path structure)
        candidates: list[tuple[Path, int]] = []

        try:
            # Walk through year/month/day directories
            for year_dir in sorted(sessions_dir.iterdir(), reverse=True):
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                for month_dir in sorted(year_dir.iterdir(), reverse=True):
                    if not month_dir.is_dir() or not month_dir.name.isdigit():
                        continue
                    for day_dir in sorted(month_dir.iterdir(), reverse=True):
                        if not day_dir.is_dir() or not day_dir.name.isdigit():
                            continue
                        for session_file in day_dir.glob("*.jsonl"):
                            meta = self._read_session_meta(session_file)
                            if meta and meta.get("cwd") == cwd:
                                try:
                                    mtime_ns = session_file.stat().st_mtime_ns
                                    candidates.append((session_file, mtime_ns))
                                except OSError:
                                    continue
        except OSError:
            return None

        if not candidates:
            return None

        # Return the most recently modified session
        latest_path = max(candidates, key=lambda x: x[1])[0]
        return self.parse_session_file(latest_path)

    def _read_session_meta(self, path: Path, max_lines: int = 5) -> dict | None:
        """Extract session_meta payload from the beginning of a JSONL file."""
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
                    # Codex format: {"type":"session_meta","payload":{...}}
                    if isinstance(obj, dict) and obj.get("type") == "session_meta":
                        payload = obj.get("payload")
                        if isinstance(payload, dict):
                            return payload
        except OSError:
            pass
        return None

    def _derive_project_from_git(self, git: dict) -> str | None:
        repo = git.get("repository_url")
        if not isinstance(repo, str) or not repo:
            return None

        # Handle common repo URL forms:
        # - git@github.com:org/name.git
        # - https://github.com/org/name(.git)
        tail = repo.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        tail = tail.strip()
        return tail or None

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
                        if _is_turn_event(obj):
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
            if _is_turn_event(obj):
                turns += 1

        return TurnCountState(
            pos=stat.st_size,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            remainder=remainder,
            turn_count=turns,
        )
