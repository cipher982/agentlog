"""Codex CLI session provider.

Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/*.jsonl where:
- Each file is JSONL (JSON Lines) format.
- First line is a metadata object with keys like: {"id":"uuid","timestamp":"ISO8601",...}
- Conversation events are {"type":"message","role":"user|assistant",...}
- Tool calls use {"type":"function_call",...} and {"type":"function_call_output",...}
- Many {"record_type":"state"} lines can appear and should be ignored.

Turn counting rules:
- Count: type="message" with role in ("user", "assistant")
- Skip: metadata header (no type)
- Skip: record_type="state" (metadata)
- Skip: tool calls (type="function_call" and "function_call_output")
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
    - type="message" and role in ("user", "assistant")
    """
    if not isinstance(obj, dict):
        return False

    if obj.get("type") != "message":
        return False

    return obj.get("role") in ("user", "assistant")


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

        # Codex sessions don't reliably include cwd; keep blank.
        cwd = ""

        # Get turn count
        turn_count = self.count_turns(path)

        # Use file mtime as last activity (timestamp in meta is start time).
        last_activity = None
        try:
            last_activity = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            pass

        project = self._derive_project_from_git(meta.get("git"))

        extra: dict[str, object] = {}
        git_info = meta.get("git")
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

        Codex session logs in ~/.codex/sessions do not reliably include cwd.
        This method is currently unsupported and returns None.
        """
        return None

    def _read_session_meta(self, path: Path, max_lines: int = 5) -> dict | None:
        """Extract the metadata header from the beginning of a JSONL file."""
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
                    # Metadata line has the session id/timestamp and typically no "type".
                    if isinstance(obj, dict) and "id" in obj and "timestamp" in obj and "type" not in obj:
                        return obj
        except OSError:
            pass
        return None

    def _derive_project_from_git(self, git: object) -> str | None:
        if not isinstance(git, dict):
            return None
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
