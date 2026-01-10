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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..providers.base import JSONLProvider, SessionInfo, AgentEvent


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


class CodexProvider(JSONLProvider):
    """Session provider for Codex CLI (OpenAI)."""

    def __init__(self, home: Path | None = None):
        """Initialize Codex provider.

        Args:
            home: Home directory path. Defaults to Path.home().
        """
        super().__init__()
        self._home = home or Path.home()

    @property
    def name(self) -> str:
        return "codex"

    def is_turn_event(self, obj: dict) -> bool:
        """Check if a JSONL event counts as a conversational turn."""
        return _is_turn_event(obj)

    def parse_events(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a Codex session file."""
        session_id: str | None = None
        cwd: str | None = None
        git_repo: str | None = None
        git_branch: str | None = None
        git_commit: str | None = None

        def _capture_git_from_mapping(m: Mapping[str, Any]) -> None:
            nonlocal git_repo, git_branch, git_commit
            repo = m.get("repository_url")
            if isinstance(repo, str) and repo:
                git_repo = repo
            branch = m.get("branch")
            if isinstance(branch, str) and branch:
                git_branch = branch
            commit = m.get("commit_hash")
            if isinstance(commit, str) and commit:
                git_commit = commit

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue

                # New format: session metadata.
                if raw.get("type") == "session_meta":
                    payload = raw.get("payload")
                    if isinstance(payload, dict):
                        sid = payload.get("id")
                        if isinstance(sid, str) and sid:
                            session_id = sid
                        session_cwd = payload.get("cwd")
                        if isinstance(session_cwd, str) and session_cwd:
                            cwd = session_cwd
                        git = payload.get("git")
                        if isinstance(git, dict):
                            _capture_git_from_mapping(git)
                    continue

                # New format: turn context can include cwd.
                if raw.get("type") == "turn_context":
                    payload = raw.get("payload")
                    if isinstance(payload, dict):
                        context_cwd = payload.get("cwd")
                        if isinstance(context_cwd, str) and context_cwd:
                            cwd = context_cwd
                    continue

                # Metadata header (first line): {"id": "...", "timestamp": "...", ...} with no "type"
                if "type" not in raw and "id" in raw and "timestamp" in raw:
                    sid = raw.get("id")
                    if isinstance(sid, str) and sid:
                        session_id = sid
                    git = raw.get("git")
                    if isinstance(git, dict):
                        _capture_git_from_mapping(git)
                    continue

                # Parse the event (message, function_call, or function_call_output)
                event = self._parse_event(raw)
                if event is None:
                    continue

                # Enrich event with session context
                if session_id or cwd or git_repo or git_branch or git_commit:
                    event = AgentEvent(
                        timestamp=event.timestamp,
                        role=event.role,
                        event_type=event.event_type,
                        content_text=event.content_text,
                        raw_json=event.raw_json,
                        tool_name=event.tool_name,
                        tool_input_json=event.tool_input_json,
                        tool_output_text=event.tool_output_text,
                        session_id=event.session_id or session_id,
                        cwd=event.cwd or cwd,
                        git_repo=event.git_repo or git_repo,
                        git_branch=event.git_branch or git_branch,
                        git_commit=event.git_commit or git_commit,
                    )
                yield event

    def _parse_event(self, raw: Mapping[str, Any]) -> AgentEvent | None:
        """Parse a single Codex event."""
        raw_type = raw.get("type")

        # Extract timestamp (used by all event types)
        timestamp = datetime.now()
        ts_str = raw.get("timestamp")
        if isinstance(ts_str, str) and ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Handle response_item wrapper (current format)
        if raw_type == "response_item":
            payload = raw.get("payload")
            if not isinstance(payload, dict):
                return None
            payload_type = payload.get("type")

            if payload_type == "message":
                return self._parse_codex_message(payload, raw, timestamp)
            elif payload_type == "function_call":
                return self._parse_codex_function_call(payload, raw, timestamp)
            elif payload_type == "function_call_output":
                return self._parse_codex_function_call_output(payload, raw, timestamp)
            else:
                return None

        # Handle legacy format (type at top level)
        elif raw_type == "message":
            return self._parse_codex_message(raw, raw, timestamp)
        elif raw_type == "function_call":
            return self._parse_codex_function_call(raw, raw, timestamp)
        elif raw_type == "function_call_output":
            return self._parse_codex_function_call_output(raw, raw, timestamp)

        return None

    def _parse_codex_message(
        self, obj: Mapping[str, Any], raw: Mapping[str, Any], timestamp: datetime
    ) -> AgentEvent:
        """Parse a Codex message event."""
        role = obj.get("role", "user")

        content_text = None
        content = obj.get("content")
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            content_text = "\n".join(parts) if parts else None

        return AgentEvent(
            timestamp=timestamp,
            role=role if role in ("user", "assistant", "system") else "user",
            event_type="message",
            content_text=content_text,
            raw_json=dict(raw),
        )

    def _parse_codex_function_call(
        self, obj: Mapping[str, Any], raw: Mapping[str, Any], timestamp: datetime
    ) -> AgentEvent:
        """Parse a Codex function_call event."""
        tool_name = obj.get("name")

        # Arguments may be JSON string or dict
        args = obj.get("arguments")
        tool_input_json: Any = None
        if isinstance(args, str):
            try:
                tool_input_json = json.loads(args)
            except json.JSONDecodeError:
                tool_input_json = args
        elif args is not None:
            tool_input_json = args

        return AgentEvent(
            timestamp=timestamp,
            role="assistant",
            event_type="tool_call",
            content_text=None,
            raw_json=dict(raw),
            tool_name=tool_name,
            tool_input_json=tool_input_json,
        )

    def _parse_codex_function_call_output(
        self, obj: Mapping[str, Any], raw: Mapping[str, Any], timestamp: datetime
    ) -> AgentEvent:
        """Parse a Codex function_call_output event."""
        output = obj.get("output")
        tool_output_text: str | None = None

        if isinstance(output, str):
            # Output may be JSON-encoded with nested "output" field
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict) and "output" in parsed:
                    tool_output_text = parsed.get("output")
                else:
                    tool_output_text = output
            except json.JSONDecodeError:
                tool_output_text = output
        elif output is not None:
            tool_output_text = json.dumps(output, ensure_ascii=False)

        # call_id can be used to link back to the function_call
        call_id = obj.get("call_id")

        return AgentEvent(
            timestamp=timestamp,
            role="tool",
            event_type="tool_output",
            content_text=None,
            raw_json=dict(raw),
            tool_name=call_id,  # Store call_id in tool_name for linking
            tool_output_text=tool_output_text,
        )

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
