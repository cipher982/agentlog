"""Event parser for extracting events from session files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .models import AgentEvent

if TYPE_CHECKING:
    from typing import Iterator


class EventParser:
    """Parser for extracting events from agent session files.
    
    Handles the different formats used by each provider:
    - Claude: JSONL events
    - Codex: JSONL events (metadata header + messages + tool calls)
    - Gemini: single JSON file with a messages array
    
    Example:
        >>> parser = EventParser('claude')
        >>> for event in parser.parse_session_file(path):
        ...     print(f"{event.role}: {event.content_text[:50]}")
    """
    
    def __init__(self, provider: str):
        """Initialize the parser.
        
        Args:
            provider: Name of the provider ('claude', 'codex', 'gemini')
        """
        self.provider = provider
    
    def parse_session_file(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a session file.
        
        Args:
            path: Path to the session file
            
        Yields:
            AgentEvent for each event in the file
        """
        if self.provider == 'claude':
            yield from self._parse_claude_jsonl(path)
        elif self.provider == 'codex':
            yield from self._parse_codex_jsonl(path)
        elif self.provider == 'gemini':
            yield from self._parse_gemini_json(path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def parse_event(self, raw_json: Mapping[str, Any]) -> AgentEvent:
        """Parse a single event from raw JSON.
        
        Args:
            raw_json: The raw event JSON
            
        Returns:
            Parsed AgentEvent
        """
        if self.provider == 'claude':
            return self._parse_claude_event(raw_json)
        elif self.provider == 'codex':
            event = self._parse_codex_event(raw_json)
            if event is None:
                # Return a minimal event for unsupported types
                return AgentEvent(
                    timestamp=datetime.now(),
                    role="system",
                    event_type=raw_json.get("type"),
                    content_text=None,
                    raw_json=dict(raw_json),
                )
            return event
        elif self.provider == 'gemini':
            return self._parse_gemini_event(raw_json)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _parse_claude_jsonl(self, path: Path) -> Iterator[AgentEvent]:
        """Parse Claude JSONL format."""
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    yield self._parse_claude_event(raw)
                except (json.JSONDecodeError, KeyError):
                    continue
    
    def _parse_claude_event(self, raw: Mapping[str, Any]) -> AgentEvent:
        """Parse a single Claude event."""
        event_type = raw.get('type')
        role = event_type if event_type in ('user', 'assistant', 'system') else 'system'
        
        # Extract timestamp
        ts_str = raw.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        # Extract content
        content_text = None
        message = raw.get('message', {})
        if isinstance(message, dict):
            content = message.get('content', [])
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                content_text = '\n'.join(text_parts) if text_parts else None
            elif isinstance(content, str):
                content_text = content
        
        # Extract tool info
        tool_name = None
        tool_input = None
        tool_output = None
        
        if isinstance(message, dict):
            content = message.get('content', [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get('type') == 'tool_use':
                            tool_name = part.get('name')
                            tool_input = part.get('input')
                        elif part.get('type') == 'tool_result':
                            tool_output = part.get('content')

        tool_output_text = None
        if tool_output is not None:
            tool_output_text = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=event_type,
            content_text=content_text,
            raw_json=dict(raw),
            tool_name=tool_name,
            tool_input_json=tool_input,
            tool_output_text=tool_output_text,
            session_id=raw.get('sessionId') if isinstance(raw.get('sessionId'), str) else None,
            cwd=raw.get('cwd') if isinstance(raw.get('cwd'), str) else None,
        )
    
    def _parse_codex_jsonl(self, path: Path) -> Iterator[AgentEvent]:
        """Parse Codex JSONL format.

        Expected line types:
        New format (Codex CLI):
        - {"type":"session_meta","payload":{"id":"...","cwd":"/path",...}}
        - {"type":"response_item","payload":{"type":"message","role":"user|assistant","content":[...]}}  (yielded)
        - {"type":"response_item","payload":{"type":"function_call","name":"...","arguments":"...","call_id":"..."}}  (yielded)
        - {"type":"response_item","payload":{"type":"function_call_output","call_id":"...","output":"..."}}  (yielded)
        - other event types (ignored)

        Legacy format (older exports):
        - Metadata header (no "type") with keys like: id, timestamp, git
        - {"record_type":"state"} (ignored)
        - {"type":"message","role":"user|assistant","content":[{"type":"input_text|output_text","text":"..."}]} (yielded)
        - {"type":"function_call","name":"...","arguments":"..."} (yielded)
        - {"type":"function_call_output","call_id":"...","output":"..."} (yielded)
        """
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
                event = self._parse_codex_event(raw)
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
    
    def _parse_codex_event(self, raw: Mapping[str, Any]) -> AgentEvent | None:
        """Parse a single Codex event (message, function_call, or function_call_output).

        Supports both:
        - legacy: {"type":"message","role":...,"content":[...]}
        - legacy: {"type":"function_call","name":"...","arguments":"..."}
        - legacy: {"type":"function_call_output","call_id":"...","output":"..."}
        - current: {"type":"response_item","payload":{"type":"message","role":...,"content":[...]}}
        - current: {"type":"response_item","payload":{"type":"function_call","name":"...","arguments":"...","call_id":"..."}}
        - current: {"type":"response_item","payload":{"type":"function_call_output","call_id":"...","output":"..."}}
        """
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
    
    def _parse_gemini_json(self, path: Path) -> Iterator[AgentEvent]:
        """Parse Gemini JSON format."""
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Gemini uses a 'messages' array and stores sessionId at root
        session_id = data.get("sessionId") if isinstance(data, dict) else None
        messages = data.get('messages', []) if isinstance(data, dict) else []
        for msg in messages:
            event = self._parse_gemini_event(msg)
            if isinstance(session_id, str) and session_id and event.session_id is None:
                event = AgentEvent(
                    timestamp=event.timestamp,
                    role=event.role,
                    event_type=event.event_type,
                    content_text=event.content_text,
                    raw_json=event.raw_json,
                    tool_name=event.tool_name,
                    tool_input_json=event.tool_input_json,
                    tool_output_text=event.tool_output_text,
                    session_id=session_id,
                    cwd=event.cwd,
                    git_repo=event.git_repo,
                    git_branch=event.git_branch,
                    git_commit=event.git_commit,
                )
            yield event
    
    def _parse_gemini_event(self, raw: Mapping[str, Any]) -> AgentEvent:
        """Parse a single Gemini event."""
        msg_type = raw.get("type")
        if msg_type == "gemini":
            role = "assistant"
        else:
            role = "user"
        
        content_text = None
        content = raw.get("content")
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            content_text = "\n".join(parts) if parts else None
        
        timestamp = datetime.now()
        ts_str = raw.get("timestamp")
        if isinstance(ts_str, str) and ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=msg_type if isinstance(msg_type, str) else None,
            content_text=content_text,
            raw_json=dict(raw),
        )
