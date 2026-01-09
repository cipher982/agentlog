"""Event parser for extracting events from session files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .models import AgentEvent

if TYPE_CHECKING:
    from typing import Iterator


class EventParser:
    """Parser for extracting events from agent session files.
    
    Handles the different formats used by each provider
    (Claude JSONL, Codex/Gemini JSON).
    
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
            yield from self._parse_codex_json(path)
        elif self.provider == 'gemini':
            yield from self._parse_gemini_json(path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def parse_event(self, raw_json: dict) -> AgentEvent:
        """Parse a single event from raw JSON.
        
        Args:
            raw_json: The raw event JSON
            
        Returns:
            Parsed AgentEvent
        """
        if self.provider == 'claude':
            return self._parse_claude_event(raw_json)
        elif self.provider == 'codex':
            return self._parse_codex_event(raw_json)
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
    
    def _parse_claude_event(self, raw: dict) -> AgentEvent:
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
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=event_type,
            content_text=content_text,
            raw_json=raw,
            tool_name=tool_name,
            tool_input_json=tool_input,
            tool_output_text=tool_output,
            session_id=raw.get('sessionId'),
            cwd=raw.get('cwd'),
        )
    
    def _parse_codex_json(self, path: Path) -> Iterator[AgentEvent]:
        """Parse Codex JSON format."""
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Codex stores conversations in an 'instructions' array
        instructions = data.get('instructions', [])
        for entry in instructions:
            yield self._parse_codex_event(entry)
    
    def _parse_codex_event(self, raw: dict) -> AgentEvent:
        """Parse a single Codex event."""
        role = raw.get('role', 'user')
        content = raw.get('content', '')
        
        # Codex may have timestamps
        ts_str = raw.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=role,
            content_text=content if isinstance(content, str) else None,
            raw_json=raw,
        )
    
    def _parse_gemini_json(self, path: Path) -> Iterator[AgentEvent]:
        """Parse Gemini JSON format."""
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Gemini uses a 'messages' array
        messages = data.get('messages', [])
        for msg in messages:
            yield self._parse_gemini_event(msg)
    
    def _parse_gemini_event(self, raw: dict) -> AgentEvent:
        """Parse a single Gemini event."""
        role = raw.get('role', 'user')
        
        # Gemini content can be a string or list of parts
        content = raw.get('content', '')
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            content_text = '\n'.join(text_parts) if text_parts else None
        else:
            content_text = content
        
        ts_str = raw.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        return AgentEvent(
            timestamp=timestamp,
            role=role,
            event_type=role,
            content_text=content_text,
            raw_json=raw,
        )
