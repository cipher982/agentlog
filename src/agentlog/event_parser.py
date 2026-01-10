"""Event parser for extracting events from session files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .registry import SessionRegistry
from .models import AgentEvent

if TYPE_CHECKING:
    from typing import Iterator


class EventParser:
    """Parser for extracting events from agent session files.
    
    Delegates parsing to the appropriate SessionProvider.
    
    Example:
        >>> parser = EventParser('claude')
        >>> for event in parser.parse_session_file(path):
        ...     print(f"{event.role}: {event.content_text[:50]}")
    """
    
    def __init__(self, provider: str, registry: SessionRegistry | None = None):
        """Initialize the parser.
        
        Args:
            provider: Name of the provider ('claude', 'codex', 'gemini', 'cursor')
            registry: Optional SessionRegistry. If not provided, creates a new one.
        """
        self.provider_name = provider
        self.registry = registry or SessionRegistry()
        self.provider = self.registry.get_provider(provider)
    
    def parse_session_file(self, path: Path) -> Iterator[AgentEvent]:
        """Parse all events from a session file.
        
        Args:
            path: Path to the session file
            
        Yields:
            AgentEvent for each event in the file
        """
        yield from self.provider.parse_events(path)
    
    def parse_session(self, session: SessionInfo) -> Iterator[AgentEvent]:
        """Parse all events for a given session.
        
        This is the preferred way to parse events as it handles non-file
        providers (like Cursor) correctly.
        
        Args:
            session: The session info to parse events for
            
        Yields:
            AgentEvent for each event in the session
        """
        yield from self.provider.parse_session(session)
    
    def parse_event(self, raw_json: Mapping[str, Any]) -> AgentEvent:
        """Parse a single event from raw JSON.
        
        Args:
            raw_json: The raw event JSON
            
        Returns:
            Parsed AgentEvent
        """
        # Some providers might not have a public single-event parser
        # Fallback to a generic AgentEvent if _parse_event is not available
        parse_method = getattr(self.provider, "_parse_event", None)
        if callable(parse_method):
            event = parse_method(raw_json)
            if event is not None:
                return event
        
        from datetime import datetime
        return AgentEvent(
            timestamp=datetime.now(),
            role="system",
            event_type=raw_json.get("type"),
            content_text=None,
            raw_json=dict(raw_json),
        )
