"""Session registry for discovering and managing sessions across providers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .providers.base import SessionProvider
from .models import SessionInfo

if TYPE_CHECKING:
    from typing import Iterator


class SessionRegistry:
    """Registry for discovering sessions across all providers.
    
    The registry discovers provider implementations and provides
    a unified interface for listing sessions from all providers.
    
    Example:
        >>> registry = SessionRegistry()
        >>> for session in registry.list_sessions():
        ...     print(f"{session.provider}: {session.session_id}")
    """
    
    def __init__(self, providers: list[SessionProvider] | None = None):
        """Initialize the registry.
        
        Args:
            providers: Optional list of providers. If not provided,
                      discovers all available providers automatically.
        """
        if providers is not None:
            self._providers = {p.name: p for p in providers}
        else:
            self._providers = self._discover_providers()
    
    def _discover_providers(self) -> dict[str, SessionProvider]:
        """Discover all available provider implementations."""
        providers = {}
        
        # Import providers lazily to avoid circular imports
        try:
            from .providers.claude import ClaudeProvider
            providers['claude'] = ClaudeProvider()
        except ImportError:
            pass
        
        try:
            from .providers.codex import CodexProvider
            providers['codex'] = CodexProvider()
        except ImportError:
            pass
        
        try:
            from .providers.gemini import GeminiProvider
            providers['gemini'] = GeminiProvider()
        except ImportError:
            pass
        
        try:
            from .providers.cursor import CursorProvider
            providers['cursor'] = CursorProvider()
        except ImportError:
            pass
        
        return providers
    
    def get_provider(self, name: str) -> SessionProvider:
        """Get a provider by name.
        
        Args:
            name: Provider name (e.g., 'claude', 'codex')
            
        Returns:
            The provider instance
            
        Raises:
            KeyError: If provider not found
        """
        return self._providers[name]
    
    @property
    def providers(self) -> list[SessionProvider]:
        """List all registered providers."""
        return list(self._providers.values())
    
    @property
    def provider_names(self) -> list[str]:
        """List names of all registered providers."""
        return list(self._providers.keys())
    
    def list_sessions(self, provider: str | None = None) -> Iterator[SessionInfo]:
        """List all sessions across providers.
        
        Args:
            provider: Optional provider name to filter by
            
        Yields:
            SessionInfo for each discovered session
        """
        if provider is not None:
            providers_to_check = [self._providers[provider]]
        else:
            providers_to_check = self._providers.values()
        
        for prov in providers_to_check:
            yield from self._list_provider_sessions(prov)
    
    def _list_provider_sessions(self, provider: SessionProvider) -> Iterator[SessionInfo]:
        """List sessions for a specific provider."""
        for pattern in provider.watch_patterns:
            # Expand ~ and relative paths to absolute
            if pattern.startswith('~'):
                expanded = Path(pattern).expanduser()
            elif pattern.startswith('/'):
                expanded = Path(pattern)
            else:
                # Relative pattern - assume relative to home directory
                expanded = Path.home() / pattern
            
            # Find the non-glob prefix to use as base directory
            pattern_str = str(expanded)
            pattern_parts = pattern_str.split('/')
            base_parts = []
            for part in pattern_parts:
                if '*' in part or '?' in part or '[' in part:
                    break
                base_parts.append(part)
            
            if base_parts:
                base_dir = Path('/'.join(base_parts))
                remaining = '/'.join(pattern_parts[len(base_parts):])
            else:
                base_dir = Path.home()
                remaining = pattern_str.replace(str(Path.home()) + '/', '')
            
            if not base_dir.exists():
                continue
            
            for file_path in base_dir.glob(remaining):
                if file_path.is_file():
                    try:
                        session = provider.parse_session_file(file_path)
                        if session is not None:
                            yield session
                    except Exception:
                        # Skip files that can't be parsed
                        continue
    
    def find_session_for_cwd(self, cwd: str, provider: str | None = None) -> SessionInfo | None:
        """Find the most recent session for a working directory.
        
        Args:
            cwd: Working directory path
            provider: Optional provider to search in
            
        Returns:
            SessionInfo for the most recent session, or None
        """
        if provider is not None:
            providers_to_check = [self._providers[provider]]
        else:
            providers_to_check = self._providers.values()
        
        sessions: list[SessionInfo] = []
        for prov in providers_to_check:
            session = prov.find_session_for_cwd(cwd)
            if session is not None:
                sessions.append(session)
        
        if not sessions:
            return None
        
        # Return the most recently active session
        def _activity_key(s: SessionInfo) -> datetime:
            if s.last_activity is not None:
                return s.last_activity
            try:
                return datetime.fromtimestamp(s.file_path.stat().st_mtime)
            except OSError:
                return datetime.fromtimestamp(0)

        return max(sessions, key=_activity_key)
