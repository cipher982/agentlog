"""Backwards compatibility re-exports.

Providers import from ..provider, so this module re-exports
the base class and models from their canonical locations.
"""

from .providers.base import SessionProvider
from .models import SessionInfo

__all__ = ["SessionProvider", "SessionInfo"]
