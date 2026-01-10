# agentlog

Unified Python API for reading agent session logs from:
- Claude Code
- Codex CLI
- Gemini CLI
- Cursor

## Features

- **Unified Model**: Consistent `SessionInfo` and `AgentEvent` models across all providers.
- **Incremental Turn Counting**: Fast, cached turn counting for JSONL logs.
- **Project Discovery**: Automatically derives project names by searching for `.git` roots.
- **Lazy Discovery**: Providers are loaded only when needed.

## Installation

```bash
pip install agentlog
```

## Usage

```python
from agentlog import SessionRegistry, EventParser

# List all sessions across all providers
registry = SessionRegistry()
for session in registry.list_sessions():
    print(f"{session.provider}: {session.session_id} ({session.turn_count} turns)")

# Parse events from a specific session
parser = EventParser('claude')
for event in parser.parse_session_file(session.file_path):
    print(f"{event.role}: {event.content_text[:50]}")
```

## Providers

### Claude Code
Sessions stored in `~/.claude/projects/<slug>/*.jsonl`

### Codex CLI  
Sessions stored in `~/.codex/sessions/YYYY/MM/DD/*.jsonl`

### Gemini CLI
Sessions stored in `~/.gemini/tmp/<hash>/chats/session-*.json`

### Cursor
Sessions stored in SQLite database at `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb`

## License

MIT
