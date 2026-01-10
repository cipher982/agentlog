<p align="center">
  <img src="assets/agentlog-banner.svg" alt="agentlog" width="900" />
</p>

Unified Python API for reading and normalizing agent session logs from **Claude Code**, **Codex CLI**, **Gemini CLI**, and **Cursor**.

## Why

If you bounce between agent tools, your local history ends up scattered across JSONL files and SQLite databases. `agentlog` gives you:

- **One schema**: `SessionInfo` + `AgentEvent` across all providers (raw JSON preserved).
- **Fast turn counts**: incremental, cached turn counting for JSONL logs.
- **Zero deps**: pure stdlib (Python `>=3.11`).
- **Project-aware sessions**: derives a project name by walking up to a `.git` root.
- **Lazy providers**: provider implementations load only when used.

## Architecture

![agentlog architecture](assets/agentlog-architecture.svg)

## Install

```bash
uv pip install agentlog
# or
pip install agentlog
```

## Quickstart

```python
from agentlog import SessionRegistry, EventParser

# 1) Discover sessions
registry = SessionRegistry()
for session in registry.list_sessions():
    print(session.provider, session.session_id, session.turn_count, session.project)

# 2) Stream normalized events
session = next(registry.list_sessions(provider="codex"))
parser = EventParser(session.provider, registry=registry)
for event in parser.parse_session_file(session.file_path):
    if event.is_tool_call:
        print("tool:", event.tool_name)
    elif event.content_text:
        print(event.role + ":", event.content_text[:60])
```

## Event Stream

![agentlog event stream](assets/agentlog-event-stream.svg)

## Providers

| Provider | Source | Format |
|---|---|---|
| Claude Code | `~/.claude/projects/<slug>/*.jsonl` | JSONL |
| Codex CLI | `~/.codex/sessions/YYYY/MM/DD/*.jsonl` | JSONL |
| Gemini CLI | `~/.gemini/tmp/<hash>/chats/session*.json` | JSON |
| Cursor | `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb` | SQLite |

## License

MIT
