from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agentlog import SessionRegistry
from agentlog.models import SessionInfo
from agentlog.providers.base import SessionProvider


class _DummyProvider(SessionProvider):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def watch_patterns(self) -> list[str]:
        return []

    def parse_session_file(self, path: Path) -> SessionInfo | None:
        return None

    def count_turns(self, path: Path) -> int:
        return 0

    def find_session_for_cwd(self, cwd: str) -> SessionInfo | None:
        if cwd != "/tmp/project":
            return None
        return SessionInfo(
            session_id="s1",
            provider=self.name,
            cwd=cwd,
            file_path=Path("/tmp/fake.jsonl"),
            turn_count=2,
            project="project",
            last_activity=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )


def test_registry_find_session_for_cwd_uses_sessioninfo() -> None:
    reg = SessionRegistry(providers=[_DummyProvider()])
    info = reg.find_session_for_cwd("/tmp/project")
    assert info is not None
    assert info.session_id == "s1"
    assert info.provider == "dummy"


def test_registry_list_sessions_globs_and_skips_parse_errors(tmp_path: Path) -> None:
    (tmp_path / "good-1.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "good-2.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text("{}", encoding="utf-8")

    class _GlobProvider(SessionProvider):
        @property
        def name(self) -> str:
            return "glob"

        @property
        def watch_patterns(self) -> list[str]:
            return [str(tmp_path / "*.jsonl")]

        def parse_session_file(self, path: Path) -> SessionInfo | None:
            if path.name == "bad.jsonl":
                raise ValueError("boom")
            return SessionInfo(
                session_id=path.stem,
                provider=self.name,
                cwd="/tmp/project",
                file_path=path,
                turn_count=1,
            )

        def count_turns(self, path: Path) -> int:
            return 1

    reg = SessionRegistry(providers=[_GlobProvider()])
    sessions = list(reg.list_sessions())
    assert sorted(s.session_id for s in sessions) == ["good-1", "good-2"]
