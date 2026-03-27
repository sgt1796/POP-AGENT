from __future__ import annotations

import pytest

import run_web_ui


def test_resolve_frontend_command_prefers_windows_npm_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(command: str) -> str | None:
        return {"npm.cmd": r"C:\Program Files\nodejs\npm.cmd"}.get(command)

    monkeypatch.setattr(run_web_ui.shutil, "which", fake_which)

    assert (
        run_web_ui._resolve_frontend_command("nt")
        == r"C:\Program Files\nodejs\npm.cmd"
    )


def test_resolve_frontend_command_uses_npm_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_web_ui.shutil,
        "which",
        lambda command: "/usr/bin/npm" if command == "npm" else None,
    )

    assert run_web_ui._resolve_frontend_command("posix") == "/usr/bin/npm"


def test_resolve_frontend_command_raises_helpful_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_web_ui.shutil, "which", lambda _command: None)

    with pytest.raises(FileNotFoundError, match="Could not find npm in PATH"):
        run_web_ui._resolve_frontend_command("nt")
