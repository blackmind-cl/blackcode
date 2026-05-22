"""Tests del helper de auto-instalación."""

import subprocess

import pytest

from blackcode import install


def _reset_cache():
    install._attempted.clear()


def test_ensure_extra_noop_when_modules_present(monkeypatch):
    _reset_cache()
    # Simula que todos los módulos están instalados.
    monkeypatch.setattr(
        install.importlib.util, "find_spec", lambda module: object()
    )
    called = []
    monkeypatch.setattr(
        install.subprocess, "run", lambda *a, **k: called.append(a)
    )
    install.ensure_extra("notion", "notion_client")
    assert called == []  # No se ejecuta pip


def test_ensure_extra_runs_pip_when_missing(monkeypatch, capsys):
    _reset_cache()
    monkeypatch.setattr(install.importlib.util, "find_spec", lambda m: None)
    commands = []

    def fake_run(cmd, check):
        commands.append(cmd)

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    install.ensure_extra("notion", "notion_client")
    assert len(commands) == 1
    assert "pip" in commands[0] and "blackcode[notion]" in commands[0]
    # Y se informó por stderr.
    err = capsys.readouterr().err
    assert "blackcode[notion]" in err


def test_ensure_extra_caches_failure_within_process(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(install.importlib.util, "find_spec", lambda m: None)

    def failing_run(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(install.subprocess, "run", failing_run)
    with pytest.raises(ImportError):
        install.ensure_extra("xyz", "modulo_x")

    # Segundo intento: NO se ejecuta pip otra vez (cacheado), pero igual lanza.
    calls = []
    monkeypatch.setattr(
        install.subprocess, "run", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ImportError, match="intento previo"):
        install.ensure_extra("xyz", "modulo_x")
    assert calls == []  # No reintenta


def test_ensure_extra_handles_dotted_module_names(monkeypatch):
    _reset_cache()
    seen = []

    def find_spec(name):
        seen.append(name)
        return None  # missing

    monkeypatch.setattr(install.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: None)
    install.ensure_extra("gdocs", "google.oauth2", "googleapiclient")
    assert "google.oauth2" in seen and "googleapiclient" in seen
