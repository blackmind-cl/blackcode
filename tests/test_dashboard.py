"""Tests del tablero de métricas local."""

import json
from pathlib import Path

from blackcode.dashboard import find_runs, render_dashboard


def _write_metrics(directory: Path, **metrics) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "blackcode_metrics.json").write_text(json.dumps(metrics))


def test_find_runs_collects_metrics_files(tmp_path: Path):
    _write_metrics(tmp_path / "run-a", accuracy=0.9)
    _write_metrics(tmp_path / "run-b", loss=0.3)
    assert len(find_runs([str(tmp_path)])) == 2


def test_find_runs_ignores_invalid_json(tmp_path: Path):
    (tmp_path / "blackcode_metrics.json").write_text("{no es json")
    assert find_runs([str(tmp_path)]) == []


def test_render_dashboard_includes_metric_values():
    runs = [("run-a", {"accuracy": 0.92}), ("run-b", {"accuracy": 0.81})]
    out = render_dashboard(runs)
    assert "<table" in out
    assert "accuracy" in out
    assert "0.92" in out and "0.81" in out


def test_render_dashboard_handles_no_runs():
    assert "Sin ejecuciones" in render_dashboard([])
