"""Tablero de métricas local, sin servicios externos.

Recolecta los `blackcode_metrics.json` que escriben los trainers y genera un
HTML autocontenido (sin dependencias, sin servidor) para revisar y comparar
ejecuciones. Los datos nunca salen de la máquina.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

METRICS_FILE = "blackcode_metrics.json"


def find_runs(roots: list[str]) -> list[tuple[str, dict]]:
    """Busca `blackcode_metrics.json` bajo cada raíz y devuelve (carpeta, métricas)."""
    files: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob(METRICS_FILE)))
    runs: list[tuple[str, dict]] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            runs.append((str(file.parent), data))
    return runs


def render_dashboard(runs: list[tuple[str, dict]]) -> str:
    """Genera una página HTML autocontenida con una tabla de ejecuciones."""
    metric_keys = sorted({key for _, metrics in runs for key in metrics})
    head = "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys)
    body_rows = []
    for name, metrics in runs:
        cells = "".join(
            f"<td>{html.escape(str(metrics.get(k, '—')))}</td>" for k in metric_keys
        )
        body_rows.append(f"<tr><td>{html.escape(name)}</td>{cells}</tr>")
    body = "\n".join(body_rows) or (
        '<tr><td colspan="99">Sin ejecuciones. Entrena un modelo primero.</td></tr>'
    )
    return _HTML.format(rows=body, head=head, n=len(runs))


_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Blackcode · métricas</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#0f1115;color:#e6e6e6}}
h1{{font-size:1.25rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #2a2e38;padding:.5rem .75rem;text-align:left}}
th{{background:#1a1d24}}
tr:nth-child(even) td{{background:#15171d}}
</style>
</head>
<body>
<h1>Blackcode · tablero de métricas ({n} ejecuciones)</h1>
<table>
<thead><tr><th>ejecución</th>{head}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""
