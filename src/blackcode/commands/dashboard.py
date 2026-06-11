"""`blackcode dashboard` — tablero HTML estático con las métricas de los runs."""

from __future__ import annotations

import argparse


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from blackcode.dashboard import find_runs, render_dashboard

    runs = find_runs(args.paths or ["."])
    out = Path(args.output)
    out.write_text(render_dashboard(runs), encoding="utf-8")
    print(f"Tablero con {len(runs)} ejecución(es) escrito en {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("dashboard", help="Generar un tablero HTML de métricas")
    p.add_argument(
        "paths", nargs="*", help="Carpetas a escanear (por defecto: el dir actual)"
    )
    p.add_argument("--output", default="dashboard.html", help="HTML de salida")
    p.set_defaults(func=_run)
