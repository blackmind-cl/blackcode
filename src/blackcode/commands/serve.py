"""`blackcode serve` — sirve el modelo entrenado con una API estilo OpenAI."""

from __future__ import annotations

import argparse

from blackcode.commands import PROJECT_CONFIG


def _run(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.serving import serve_model

    serve_model(load_config(args.config))
    return 0


def register(sub) -> None:
    p = sub.add_parser("serve", help="Servir el modelo entrenado")
    p.add_argument("config", nargs="?", default=PROJECT_CONFIG)
    p.set_defaults(func=_run)
