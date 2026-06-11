"""`blackcode export` — convierte el modelo entrenado a un formato de despliegue."""

from __future__ import annotations

import argparse

from blackcode.commands import PROJECT_CONFIG


def _run(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.exporters import export_model

    config = load_config(args.config)
    results = export_model(config, args.to)
    print("Exportación completada:")
    for result in results:
        print(f"  {result.format:<12} -> {result.path}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("export", help="Exportar el modelo a despliegue")
    p.add_argument("config", nargs="?", default=PROJECT_CONFIG)
    p.add_argument(
        "--to",
        action="append",
        metavar="FORMATO",
        help="Formato de exportación (repetible): merge-lora | gguf | onnx",
    )
    p.set_defaults(func=_run)
