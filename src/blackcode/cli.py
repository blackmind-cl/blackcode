"""Interfaz de línea de comandos de Blackcode.

Usa solo argparse (sin dependencias) para que la CLI esté siempre disponible.
Cada subcomando vive en su propio módulo bajo `blackcode.commands`; este
archivo solo arma el parser raíz y despacha.

Comandos:
  blackcode doctor              Reporte de hardware y estrategia recomendada
  blackcode models              Catálogo de modelos recomendados por VRAM
  blackcode init [carpeta]      Crea un proyecto (carpeta con su configuración)
  blackcode ingest [fuentes]    Ingiere fuentes (archivos/web/sitemap/SQL/Notion/…) a un JSONL
  blackcode generate-qa <jsonl> Genera pares pregunta/respuesta desde un corpus
  blackcode trainers            Lista los trainers disponibles
  blackcode exporters           Lista los exportadores disponibles
  blackcode train  <config>     Ejecuta el pipeline completo
  blackcode eval   <config>     Evalúa un modelo ya entrenado (sin reentrenar)
  blackcode export <config>     Exporta el modelo (GGUF / ONNX / merge-lora)
  blackcode serve  <config>     Sirve el modelo entrenado (API estilo OpenAI)
  blackcode dashboard [rutas]   Genera un tablero HTML de métricas
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from blackcode import __version__
from blackcode.commands import all_commands
from blackcode.log import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackcode",
        description="Motor open source de entrenamiento e inferencia de IA local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Ejemplos:
              blackcode doctor
              blackcode init mi-proyecto       # crea la carpeta del proyecto
              cd mi-proyecto
              blackcode ingest https://miempresa.com --crawl-depth 2
              blackcode train                  # usa ./blackcode.yaml
              blackcode serve
            """
        ),
    )
    parser.add_argument("--version", action="version", version=f"blackcode {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Salida detallada (nivel DEBUG)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in all_commands():
        command.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCancelado.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        get_logger("cli").error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
