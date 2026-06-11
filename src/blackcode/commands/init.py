"""`blackcode init` — crea un proyecto (asistente interactivo o plantilla)."""

from __future__ import annotations

import argparse
import sys

from blackcode.commands import PROJECT_CONFIG


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from blackcode.templates import DEFAULT_TEMPLATE, TEMPLATES

    if args.template and args.template not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES))
        print(
            f"Plantilla desconocida: '{args.template}'. Disponibles: {available}",
            file=sys.stderr,
        )
        return 1

    # Sin argumento, el proyecto es el directorio actual; con argumento, se
    # crea esa carpeta. Cada proyecto es autónomo: config, datos y salidas.
    project_dir = Path(args.directory) if args.directory else Path.cwd()
    if project_dir.exists() and not project_dir.is_dir():
        print(f"{project_dir} existe y no es un directorio.", file=sys.stderr)
        return 1
    config_path = project_dir / PROJECT_CONFIG
    if config_path.exists() and not args.force:
        print(
            f"Ya existe un proyecto en {project_dir} ({PROJECT_CONFIG}). "
            "Usa --force para sobrescribir.",
            file=sys.stderr,
        )
        return 1

    if args.template:
        content = TEMPLATES[args.template]
        origen = f"plantilla '{args.template}'"
    elif sys.stdin.isatty():
        import yaml

        from blackcode.config import RunConfig
        from blackcode.wizard import run_wizard

        config = run_wizard(default_name=project_dir.resolve().name or "proyecto")
        if config is None:  # el usuario eligió «salir»
            print("Asistente cancelado; no se creó ningún proyecto.")
            return 0
        RunConfig.from_dict(config)  # valida antes de escribir
        content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        origen = "asistente interactivo"
    else:
        content = TEMPLATES[DEFAULT_TEMPLATE]
        origen = f"plantilla '{DEFAULT_TEMPLATE}' (sin terminal interactiva)"

    project_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    (project_dir / "data").mkdir(exist_ok=True)

    print(f"\nProyecto Blackcode creado en {project_dir} ({origen}).")
    print(f"  {PROJECT_CONFIG}   configuración del proyecto")
    print("  data/            coloca aquí tus fuentes y datasets")
    if args.directory:
        print(f"\nSiguiente paso:  cd {args.directory} && blackcode train")
    else:
        print("\nSiguiente paso:  blackcode train")
    return 0


def register(sub) -> None:
    p = sub.add_parser(
        "init", help="Crear un proyecto (asistente interactivo o plantilla)"
    )
    p.add_argument(
        "directory",
        nargs="?",
        help="Carpeta del proyecto (por defecto: el directorio actual)",
    )
    p.add_argument(
        "--template",
        help="Plantilla no interactiva: chatbot | clasificador | embeddings",
    )
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_run)
