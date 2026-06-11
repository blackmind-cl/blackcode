"""`blackcode trainers` / `blackcode exporters` — listar plugins registrados."""

from __future__ import annotations

import argparse


def _run_trainers(_: argparse.Namespace) -> int:
    import blackcode.trainers  # noqa: F401  (registra los de fábrica)
    from blackcode.registry import list_trainers

    print("Trainers disponibles:")
    for name in list_trainers():
        print(f"  - {name}")
    return 0


def _run_exporters(_: argparse.Namespace) -> int:
    import blackcode.exporters  # noqa: F401  (registra los de fábrica)
    from blackcode.registry import list_exporters

    print("Exportadores disponibles:")
    for name in list_exporters():
        print(f"  - {name}")
    return 0


def register(sub) -> None:
    sub.add_parser("trainers", help="Listar trainers").set_defaults(
        func=_run_trainers
    )
    sub.add_parser("exporters", help="Listar exportadores").set_defaults(
        func=_run_exporters
    )
