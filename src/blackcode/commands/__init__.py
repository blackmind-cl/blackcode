"""Subcomandos de la CLI de Blackcode.

Cada módulo implementa un subcomando con dos piezas:

  - ``register(sub)``: añade su subparser al ``add_subparsers`` raíz y fija
    ``func`` con ``set_defaults``.
  - un manejador privado ``_run(args) -> int`` con la lógica.

La CLI raíz (`blackcode.cli`) solo recorre `ALL_COMMANDS` y delega. Para
añadir un comando nuevo: crear el módulo aquí y sumarlo a la lista, en la
posición en la que deba aparecer en `blackcode --help`.
"""

from __future__ import annotations

# Archivo de configuración que identifica una carpeta como proyecto Blackcode.
PROJECT_CONFIG = "blackcode.yaml"


def all_commands() -> list:
    """Devuelve los módulos de subcomando en el orden del --help."""
    from blackcode.commands import (
        dashboard,
        doctor,
        export,
        generate_qa,
        ingest,
        init,
        models,
        plugins,
        serve,
        train,
    )

    return [
        doctor,
        models,
        init,
        ingest,
        generate_qa,
        plugins,
        train,
        export,
        serve,
        dashboard,
    ]
