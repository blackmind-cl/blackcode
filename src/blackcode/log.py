"""Logging estructurado y progreso, sin dependencias.

El núcleo no añade librerías para esto: usa la estándar. Los logs van a
stderr; los resultados (métricas, reportes) se imprimen a stdout con `print`,
de modo que ambos flujos se puedan separar y redirigir por separado.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, Iterator, TypeVar

_T = TypeVar("_T")
_ROOT = "blackcode"
_configured = False


def configure_logging(verbose: bool = False) -> None:
    """Configura el logger 'blackcode' con un handler a stderr. Idempotente."""
    global _configured
    logger = logging.getLogger(_ROOT)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s · %(name)s · %(levelname)s · %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger hijo de 'blackcode' (p.ej. 'blackcode.pipeline')."""
    return logging.getLogger(f"{_ROOT}.{name}")


def track(iterable: Iterable[_T], description: str = "") -> Iterator[_T]:
    """Itera mostrando una barra de progreso simple en stderr.

    Sin dependencias. Si stderr no es una terminal interactiva no imprime
    nada, para no contaminar logs ni archivos redirigidos. Pensado para
    bucles acotados (p.ej. evaluación); el entrenamiento de LLM ya muestra
    su propia barra vía transformers.
    """
    items = list(iterable)
    total = len(items)
    show = sys.stderr.isatty() and total > 0
    for i, item in enumerate(items, 1):
        if show:
            filled = "#" * (20 * i // total)
            print(
                f"\r{description} [{filled:<20}] {i}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )
        yield item
    if show:
        print(file=sys.stderr)
