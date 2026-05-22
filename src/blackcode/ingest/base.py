"""Contrato de los extractores de documentos.

Un extractor convierte un tipo de archivo en texto plano. El subsistema de
ingesta los resuelve por extensión para producir un corpus de entrenamiento.
"""

from __future__ import annotations

import abc
from pathlib import Path


class BaseExtractor(abc.ABC):
    """Interfaz común a todos los extractores de documentos."""

    # Extensiones de archivo que maneja este extractor, p.ej. (".pdf",).
    extensions: tuple[str, ...] = ()

    @abc.abstractmethod
    def extract(self, path: Path) -> str:
        """Devuelve el texto plano del documento en `path`."""
