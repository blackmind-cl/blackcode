"""Contrato que todo exportador debe cumplir.

Un exportador toma un modelo ya entrenado (en `train.output_dir`) y produce
un artefacto de despliegue: lo convierte de formato (GGUF, ONNX) o lo
transforma (fusión de adaptadores LoRA en el modelo base).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blackcode.config import RunConfig


@dataclass
class ExportResult:
    format: str
    path: str                    # archivo o carpeta resultante
    extra: dict[str, Any]


class BaseExporter(abc.ABC):
    """Interfaz común a todos los exportadores."""

    def __init__(self, config: RunConfig):
        self.config = config

    @property
    def model_dir(self) -> Path:
        """Carpeta del modelo entrenado que se va a exportar."""
        return Path(self.config.train.output_dir)

    @abc.abstractmethod
    def export(self) -> ExportResult:
        """Genera el artefacto de despliegue y devuelve su ruta."""
