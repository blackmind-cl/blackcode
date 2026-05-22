"""Contrato que todo trainer debe cumplir.

Un trainer encapsula un tipo de modelo (LLM, clásico, etc.) detrás de una
interfaz común para que el pipeline sea agnóstico al backend.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blackcode.config import RunConfig
from blackcode.data.base import Dataset
from blackcode.hardware import HardwareProfile, Strategy


@dataclass
class TrainResult:
    output_dir: str
    metrics: dict[str, float]
    extra: dict[str, Any]


class BaseTrainer(abc.ABC):
    """Interfaz común a todos los trainers."""

    def __init__(self, config: RunConfig, hardware: HardwareProfile):
        self.config = config
        self.hardware = hardware

    def resolve_strategy(self, model_billions: float | None = None) -> Strategy:
        """Resuelve la estrategia: respeta la del usuario o autodetecta."""
        declared = self.config.train.strategy
        if declared and declared != "auto":
            return Strategy(declared)
        return self.hardware.recommend_strategy(model_billions)

    @abc.abstractmethod
    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Valida y divide el dataset en entrenamiento/validación."""

    @abc.abstractmethod
    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        """Ejecuta el entrenamiento y persiste el modelo en output_dir."""

    @abc.abstractmethod
    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        """Evalúa el modelo entrenado y devuelve métricas."""

    def _ensure_output_dir(self) -> Path:
        out = Path(self.config.train.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
