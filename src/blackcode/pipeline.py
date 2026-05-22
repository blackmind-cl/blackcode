"""Orquestador de alto nivel.

Conecta las etapas (carga de datos -> trainer -> evaluación) usando solo las
abstracciones del core. Es agnóstico al backend concreto: resuelve el trainer
por nombre desde el registro.
"""

from __future__ import annotations

from dataclasses import dataclass

from blackcode.augment import augment_dataset
from blackcode.config import RunConfig
from blackcode.data.base import load_dataset
from blackcode.hardware import HardwareProfile, detect_hardware
from blackcode.log import get_logger
from blackcode.registry import get_trainer
from blackcode.trainers.base import TrainResult

_log = get_logger("pipeline")


@dataclass
class PipelineResult:
    train: TrainResult
    eval_metrics: dict[str, float]
    hardware: HardwareProfile


class Pipeline:
    """Punto de entrada programático equivalente a `blackcode train`."""

    def __init__(self, config: RunConfig, hardware: HardwareProfile | None = None):
        self.config = config
        self.hardware = hardware or detect_hardware()

    def _prepare(self):
        """Resuelve el trainer, carga los datos y divide train/validación."""
        cfg = self.config
        _log.info(
            "Acelerador: %s · trainer: %s",
            self.hardware.accelerator.value,
            cfg.train.trainer,
        )
        trainer = get_trainer(cfg.train.trainer)(cfg, self.hardware)
        dataset = load_dataset(cfg.data)
        _log.info("Datos cargados: %d registros desde %s", len(dataset), cfg.data.path)
        train_ds, val_ds = trainer.prepare(dataset)
        _log.info(
            "División: %d entrenamiento / %d validación", len(train_ds), len(val_ds)
        )
        return trainer, train_ds, val_ds

    def run(self, do_eval: bool = True) -> PipelineResult:
        trainer, train_ds, val_ds = self._prepare()
        if self.config.data.augment > 0:
            before = len(train_ds)
            train_ds = augment_dataset(
                train_ds,
                self.config.data.augment,
                self.config.data.text_field,
                self.config.train.seed,
            )
            _log.info(
                "Aumento de datos: %d -> %d registros de entrenamiento",
                before,
                len(train_ds),
            )
        _log.info("Iniciando entrenamiento…")
        result = trainer.train(train_ds, val_ds)
        _log.info("Entrenamiento finalizado · modelo en %s", result.output_dir)

        eval_metrics: dict[str, float] = {}
        if do_eval:
            _log.info("Evaluando…")
            eval_metrics = trainer.evaluate(val_ds, self.config.evaluate.metrics)

        return PipelineResult(
            train=result,
            eval_metrics=eval_metrics,
            hardware=self.hardware,
        )

    def evaluate(self) -> dict[str, float]:
        """Evalúa un modelo ya entrenado en output_dir, sin reentrenar.

        Equivalente a `blackcode eval`: reutiliza la misma división de validación
        (determinista por seed) que se usó al entrenar.
        """
        trainer, _, val_ds = self._prepare()
        _log.info("Evaluando modelo en %s…", self.config.train.output_dir)
        return trainer.evaluate(val_ds, self.config.evaluate.metrics)
