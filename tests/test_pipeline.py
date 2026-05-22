"""Tests del orquestador Pipeline con un trainer de prueba en memoria.

Registramos un trainer falso para ejercitar el flujo completo (datos ->
prepare -> train -> evaluate) sin torch ni sklearn.
"""

import json
from pathlib import Path

import pytest

from blackcode.config import RunConfig
from blackcode.data.base import Dataset
from blackcode.pipeline import Pipeline
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult


@register_trainer("fake-test-trainer")
class _FakeTrainer(BaseTrainer):
    def prepare(self, dataset: Dataset):
        return dataset.split(
            self.config.data.validation_split, self.config.train.seed
        )

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        return TrainResult(
            output_dir=self.config.train.output_dir,
            metrics={"train_loss": 0.5, "n_train": float(len(train_ds))},
            extra={},
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        return {"loss": 0.25, "n_val": float(len(val_ds))}


def _config(data_path: Path, augment: int = 0) -> RunConfig:
    return RunConfig.from_dict(
        {
            "name": "pipeline-test",
            "data": {
                "path": str(data_path),
                "validation_split": 0.2,
                "augment": augment,
            },
            "train": {"trainer": "fake-test-trainer", "model": "none"},
        }
    )


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"r{i}"}) for i in range(20)))
    return f


def test_pipeline_run_trains_and_evaluates(dataset_file: Path):
    result = Pipeline(_config(dataset_file)).run()
    assert result.train.metrics["train_loss"] == 0.5
    assert result.train.metrics["n_train"] == 16  # 20 * (1 - 0.2)
    assert result.eval_metrics == {"loss": 0.25, "n_val": 4.0}
    assert result.hardware is not None


def test_pipeline_run_can_skip_eval(dataset_file: Path):
    result = Pipeline(_config(dataset_file)).run(do_eval=False)
    assert result.eval_metrics == {}


def test_pipeline_augments_only_the_training_split(dataset_file: Path):
    result = Pipeline(_config(dataset_file, augment=2)).run()
    # 16 de entrenamiento -> 16 * (1 + 2) = 48; la validación queda intacta.
    assert result.train.metrics["n_train"] == 48
    assert result.eval_metrics["n_val"] == 4.0


def test_pipeline_evaluate_only_does_not_train(dataset_file: Path):
    metrics = Pipeline(_config(dataset_file)).evaluate()
    assert metrics == {"loss": 0.25, "n_val": 4.0}


def test_pipeline_rejects_unknown_trainer(dataset_file: Path):
    cfg = RunConfig.from_dict(
        {
            "data": {"path": str(dataset_file)},
            "train": {"trainer": "no-existe", "model": "m"},
        }
    )
    with pytest.raises(KeyError, match="no-existe"):
        Pipeline(cfg).run()
