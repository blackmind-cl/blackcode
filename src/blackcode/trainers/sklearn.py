"""Trainer de ML clásico (scikit-learn).

Cubre el caso "no quiero un LLM, solo un buen clasificador/regresor sobre mis
datos". Funciona perfectamente en CPU. Soporta texto (TF-IDF) y tabular.

Extra requerido:  pip install 'blackcode[sklearn]'
"""

from __future__ import annotations

import json
import pickle

from blackcode.data.base import Dataset
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult

_log = get_logger("trainers.sklearn")


@register_trainer("sklearn")
class SklearnTrainer(BaseTrainer):
    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        label = self.config.data.label_field
        if not label:
            raise ValueError("data.label_field es obligatorio para 'sklearn'.")
        if len(dataset) == 0 or label not in dataset[0]:
            raise KeyError(f"Los registros deben incluir la etiqueta '{label}'.")
        return dataset.split(self.config.data.validation_split, self.config.train.seed)

    def _build_estimator(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import (  # type: ignore
            LogisticRegression,
            Ridge,
        )
        from sklearn.pipeline import Pipeline as SkPipeline  # type: ignore

        opts = self.config.train.options
        model_name = self.config.train.model.lower()
        is_text = self.config.data.label_field and self.config.data.text_field

        estimators = {
            "logreg": LogisticRegression(max_iter=opts.get("max_iter", 1000)),
            "ridge": Ridge(alpha=opts.get("alpha", 1.0)),
        }
        est = estimators.get(model_name, estimators["logreg"])
        if is_text:
            return SkPipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=opts.get("max_features", 20000))),
                    ("clf", est),
                ]
            )
        return est

    def _xy(self, ds: Dataset):
        text_field = self.config.data.text_field
        label = self.config.data.label_field
        X = [r[text_field] for r in ds]
        y = [r[label] for r in ds]
        return X, y

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        ensure_extra("sklearn", "sklearn")
        from sklearn.metrics import accuracy_score  # type: ignore

        if self.config.train.resume:
            _log.warning(
                "El trainer 'sklearn' no soporta reanudar; se entrena desde cero."
            )
        out = self._ensure_output_dir()
        model = self._build_estimator()
        Xtr, ytr = self._xy(train_ds)
        _log.info("Ajustando estimador '%s' con %d muestras", self.config.train.model, len(Xtr))
        model.fit(Xtr, ytr)

        metrics = {}
        if len(val_ds):
            Xv, yv = self._xy(val_ds)
            try:
                metrics["accuracy"] = round(accuracy_score(yv, model.predict(Xv)), 4)
            except ValueError:
                pass  # objetivo continuo: la accuracy no aplica

        with (out / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh)
        (out / "blackcode_metrics.json").write_text(json.dumps(metrics, indent=2))
        return TrainResult(
            output_dir=str(out), metrics=metrics, extra={"estimator": self.config.train.model}
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        ensure_extra("sklearn", "sklearn")
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            f1_score,
        )

        if len(val_ds) == 0:
            return {}
        with (self._ensure_output_dir() / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        Xv, yv = self._xy(val_ds)
        pred = model.predict(Xv)
        result: dict[str, float] = {}
        if "accuracy" in metrics or not metrics:
            result["accuracy"] = round(accuracy_score(yv, pred), 4)
        if "f1" in metrics:
            result["f1"] = round(f1_score(yv, pred, average="weighted"), 4)
        return result
