"""Trainer de embeddings con sentence-transformers.

Afina un modelo de embeddings (bi-encoder) sobre pares de textos relacionados
(ancla / positivo) con MultipleNegativesRankingLoss — el caso de uso más común
para búsqueda semántica y RAG.

Datos esperados: cada registro tiene un texto ancla (`data.text_field`) y un
texto positivo (`data.options['positive_field']`, `positive` por defecto).

Extra requerido:  pip install 'blackcode[embeddings]'
"""

from __future__ import annotations

from blackcode.data.base import Dataset
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_trainer
from blackcode.trainers._hf import common_training_kwargs, save_and_record
from blackcode.trainers.base import BaseTrainer, TrainResult

_log = get_logger("trainers.embeddings")


@register_trainer("embeddings")
class EmbeddingsTrainer(BaseTrainer):
    def _positive_field(self) -> str:
        return self.config.data.options.get("positive_field", "positive")

    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        if len(dataset) == 0:
            raise ValueError("El dataset está vacío.")
        anchor = self.config.data.text_field
        missing = {anchor, self._positive_field()} - set(dataset[0])
        if missing:
            raise KeyError(
                f"Los registros deben incluir los campos {sorted(missing)}. "
                "Define data.text_field y data.options.positive_field."
            )
        return dataset.split(
            self.config.data.validation_split, self.config.train.seed
        )

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        ensure_extra("embeddings", "sentence_transformers", "datasets")
        from datasets import Dataset as HFDataset  # type: ignore
        from sentence_transformers import (  # type: ignore
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import (  # type: ignore
            MultipleNegativesRankingLoss,
        )

        tc = self.config.train
        out_dir = str(self._ensure_output_dir())
        anchor, positive = self.config.data.text_field, self._positive_field()

        _log.info("Afinando embeddings sobre %s", tc.model)
        model = SentenceTransformer(tc.model)
        loss = MultipleNegativesRankingLoss(model)

        def to_hf(ds: Dataset):
            return HFDataset.from_list(
                [{"anchor": r[anchor], "positive": r[positive]} for r in ds]
            )

        lr = tc.learning_rate if tc.learning_rate is not None else 2e-5
        args = SentenceTransformerTrainingArguments(
            **common_training_kwargs(
                tc, out_dir, batch_size=tc.batch_size or 16, learning_rate=lr
            ),
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=to_hf(train_ds),
            eval_dataset=to_hf(val_ds) if len(val_ds) else None,
            loss=loss,
        )
        train_output = trainer.train()
        metrics = save_and_record(
            trainer, None, out_dir, train_output, {"base_model": tc.model}
        )
        return TrainResult(
            output_dir=out_dir, metrics=metrics, extra={"base_model": tc.model}
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        ensure_extra("embeddings", "sentence_transformers")
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sentence_transformers.util import cos_sim  # type: ignore

        if len(val_ds) == 0:
            return {}
        model = SentenceTransformer(self.config.train.output_dir)
        anchor, positive = self.config.data.text_field, self._positive_field()
        rows = list(val_ds)[:128]  # muestra acotada para rapidez
        emb_a = model.encode([r[anchor] for r in rows])
        emb_p = model.encode([r[positive] for r in rows])
        sims = [float(cos_sim(emb_a[i], emb_p[i])) for i in range(len(rows))]
        return {"cosine_similarity": round(sum(sims) / len(sims), 4)}
