"""Trainer de clasificación de texto con transformers.

Afina un modelo encoder (BERT, RoBERTa, …) con una cabeza de clasificación.
A diferencia del trainer 'llm' (causal LM) y del 'sklearn' (ML clásico), da
clasificación supervisada de alta calidad con modelos preentrenados.

Datos esperados: un texto (`data.text_field`) y una etiqueta
(`data.label_field`).

Extra requerido:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

import json

from blackcode.data.base import Dataset
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult

_log = get_logger("trainers.transformer-classifier")


@register_trainer("transformer-classifier")
class TransformerClassifierTrainer(BaseTrainer):
    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        if len(dataset) == 0:
            raise ValueError("El dataset está vacío.")
        label = self.config.data.label_field
        if not label:
            raise ValueError(
                "data.label_field es obligatorio para 'transformer-classifier'."
            )
        text = self.config.data.text_field
        if text not in dataset[0] or label not in dataset[0]:
            raise KeyError(f"Los registros deben incluir '{text}' y '{label}'.")
        return dataset.split(
            self.config.data.validation_split, self.config.train.seed
        )

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        ensure_extra("llm", "transformers", "datasets", "numpy")
        import numpy as np  # type: ignore
        from datasets import Dataset as HFDataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        tc = self.config.train
        text_field = self.config.data.text_field
        label_field = self.config.data.label_field
        out_dir = str(self._ensure_output_dir())

        labels = sorted({r[label_field] for r in train_ds})
        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = {i: lab for lab, i in label2id.items()}
        _log.info("Clasificación con %d clases: %s", len(labels), labels)

        tokenizer = AutoTokenizer.from_pretrained(tc.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            tc.model,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )

        def to_hf(ds: Dataset):
            hf = HFDataset.from_list(
                [
                    {"text": r[text_field], "label": label2id[r[label_field]]}
                    for r in ds
                ]
            )
            return hf.map(
                lambda batch: tokenizer(
                    batch["text"], truncation=True, max_length=tc.max_seq_length
                ),
                batched=True,
            )

        def compute_metrics(pred):
            preds = np.argmax(pred.predictions, axis=1)
            return {"accuracy": float((preds == pred.label_ids).mean())}

        lr = tc.learning_rate if tc.learning_rate is not None else 2e-5
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=tc.epochs,
            per_device_train_batch_size=tc.batch_size or 16,
            learning_rate=lr,
            seed=tc.seed,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=not tc.options.get("save_optimizer_state", False),
            report_to=[],  # sin telemetría externa
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=to_hf(train_ds),
            eval_dataset=to_hf(val_ds) if len(val_ds) else None,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
        )
        train_output = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        metrics = {"train_loss": float(train_output.training_loss)}
        (self._ensure_output_dir() / "blackcode_metrics.json").write_text(
            json.dumps({"labels": labels, **metrics}, indent=2)
        )
        return TrainResult(
            output_dir=out_dir, metrics=metrics, extra={"labels": labels}
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        ensure_extra("llm", "torch", "transformers")
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if len(val_ds) == 0:
            return {}
        out_dir = self.config.train.output_dir
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        model = AutoModelForSequenceClassification.from_pretrained(out_dir)
        model.eval()
        text_field = self.config.data.text_field
        label_field = self.config.data.label_field
        label2id = model.config.label2id

        correct, total = 0, 0
        with torch.no_grad():
            for row in list(val_ds)[:256]:  # muestra acotada para rapidez
                enc = tokenizer(
                    row[text_field],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.train.max_seq_length,
                )
                pred = int(model(**enc).logits.argmax(-1))
                gold = label2id.get(row[label_field])
                if gold is not None:
                    correct += int(pred == gold)
                    total += 1
        return {"accuracy": round(correct / max(total, 1), 4)}
