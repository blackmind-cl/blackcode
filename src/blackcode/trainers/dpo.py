"""Trainer de alineación de preferencias (DPO / ORPO) con trl.

Ajusta un LLM a partir de preferencias: por cada prompt hay una respuesta
preferida ('chosen') y una rechazada ('rejected'). DPO usa un modelo de
referencia; ORPO no (más ligero). Se elige con `train.options.method`.

Datos esperados: campos 'prompt', 'chosen' y 'rejected' (renombrables vía
`data.options`).

Extra requerido:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

import json

from blackcode.data.base import Dataset
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult

_log = get_logger("trainers.dpo")


@register_trainer("dpo")
class PreferenceTrainer(BaseTrainer):
    def _fields(self) -> tuple[str, str, str]:
        opts = self.config.data.options
        return (
            opts.get("prompt_field", "prompt"),
            opts.get("chosen_field", "chosen"),
            opts.get("rejected_field", "rejected"),
        )

    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        if len(dataset) == 0:
            raise ValueError("El dataset está vacío.")
        missing = set(self._fields()) - set(dataset[0])
        if missing:
            raise KeyError(
                f"Los registros de preferencias deben incluir {sorted(missing)}."
            )
        return dataset.split(
            self.config.data.validation_split, self.config.train.seed
        )

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        ensure_extra("llm", "torch", "datasets", "transformers", "trl")
        import torch  # type: ignore
        from datasets import Dataset as HFDataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        method = self.config.train.options.get("method", "dpo").lower()
        if method not in ("dpo", "orpo"):
            raise ValueError("train.options.method debe ser 'dpo' u 'orpo'.")
        if method == "orpo":
            from trl import ORPOConfig as PrefConfig  # type: ignore
            from trl import ORPOTrainer as PrefTrainer  # type: ignore
        else:
            from trl import DPOConfig as PrefConfig  # type: ignore
            from trl import DPOTrainer as PrefTrainer  # type: ignore

        tc = self.config.train
        out_dir = str(self._ensure_output_dir())
        prompt_f, chosen_f, rejected_f = self._fields()
        _log.info("Alineación de preferencias con método '%s'", method)

        tokenizer = AutoTokenizer.from_pretrained(tc.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            tc.model, torch_dtype=torch.bfloat16
        )

        def to_hf(ds: Dataset):
            return HFDataset.from_list(
                [
                    {
                        "prompt": r[prompt_f],
                        "chosen": r[chosen_f],
                        "rejected": r[rejected_f],
                    }
                    for r in ds
                ]
            )

        lr = tc.learning_rate if tc.learning_rate is not None else 5e-6
        args = PrefConfig(
            output_dir=out_dir,
            num_train_epochs=tc.epochs,
            per_device_train_batch_size=tc.batch_size or 1,
            learning_rate=lr,
            seed=tc.seed,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=not tc.options.get("save_optimizer_state", False),
            report_to=[],  # sin telemetría externa
        )
        # DPO necesita un modelo de referencia; ORPO no lo usa.
        ref_kwargs = {"ref_model": None} if method == "dpo" else {}
        trainer = PrefTrainer(
            model,
            args=args,
            train_dataset=to_hf(train_ds),
            eval_dataset=to_hf(val_ds) if len(val_ds) else None,
            processing_class=tokenizer,
            **ref_kwargs,
        )
        train_output = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        metrics = {"train_loss": float(train_output.training_loss)}
        (self._ensure_output_dir() / "blackcode_metrics.json").write_text(
            json.dumps({"method": method, **metrics}, indent=2)
        )
        return TrainResult(
            output_dir=out_dir, metrics=metrics, extra={"method": method}
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        # La calidad de la alineación se mide mejor con juicio humano o un
        # modelo de recompensa; no hay una métrica escalar simple y fiable.
        _log.info("El trainer 'dpo' no expone una métrica de evaluación automática.")
        return {}
