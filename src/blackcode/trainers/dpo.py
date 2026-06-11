"""Trainer de alineación de preferencias (DPO / ORPO) con trl.

Ajusta un LLM a partir de preferencias: por cada prompt hay una respuesta
preferida ('chosen') y una rechazada ('rejected'). DPO usa un modelo de
referencia; ORPO no (más ligero). Se elige con `train.options.method`.

Datos esperados: campos 'prompt', 'chosen' y 'rejected' (renombrables vía
`data.options`).

Extra requerido:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

from blackcode.data.base import Dataset
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_trainer
from blackcode.trainers._hf import (
    common_training_kwargs,
    load_tokenizer,
    save_and_record,
)
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
        from transformers import AutoModelForCausalLM  # type: ignore

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

        tokenizer = load_tokenizer(tc.model)
        model = AutoModelForCausalLM.from_pretrained(tc.model, dtype=torch.bfloat16)

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
            **common_training_kwargs(
                tc, out_dir, batch_size=tc.batch_size or 1, learning_rate=lr
            ),
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
        metrics = save_and_record(
            trainer, tokenizer, out_dir, train_output, {"method": method}
        )
        return TrainResult(
            output_dir=out_dir, metrics=metrics, extra={"method": method}
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        # La calidad de la alineación se mide mejor con juicio humano o un
        # modelo de recompensa; no hay una métrica escalar simple y fiable.
        _log.info("El trainer 'dpo' no expone una métrica de evaluación automática.")
        return {}
