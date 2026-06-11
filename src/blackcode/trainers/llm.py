"""Trainer de fine-tuning de LLMs open source.

Envuelve el ecosistema Hugging Face (transformers + peft + trl). Selecciona
automáticamente QLoRA / LoRA / full según la VRAM detectada. Todas las
dependencias pesadas se importan dentro de los métodos para que el resto del
paquete funcione sin ellas instaladas.

Extra requerido:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

import re

from blackcode.data.base import Dataset
from blackcode.hardware import Strategy
from blackcode.install import ensure_extra
from blackcode.log import get_logger, track
from blackcode.registry import register_trainer
from blackcode.trainers._hf import (
    common_training_kwargs,
    load_tokenizer,
    save_and_record,
)
from blackcode.trainers.base import BaseTrainer, TrainResult

_log = get_logger("trainers.llm")

_TARGET_EFFECTIVE_BATCH = 16


def _default_learning_rate(strategy: Strategy) -> float:
    """LR sensata según estrategia: 2e-5 para full (todos los pesos), 2e-4 para LoRA."""
    if strategy in (Strategy.FULL, Strategy.CPU):
        return 2e-5
    return 2e-4


def _guess_billions(model_id: str) -> float | None:
    """Extrae el tamaño en miles de millones de params del nombre del modelo."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    return float(m.group(1)) if m else None


def _auto_batch_config(
    strategy: Strategy,
    vram_gb: float,
    declared_batch_size: int,
    declared_accum: int | None,
) -> tuple[int, int]:
    """Calcula (batch por dispositivo, pasos de acumulación de gradiente).

    Respeta los valores que declare el usuario; si no, autodetecta. El batch
    por dispositivo se elige según VRAM y estrategia para no agotar memoria;
    la acumulación de gradiente compensa para alcanzar un batch efectivo
    estable (~16) sea cual sea el hardware.
    """
    if declared_batch_size and declared_batch_size > 0:
        batch_size = declared_batch_size
    elif strategy in (Strategy.QLORA, Strategy.CPU):
        batch_size = 1
    elif vram_gb >= 48:
        batch_size = 8
    elif vram_gb >= 24:
        batch_size = 4
    elif vram_gb >= 12:
        batch_size = 2
    else:
        batch_size = 1
    if declared_accum and declared_accum > 0:
        accum = declared_accum
    else:
        accum = max(1, _TARGET_EFFECTIVE_BATCH // batch_size)
    return batch_size, accum


@register_trainer("llm")
class LLMTrainer(BaseTrainer):
    def prepare(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        if len(dataset) == 0:
            raise ValueError("El dataset está vacío.")
        field = self.config.data.text_field
        if field not in dataset[0]:
            sample = dataset[0]
            if "instruction" in sample and "output" in sample:
                raise ValueError(
                    "Los registros tienen 'instruction'/'output' (típico del "
                    f"JSONL de `generate-qa`) pero el trainer espera '{field}'. "
                    "Añade a tu blackcode.yaml, dentro de `data:`, esta línea:\n"
                    '    instruction_template: "### Pregunta: {instruction}'
                    '\\n### Respuesta: {output}"'
                )
            raise ValueError(
                f"Los registros no contienen el campo de texto '{field}'. "
                "Define data.text_field o data.instruction_template."
            )
        return dataset.split(self.config.data.validation_split, self.config.train.seed)

    def train(self, train_ds: Dataset, val_ds: Dataset) -> TrainResult:
        ensure_extra("llm", "torch", "datasets", "transformers", "trl")
        import torch  # type: ignore
        from datasets import Dataset as HFDataset  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore

        tc = self.config.train
        out_dir = str(self._ensure_output_dir())
        strategy = self.resolve_strategy(_guess_billions(tc.model))
        field = self.config.data.text_field

        quant_config = None
        if strategy == Strategy.QLORA:
            from transformers import BitsAndBytesConfig  # type: ignore

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = load_tokenizer(tc.model)

        model = AutoModelForCausalLM.from_pretrained(
            tc.model,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map="auto" if self.hardware.accelerator.value != "cpu" else None,
        )

        peft_config = None
        if strategy in (Strategy.LORA, Strategy.QLORA):
            from peft import LoraConfig  # type: ignore

            peft_config = LoraConfig(
                r=tc.options.get("lora_r", 16),
                lora_alpha=tc.options.get("lora_alpha", 32),
                lora_dropout=tc.options.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )

        batch_size, accum = _auto_batch_config(
            strategy,
            self.hardware.total_vram_gb,
            tc.batch_size,
            tc.options.get("gradient_accumulation_steps"),
        )
        _log.info(
            "Estrategia: %s · batch/dispositivo=%d · grad_accum=%d · batch efectivo=%d",
            strategy.value,
            batch_size,
            accum,
            batch_size * accum,
        )
        # Multi-GPU: si hay más de una GPU se activa FSDP automáticamente.
        # El entrenamiento distribuido debe lanzarse con `accelerate launch`.
        fsdp = ""
        if self.hardware.is_multi_gpu:
            fsdp = "full_shard auto_wrap"
            _log.info(
                "Multi-GPU detectado (%d GPUs): FSDP activado. Lanza con "
                "`accelerate launch`.",
                len(self.hardware.gpus),
            )
        lr = (
            tc.learning_rate
            if tc.learning_rate is not None
            else _default_learning_rate(strategy)
        )
        if tc.learning_rate is None:
            _log.info("Learning rate auto para estrategia %s: %g", strategy.value, lr)
        args = SFTConfig(
            **common_training_kwargs(
                tc, out_dir, batch_size=batch_size, learning_rate=lr
            ),
            gradient_accumulation_steps=accum,
            bf16=self.hardware.accelerator.value == "cuda",
            fsdp=fsdp,
            dataset_text_field=field,
            max_length=tc.max_seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=HFDataset.from_list(list(train_ds)),
            eval_dataset=HFDataset.from_list(list(val_ds)) if len(val_ds) else None,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        if tc.resume:
            _log.info("Reanudando desde el último checkpoint en %s", out_dir)
        train_output = trainer.train(resume_from_checkpoint=tc.resume or None)
        metrics = save_and_record(
            trainer, tokenizer, out_dir, train_output, {"strategy": strategy.value}
        )
        return TrainResult(
            output_dir=out_dir,
            metrics=metrics,
            extra={"strategy": strategy.value, "base_model": tc.model},
        )

    def evaluate(self, val_ds: Dataset, metrics: list[str]) -> dict[str, float]:
        ensure_extra("llm", "torch", "transformers")
        import math

        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if len(val_ds) == 0:
            return {}
        out_dir = self.config.train.output_dir
        tok = AutoTokenizer.from_pretrained(out_dir)
        model = AutoModelForCausalLM.from_pretrained(out_dir)
        model.eval()
        field = self.config.data.text_field

        total_loss, n = 0.0, 0
        with torch.no_grad():
            for row in track(list(val_ds)[:64], "Evaluando"):  # muestra acotada
                enc = tok(
                    row[field],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.train.max_seq_length,
                )
                out = model(**enc, labels=enc["input_ids"])
                total_loss += float(out.loss)
                n += 1
        avg = total_loss / max(n, 1)
        results = {"loss": round(avg, 4)}
        if "perplexity" in metrics:
            results["perplexity"] = round(math.exp(avg), 4)
        return results
