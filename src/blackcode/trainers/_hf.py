"""Utilidades compartidas por los trainers basados en Hugging Face.

Los trainers `llm`, `dpo`, `transformer-classifier` y `embeddings` repiten el
mismo esqueleto: cargar tokenizer, construir TrainingArguments con la misma
política de guardado/privacidad, entrenar, guardar y registrar métricas. Este
módulo concentra ese esqueleto para que un cambio (p. ej. la política de
checkpoints o `report_to`) se haga en UN lugar.

Sin imports pesados a nivel de módulo: cada helper importa lo que necesita
dentro de la función, después de que el trainer llamó a `ensure_extra`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tokenizer(model_id: str):
    """Carga el tokenizer con fallback de `pad_token`.

    Los modelos causales (Llama, Qwen, …) suelen no definir pad_token; sin él
    el data collator falla. Usar eos como pad es la convención estándar.
    """
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def common_training_kwargs(
    tc, out_dir: str, *, batch_size: int, learning_rate: float
) -> dict[str, Any]:
    """kwargs comunes a SFTConfig / DPOConfig / ORPOConfig / TrainingArguments.

    Concentra la política compartida del proyecto:
      - `report_to=[]`: sin telemetría externa (privacidad, no negociable).
      - `save_only_model` por defecto: el estado del optimizador pesa 2-3× el
        modelo y solo sirve para `--resume`; se activa con
        `options.save_optimizer_state: true`.
      - un único checkpoint por epoch (`save_total_limit=1`).
    """
    return dict(
        output_dir=out_dir,
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        seed=tc.seed,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=not tc.options.get("save_optimizer_state", False),
        report_to=[],  # sin telemetría externa
    )


def save_and_record(
    trainer, tokenizer, out_dir: str, train_output, extra: dict[str, Any]
) -> dict[str, float]:
    """Guarda modelo (y tokenizer), escribe `blackcode_metrics.json`.

    `extra` son los metadatos propios de cada trainer (estrategia, método,
    etiquetas, …) que acompañan a las métricas en el JSON. Devuelve las
    métricas de entrenamiento para el `TrainResult`.
    """
    trainer.save_model(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    metrics = {"train_loss": float(train_output.training_loss)}
    (Path(out_dir) / "blackcode_metrics.json").write_text(
        json.dumps({**extra, **metrics}, indent=2)
    )
    return metrics
