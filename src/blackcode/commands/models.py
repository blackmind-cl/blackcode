"""`blackcode models` — catálogo de modelos recomendados según la VRAM."""

from __future__ import annotations

import argparse


def _run(args: argparse.Namespace) -> int:
    from blackcode.catalog import recommend_models
    from blackcode.hardware import detect_hardware

    hw = detect_hardware()
    print(
        f"Modelos recomendados · VRAM detectada: {hw.total_vram_gb} GB · "
        f"acelerador: {hw.accelerator.value}"
    )
    print("-" * 60)
    for model, strategy in recommend_models(hw, args.task):
        print(f"  {model.id}  ({model.params_billions}B · {model.task})")
        print(f"    estrategia recomendada: {strategy} — {model.description}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("models", help="Catálogo de modelos por VRAM")
    p.add_argument(
        "--task",
        choices=["llm", "embeddings", "classifier"],
        help="Filtrar por tipo de tarea",
    )
    p.set_defaults(func=_run)
