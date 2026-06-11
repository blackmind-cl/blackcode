"""`blackcode doctor` — reporte de hardware y estrategia recomendada."""

from __future__ import annotations

import argparse


def _run(_: argparse.Namespace) -> int:
    from blackcode.hardware import detect_hardware

    hw = detect_hardware()
    print("Blackcode · reporte de hardware")
    print("-" * 40)
    print(f"  SO            : {hw.os}")
    print(f"  Python        : {hw.python}")
    print(f"  CPUs          : {hw.cpu_count}")
    print(f"  RAM           : {hw.ram_gb} GB")
    print(f"  Acelerador    : {hw.accelerator.value}")
    if hw.gpus:
        for g in hw.gpus:
            print(f"  GPU {g.index}        : {g.name} ({g.total_memory_gb} GB)")
        print(f"  VRAM total    : {hw.total_vram_gb} GB")
        if hw.is_multi_gpu:
            print(f"  Multi-GPU     : sí ({len(hw.gpus)}) · FSDP automático")
    print("-" * 40)
    print(f"  Estrategia recomendada (sin tamaño de modelo): "
          f"{hw.recommend_strategy().value}")
    print("  Para un modelo de 7B :", hw.recommend_strategy(7).value)
    return 0


def register(sub) -> None:
    sub.add_parser("doctor", help="Reporte de hardware").set_defaults(func=_run)
