"""Resumen y comparación de métricas.

El motor delega el cálculo de métricas en cada trainer (que conoce su
dominio). Este módulo solo formatea y compara resultados para mostrarlos de
forma legible y para soportar el caso "antes vs. después del fine-tuning".
"""

from __future__ import annotations


def summarize(metrics: dict[str, float]) -> str:
    if not metrics:
        return "  (sin métricas)"
    width = max(len(k) for k in metrics)
    return "\n".join(f"  {k.ljust(width)} : {v}" for k, v in metrics.items())


def compare(before: dict[str, float], after: dict[str, float]) -> str:
    """Tabla comparativa simple métrica a métrica."""
    keys = sorted(set(before) | set(after))
    if not keys:
        return "  (sin métricas para comparar)"
    lines = ["  métrica          antes      después    Δ"]
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        delta = "" if b is None or a is None else f"{a - b:+.4f}"
        lines.append(
            f"  {k:<16} {('-' if b is None else f'{b:.4f}'):<10} "
            f"{('-' if a is None else f'{a:.4f}'):<10} {delta}"
        )
    return "\n".join(lines)
