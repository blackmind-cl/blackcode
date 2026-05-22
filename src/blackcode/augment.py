"""Aumento de datos sintético, local y sin dependencias.

Genera variantes de los ejemplos aplicando perturbaciones a nivel de palabra
(intercambio y borrado aleatorios). Es la técnica EDA (Easy Data Augmentation)
sin las operaciones que requieren un diccionario de sinónimos, de modo que el
núcleo sigue sin dependencias.

El aumento lo orquesta el Pipeline y se aplica SOLO al split de entrenamiento,
nunca al de validación, para no contaminar la evaluación.
"""

from __future__ import annotations

import random

from blackcode.data.base import Dataset


def _random_swap(words: list[str], n: int, rng: random.Random) -> list[str]:
    """Intercambia `n` veces dos palabras al azar."""
    words = list(words)
    for _ in range(n):
        if len(words) < 2:
            break
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def _random_deletion(words: list[str], p: float, rng: random.Random) -> list[str]:
    """Borra cada palabra con probabilidad `p` (conserva al menos una)."""
    if len(words) <= 1:
        return list(words)
    kept = [w for w in words if rng.random() > p]
    return kept or [rng.choice(words)]


def augment_text(
    text: str, rng: random.Random, swaps: int = 1, delete_p: float = 0.1
) -> str:
    """Devuelve una variante perturbada de `text`."""
    words = text.split()
    if not words:
        return text
    words = _random_swap(words, swaps, rng)
    words = _random_deletion(words, delete_p, rng)
    return " ".join(words)


def augment_dataset(
    dataset: Dataset, factor: int, text_field: str = "text", seed: int = 42
) -> Dataset:
    """Devuelve un dataset con los registros originales más `factor` variantes.

    Por cada registro con `text_field` no vacío se generan `factor` copias con
    el texto perturbado. Los registros sin ese campo se conservan tal cual.
    Determinista por `seed`.
    """
    if factor <= 0:
        return dataset
    rng = random.Random(seed)
    records: list[dict] = []
    for row in dataset.records:
        records.append(row)
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        for _ in range(factor):
            records.append({**row, text_field: augment_text(text, rng)})
    return Dataset(records)
