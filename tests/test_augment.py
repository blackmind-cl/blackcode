"""Tests del aumento de datos sintético (sin dependencias pesadas)."""

import random

from blackcode.augment import augment_dataset, augment_text
from blackcode.data.base import Dataset


def test_augment_text_returns_a_string():
    out = augment_text("una frase con varias palabras", random.Random(1))
    assert isinstance(out, str) and out


def test_augment_text_empty_stays_empty():
    assert augment_text("", random.Random(1)) == ""


def test_augment_factor_zero_is_a_noop():
    ds = Dataset([{"text": "hola mundo"}])
    assert augment_dataset(ds, 0) is ds


def test_augment_factor_multiplies_records():
    ds = Dataset([{"text": f"palabra uno dos tres {i}"} for i in range(5)])
    out = augment_dataset(ds, 2, "text", seed=1)
    assert len(out) == 15  # 5 originales + 5 * 2 variantes


def test_augment_is_deterministic_by_seed():
    ds = Dataset([{"text": "una frase de prueba bastante larga"}])
    a = augment_dataset(ds, 3, "text", seed=7)
    b = augment_dataset(ds, 3, "text", seed=7)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_augment_keeps_original_record_first():
    ds = Dataset([{"text": "texto original aqui"}])
    out = augment_dataset(ds, 2, "text", seed=1)
    assert out[0]["text"] == "texto original aqui"


def test_augment_skips_records_without_text_field():
    ds = Dataset([{"other": "x"}, {"text": "un texto cualquiera aqui"}])
    out = augment_dataset(ds, 2, "text", seed=1)
    # 1 sin texto (sin variantes) + 1 con texto (+2 variantes) = 4
    assert len(out) == 4


def test_augment_preserves_other_fields():
    ds = Dataset([{"text": "hola que tal", "label": "saludo"}])
    out = augment_dataset(ds, 3, "text", seed=1)
    assert all(r["label"] == "saludo" for r in out)
