"""Carga de datos local-first.

Soporta JSONL, JSON y CSV con la librería estándar (cero dependencias).
El formato `hf` (Hugging Face datasets) se importa de forma perezosa solo
si se solicita. Los datos nunca salen de la máquina.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from blackcode.config import DataConfig


@dataclass
class Dataset:
    """Contenedor mínimo e iterable de registros (lista de dicts)."""

    records: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]

    def split(self, validation_fraction: float, seed: int = 42):
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_split debe estar en [0, 1).")
        rows = list(self.records)
        random.Random(seed).shuffle(rows)
        # Garantiza al menos 1 registro en entrenamiento si hay datos; un
        # validation_split que dejara el entrenamiento vacío sería inútil.
        cut = max(1, int(len(rows) * (1.0 - validation_fraction))) if rows else 0
        return Dataset(rows[:cut]), Dataset(rows[cut:])

    def apply_template(self, template: str, field_name: str = "text") -> "Dataset":
        """Renderiza una plantilla de instrucción por registro.

        Ejemplo de template: '### Instrucción:\\n{instruction}\\n### Respuesta:\\n{output}'
        """
        out = []
        for r in self.records:
            try:
                rendered = template.format(**r)
            except KeyError as exc:
                raise KeyError(
                    f"La plantilla referencia el campo {exc} ausente en un registro."
                ) from exc
            out.append({**r, field_name: rendered})
        return Dataset(out)


def _detect_format(path: Path, declared: str) -> str:
    if declared != "auto":
        return declared
    suffix = path.suffix.lower()
    return {
        ".jsonl": "jsonl",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "csv",
    }.get(suffix, "jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en {path}:{i}: {exc}") from exc
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Acepta {"data": [...]} o un único registro.
        data = data.get("data", [data])
    if not isinstance(data, list):
        raise ValueError(f"Se esperaba una lista de registros en {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def _read_hf(spec: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "El formato 'hf' requiere el extra: pip install 'blackcode[llm]'"
        ) from exc
    name, _, split = spec.partition(":")
    ds = hf_load(name, split=split or "train")
    return [dict(row) for row in ds]


def load_dataset(cfg: DataConfig) -> Dataset:
    """Carga un dataset según la configuración y aplica plantilla/recorte."""
    fmt = cfg.format
    if fmt == "hf":
        records = _read_hf(cfg.path)
    else:
        path = Path(cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el dataset: {path}")
        fmt = _detect_format(path, cfg.format)
        reader = {"jsonl": _read_jsonl, "json": _read_json, "csv": _read_csv}[fmt]
        records = reader(path)

    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]

    ds = Dataset(records)
    if cfg.instruction_template:
        ds = ds.apply_template(cfg.instruction_template, cfg.text_field)
    return ds
