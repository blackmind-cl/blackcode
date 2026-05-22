"""Tests de carga de datos local-first (sin dependencias pesadas)."""

import json
from pathlib import Path

import pytest

from blackcode.config import DataConfig
from blackcode.data.base import load_dataset


def test_load_json_list(tmp_path: Path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
    ds = load_dataset(DataConfig(path=str(f)))
    assert len(ds) == 2 and ds[1]["text"] == "b"


def test_load_json_wrapped_in_data_key(tmp_path: Path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"data": [{"text": "a"}]}))
    assert len(load_dataset(DataConfig(path=str(f)))) == 1


def test_load_csv(tmp_path: Path):
    f = tmp_path / "d.csv"
    f.write_text("text,label\nhola,1\nmundo,0\n")
    ds = load_dataset(DataConfig(path=str(f)))
    assert len(ds) == 2 and ds[0]["label"] == "1"


def test_load_tsv_uses_tab_delimiter(tmp_path: Path):
    f = tmp_path / "d.tsv"
    f.write_text("text\tlabel\nhola\t1\n")
    ds = load_dataset(DataConfig(path=str(f)))
    assert ds[0]["text"] == "hola" and ds[0]["label"] == "1"


def test_max_samples_truncates(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"text": str(i)}) for i in range(50)))
    ds = load_dataset(DataConfig(path=str(f), max_samples=10))
    assert len(ds) == 10


def test_instruction_template_applied_on_load(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"instruction": "hola", "output": "mundo"}))
    ds = load_dataset(
        DataConfig(path=str(f), instruction_template="{instruction} -> {output}")
    )
    assert ds[0]["text"] == "hola -> mundo"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(DataConfig(path=str(tmp_path / "nope.jsonl")))


def test_invalid_jsonl_reports_line_number(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"text": "ok"}\nNOT-JSON\n')
    with pytest.raises(ValueError, match=":2"):
        load_dataset(DataConfig(path=str(f)))
