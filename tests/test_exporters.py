"""Tests del sistema de exportadores (registro + orquestación).

No requieren torch ni optimum: ejercitan el registro y `export_model` con un
exportador falso en memoria.
"""

import pytest

from blackcode.config import RunConfig
from blackcode.exporters import export_model
from blackcode.exporters.base import BaseExporter, ExportResult
from blackcode.registry import get_exporter, list_exporters, register_exporter


@register_exporter("fake-export")
class _FakeExporter(BaseExporter):
    def export(self) -> ExportResult:
        return ExportResult(
            format="fake-export",
            path="/tmp/fake",
            extra={"name": self.config.name},
        )


def _config(**export_block) -> RunConfig:
    raw = {
        "name": "exp-test",
        "data": {"path": "./d.jsonl"},
        "train": {"trainer": "llm", "model": "m"},
    }
    if export_block:
        raw["export"] = export_block
    return RunConfig.from_dict(raw)


def test_builtin_exporters_are_registered():
    assert {"merge-lora", "gguf", "onnx"} <= set(list_exporters())


def test_get_unknown_exporter_lists_available():
    with pytest.raises(KeyError, match="gguf"):
        get_exporter("inexistente")


def test_register_duplicate_exporter_raises():
    with pytest.raises(ValueError):

        @register_exporter("gguf")
        class _Dup:  # pragma: no cover
            pass


def test_export_model_runs_explicit_format():
    results = export_model(_config(), ["fake-export"])
    assert len(results) == 1
    assert results[0].format == "fake-export"
    assert results[0].extra["name"] == "exp-test"


def test_export_model_falls_back_to_config_formats():
    results = export_model(_config(formats=["fake-export"]))
    assert results[0].path == "/tmp/fake"


def test_export_model_explicit_formats_override_config():
    results = export_model(_config(formats=["gguf"]), ["fake-export"])
    assert results[0].format == "fake-export"


def test_export_model_without_any_format_raises():
    with pytest.raises(ValueError, match="formato"):
        export_model(_config())
