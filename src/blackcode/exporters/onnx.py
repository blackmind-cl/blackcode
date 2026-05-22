"""Exportador a ONNX vía la librería `optimum` de Hugging Face.

ONNX permite servir el modelo con ONNX Runtime (CPU o GPU) sin arrastrar
PyTorch en el entorno de inferencia.

Extra requerido:  pip install 'blackcode[onnx]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.exporters.base import BaseExporter, ExportResult
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_exporter

_log = get_logger("exporters.onnx")


@register_exporter("onnx")
class OnnxExporter(BaseExporter):
    def export(self) -> ExportResult:
        ensure_extra("onnx", "optimum.onnxruntime", "transformers")
        from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        src = self.model_dir
        if not src.exists():
            raise FileNotFoundError(f"No existe el modelo entrenado: {src}")

        out = Path(self.config.export.output_dir) / "onnx"
        out.mkdir(parents=True, exist_ok=True)

        _log.info("Exportando %s a ONNX", src)
        model = ORTModelForCausalLM.from_pretrained(str(src), export=True)
        model.save_pretrained(str(out))
        AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
        _log.info("Modelo ONNX guardado en %s", out)
        return ExportResult(format="onnx", path=str(out), extra={})
