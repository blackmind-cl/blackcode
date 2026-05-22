"""Exportador: fusiona los adaptadores LoRA/QLoRA en el modelo base.

Tras un fine-tuning con LoRA, lo que se guarda son los adaptadores más una
referencia al modelo base. Este exportador produce un modelo único y
autónomo (pesos ya fusionados), listo para servir o para convertir luego a
GGUF / ONNX.

Extra requerido:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.exporters.base import BaseExporter, ExportResult
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_exporter

_log = get_logger("exporters.merge-lora")


@register_exporter("merge-lora")
class MergeLoraExporter(BaseExporter):
    def export(self) -> ExportResult:
        ensure_extra("llm", "peft", "transformers")
        from peft import AutoPeftModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        src = self.model_dir
        if not (src / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No hay adaptadores LoRA en {src} (falta adapter_config.json). "
                "'merge-lora' solo aplica a modelos entrenados con estrategia "
                "lora o qlora."
            )

        _log.info("Fusionando adaptadores LoRA de %s", src)
        model = AutoPeftModelForCausalLM.from_pretrained(str(src))
        merged = model.merge_and_unload()

        out = Path(self.config.export.output_dir) / "merged"
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out))
        AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
        _log.info("Modelo fusionado guardado en %s", out)
        return ExportResult(format="merge-lora", path=str(out), extra={})
