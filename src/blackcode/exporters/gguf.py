"""Exportador a GGUF para correr el modelo con llama.cpp / Ollama.

GGUF no tiene un conversor instalable vía pip: se usa el script
`convert_hf_to_gguf.py` de llama.cpp. Indica dónde está clonado llama.cpp
con la variable de entorno `LLAMA_CPP_DIR` o con `export.options.llama_cpp_dir`.

La cuantización (`export.quantization`, p.ej. `q4_k_m`) se aplica con la
herramienta `llama-quantize` si está disponible en el PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from blackcode.exporters.base import BaseExporter, ExportResult
from blackcode.log import get_logger
from blackcode.registry import register_exporter

_log = get_logger("exporters.gguf")


def _find_convert_script(config_dir: str | None) -> Path:
    """Localiza convert_hf_to_gguf.py en la config o en LLAMA_CPP_DIR."""
    for candidate in (config_dir, os.environ.get("LLAMA_CPP_DIR")):
        if candidate:
            script = Path(candidate) / "convert_hf_to_gguf.py"
            if script.exists():
                return script
    raise FileNotFoundError(
        "No se encontró 'convert_hf_to_gguf.py' de llama.cpp. Clona "
        "https://github.com/ggerganov/llama.cpp e indica su ruta con la "
        "variable de entorno LLAMA_CPP_DIR o con export.options.llama_cpp_dir."
    )


@register_exporter("gguf")
class GgufExporter(BaseExporter):
    def export(self) -> ExportResult:
        src = self.model_dir
        if not src.exists():
            raise FileNotFoundError(f"No existe el modelo entrenado: {src}")

        script = _find_convert_script(self.config.export.options.get("llama_cpp_dir"))
        out_dir = Path(self.config.export.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        f16_path = out_dir / f"{self.config.name}.f16.gguf"

        _log.info("Convirtiendo %s a GGUF (f16)", src)
        # sys.executable, no "python": el alias puede no existir (macOS) o
        # apuntar a otro intérprete sin las deps del venv actual.
        subprocess.run(
            [sys.executable, str(script), str(src),
             "--outfile", str(f16_path), "--outtype", "f16"],
            check=True,
        )

        quant = self.config.export.quantization
        if not quant:
            return ExportResult(format="gguf", path=str(f16_path), extra={})

        quant_tool = shutil.which("llama-quantize")
        if quant_tool is None:
            _log.warning(
                "GGUF sin cuantizar: 'llama-quantize' no está en el PATH. "
                "Compila llama.cpp para obtener la herramienta."
            )
            return ExportResult(format="gguf", path=str(f16_path), extra={})

        quant_path = out_dir / f"{self.config.name}.{quant}.gguf"
        _log.info("Cuantizando GGUF a %s", quant)
        subprocess.run(
            [quant_tool, str(f16_path), str(quant_path), quant], check=True
        )
        return ExportResult(
            format="gguf", path=str(quant_path), extra={"quantization": quant}
        )
