"""Exportadores incluidos en el núcleo.

Importar este módulo registra los exportadores de fábrica. Cada uno importa
sus dependencias pesadas (o herramientas externas) de forma perezosa, así que
basta tener el extra correspondiente para usarlo.
"""

from blackcode.config import RunConfig
from blackcode.exporters import gguf, merge_lora, onnx  # noqa: F401  (efecto: registro)
from blackcode.exporters.base import BaseExporter, ExportResult
from blackcode.log import get_logger
from blackcode.registry import get_exporter

_log = get_logger("exporters")

__all__ = ["BaseExporter", "ExportResult", "export_model"]


def export_model(
    config: RunConfig, formats: list[str] | None = None
) -> list[ExportResult]:
    """Exporta el modelo entrenado a uno o más formatos de despliegue.

    `formats` tiene prioridad sobre `config.export.formats` (lo usa la opción
    `--to` de la CLI). Si no se indica ninguno en ningún sitio, es un error.
    """
    targets = list(formats or config.export.formats)
    if not targets:
        raise ValueError(
            "No se indicó formato de exportación. Define 'export.formats' en "
            "la config o usa --to."
        )
    results: list[ExportResult] = []
    for fmt in targets:
        _log.info("Exportando a '%s'…", fmt)
        result = get_exporter(fmt)(config).export()
        _log.info("Listo: %s -> %s", result.format, result.path)
        results.append(result)
    return results
