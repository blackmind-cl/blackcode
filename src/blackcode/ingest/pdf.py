"""Extractor de PDF (texto embebido) vía pypdf.

No hace OCR: un PDF escaneado (solo imágenes) devolverá texto vacío; para esos
casos conviertelo a imagen y usa el extractor de imágenes.

Extra requerido:  pip install 'blackcode[pdf]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.ingest.base import BaseExtractor
from blackcode.install import ensure_extra
from blackcode.log import get_logger
from blackcode.registry import register_extractor

_log = get_logger("ingest.pdf")


@register_extractor("pdf")
class PdfExtractor(BaseExtractor):
    extensions = (".pdf",)

    def extract(self, path: Path) -> str:
        ensure_extra("pdf", "pypdf")
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            _log.warning(
                "%s no tiene texto embebido (¿PDF escaneado?). Requiere OCR.", path
            )
        return text
