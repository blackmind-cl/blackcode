"""Extractor de documentos Word (.docx) vía python-docx.

Extra requerido:  pip install 'blackcode[docx]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.ingest.base import BaseExtractor
from blackcode.install import ensure_extra
from blackcode.registry import register_extractor


@register_extractor("docx")
class DocxExtractor(BaseExtractor):
    extensions = (".docx",)

    def extract(self, path: Path) -> str:
        ensure_extra("docx", "docx")
        import docx  # type: ignore  (paquete python-docx)

        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
