"""Extractor de texto de imágenes mediante OCR (pytesseract).

Además del extra de Python necesita el binario de Tesseract instalado en el
sistema (p.ej. `apt install tesseract-ocr` o `brew install tesseract`).

Extra requerido:  pip install 'blackcode[ocr]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.ingest.base import BaseExtractor
from blackcode.install import ensure_extra
from blackcode.registry import register_extractor


@register_extractor("image")
class ImageExtractor(BaseExtractor):
    extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

    def extract(self, path: Path) -> str:
        ensure_extra("ocr", "pytesseract", "PIL")
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        return pytesseract.image_to_string(Image.open(str(path)))
