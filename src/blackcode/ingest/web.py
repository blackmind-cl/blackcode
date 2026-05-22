"""Extractor de HTML / páginas web vía BeautifulSoup.

Maneja archivos `.html` locales; el orquestador de ingesta usa `html_to_text`
también con el contenido descargado de una URL.

Extra requerido:  pip install 'blackcode[web]'
"""

from __future__ import annotations

from pathlib import Path

from blackcode.ingest.base import BaseExtractor
from blackcode.install import ensure_extra
from blackcode.registry import register_extractor


def html_to_text(html: str) -> str:
    """Convierte HTML en texto plano, descartando scripts y estilos."""
    ensure_extra("web", "bs4")
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


@register_extractor("html")
class HtmlExtractor(BaseExtractor):
    extensions = (".html", ".htm")

    def extract(self, path: Path) -> str:
        return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
