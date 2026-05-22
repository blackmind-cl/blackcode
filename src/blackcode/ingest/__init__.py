"""Ingesta de documentos: convierte fuentes heterogéneas en un corpus de texto.

Importar este módulo registra los extractores de fábrica. `ingest_paths`
recorre archivos, carpetas y URLs y devuelve registros `{text, source}` listos
para escribirse como JSONL y usarse como datos de entrenamiento.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from blackcode.ingest import image, pdf, text, web, word  # noqa: F401  (registro)
from blackcode.ingest.base import BaseExtractor
from blackcode.ingest.crawl import crawl, crawl_sitemap
from blackcode.log import get_logger
from blackcode.registry import get_extractor, list_extractors

_log = get_logger("ingest")

__all__ = ["BaseExtractor", "ingest_paths", "supported_extensions"]


def supported_extensions() -> dict[str, type]:
    """Mapa extensión -> clase de extractor, construido desde el registro."""
    mapping: dict[str, type] = {}
    for name in list_extractors():
        cls = get_extractor(name)
        for ext in getattr(cls, "extensions", ()):
            mapping.setdefault(ext.lower(), cls)
    return mapping


def _extract_file(path: Path, extractors: dict[str, type]) -> str | None:
    cls = extractors.get(path.suffix.lower())
    if cls is None:
        _log.warning("Sin extractor para %s (extensión no soportada)", path)
        return None
    try:
        return cls().extract(path)
    except Exception as exc:  # noqa: BLE001 - un archivo roto no detiene la ingesta
        _log.warning("No se pudo extraer %s: %s", path, exc)
        return None


def _fetch_url(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (OSError, ValueError) as exc:
        _log.warning("No se pudo descargar %s: %s", url, exc)
        return None
    return web.html_to_text(html)


def ingest_paths(
    paths: list[str],
    crawl_depth: int = 0,
    max_pages: int = 50,
    delay: float = 0.0,
    obey_robots: bool = True,
) -> list[dict]:
    """Recorre archivos, carpetas y URLs y devuelve registros `{text, source}`.

    Con `crawl_depth > 0`, cada URL se rastrea siguiendo sus enlaces del mismo
    dominio hasta esa profundidad (máximo `max_pages` páginas por sitio).
    """
    extractors = supported_extensions()
    records: list[dict] = []

    def add(content: str | None, source: str) -> None:
        if content and content.strip():
            records.append({"text": content.strip(), "source": source})
        else:
            _log.warning("Sin texto extraíble: %s", source)

    for raw in paths:
        if raw.startswith(("http://", "https://")):
            if raw.lower().endswith(".xml"):
                # Heurística: URL terminada en .xml = sitemap.
                records.extend(
                    crawl_sitemap(raw, max_pages, delay, obey_robots)
                )
            elif crawl_depth > 0:
                records.extend(
                    crawl(raw, crawl_depth, max_pages, delay, obey_robots)
                )
            else:
                add(_fetch_url(raw), raw)
            continue
        path = Path(raw)
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    add(_extract_file(file, extractors), str(file))
        elif path.is_file():
            add(_extract_file(path, extractors), str(path))
        else:
            _log.warning("Ruta inexistente: %s", raw)
    return records
