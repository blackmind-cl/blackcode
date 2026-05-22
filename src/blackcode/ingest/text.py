"""Extractor de texto plano y Markdown (solo stdlib)."""

from __future__ import annotations

from pathlib import Path

from blackcode.ingest.base import BaseExtractor
from blackcode.registry import register_extractor


@register_extractor("text")
class TextExtractor(BaseExtractor):
    extensions = (".txt", ".md", ".markdown", ".rst")

    def extract(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")
