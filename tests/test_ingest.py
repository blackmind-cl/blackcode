"""Tests del subsistema de ingesta de documentos.

Los extractores con dependencias pesadas (pdf, docx, web) se prueban con
`importorskip`: corren si el extra está instalado y se omiten si no.
"""

import pytest

from blackcode.ingest import ingest_paths, supported_extensions
from blackcode.ingest.text import TextExtractor
from blackcode.registry import list_extractors


def test_builtin_extractors_registered():
    assert {"text", "pdf", "docx", "html", "image"} <= set(list_extractors())


def test_supported_extensions_maps_common_types():
    exts = supported_extensions()
    assert ".txt" in exts and ".pdf" in exts and ".docx" in exts


def test_text_extractor_reads_plain_text(tmp_path):
    f = tmp_path / "nota.txt"
    f.write_text("contenido de prueba", encoding="utf-8")
    assert TextExtractor().extract(f) == "contenido de prueba"


def test_ingest_paths_collects_text_documents(tmp_path):
    (tmp_path / "a.txt").write_text("documento uno", encoding="utf-8")
    (tmp_path / "b.md").write_text("# documento dos", encoding="utf-8")
    records = ingest_paths([str(tmp_path)])
    assert len(records) == 2
    assert all("text" in r and "source" in r for r in records)


def test_ingest_paths_skips_unknown_extensions(tmp_path):
    (tmp_path / "ok.txt").write_text("válido", encoding="utf-8")
    (tmp_path / "raro.xyz").write_text("ignorado", encoding="utf-8")
    assert len(ingest_paths([str(tmp_path)])) == 1


def test_ingest_paths_skips_missing_path():
    assert ingest_paths(["/ruta/que/no/existe"]) == []


def test_ingest_paths_skips_empty_documents(tmp_path):
    (tmp_path / "vacio.txt").write_text("   ", encoding="utf-8")
    assert ingest_paths([str(tmp_path)]) == []


def test_html_extractor_strips_scripts(tmp_path):
    pytest.importorskip("bs4")
    from blackcode.ingest.web import HtmlExtractor

    f = tmp_path / "p.html"
    f.write_text(
        "<html><body><p>Hola</p><script>x=1</script></body></html>",
        encoding="utf-8",
    )
    text = HtmlExtractor().extract(f)
    assert "Hola" in text and "x=1" not in text


def test_docx_extractor_reads_paragraphs(tmp_path):
    docx = pytest.importorskip("docx")
    from blackcode.ingest.word import DocxExtractor

    document = docx.Document()
    document.add_paragraph("primer parrafo")
    document.add_paragraph("segundo parrafo")
    path = tmp_path / "doc.docx"
    document.save(str(path))
    text = DocxExtractor().extract(path)
    assert "primer parrafo" in text and "segundo parrafo" in text
