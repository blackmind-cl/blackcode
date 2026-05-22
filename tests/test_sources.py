"""Tests de los conectores externos (SQL, Notion, Confluence, Google Docs).

Los conectores remotos no se pueden ejercitar sin credenciales reales, así que
testeamos las funciones puras (parsing de respuestas) y el extremo a extremo
de SQL contra SQLite (stdlib, sin extra).
"""

import sqlite3
from pathlib import Path

import pytest


# ---------- SQL ----------


def test_sql_row_to_text_with_template():
    from blackcode.ingest.sql import _row_to_text

    assert _row_to_text({"a": "Hola", "b": "mundo"}, "{a} {b}") == "Hola mundo"


def test_sql_row_to_text_without_template_concatenates():
    from blackcode.ingest.sql import _row_to_text

    assert _row_to_text({"a": "uno", "b": "dos"}, None) == "uno\ndos"


def test_sql_row_to_text_skips_none_values():
    from blackcode.ingest.sql import _row_to_text

    assert _row_to_text({"a": "uno", "b": None}, None) == "uno"


def test_sql_safe_url_hides_password():
    from blackcode.ingest.sql import _safe_url

    safe = _safe_url("postgresql://user:secreta@host:5432/db")
    assert "secreta" not in safe
    assert "host" in safe and "/db" in safe


def test_sql_unsupported_scheme_raises():
    from blackcode.ingest.sql import _connect

    with pytest.raises(ValueError, match="Esquema SQL"):
        _connect("mongodb://x")


def test_sql_driver_dispatch_covers_all_motors():
    from blackcode.ingest.sql import _driver_for_scheme

    assert _driver_for_scheme("sqlite") == "sqlite"
    assert _driver_for_scheme("postgresql") == "postgres"
    assert _driver_for_scheme("postgres") == "postgres"
    assert _driver_for_scheme("mysql") == "mysql"
    assert _driver_for_scheme("mssql") == "mssql"
    assert _driver_for_scheme("sqlserver") == "mssql"
    assert _driver_for_scheme("oracle") == "oracle"
    assert _driver_for_scheme("MySQL") == "mysql"  # case-insensitive
    assert _driver_for_scheme("mongodb") is None
    assert _driver_for_scheme("foobar") is None


def test_ingest_sql_sqlite_end_to_end(tmp_path: Path):
    from blackcode.ingest.sql import ingest_sql

    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE articles (title TEXT, body TEXT)")
    conn.execute("INSERT INTO articles VALUES (?, ?)", ("hola", "mundo"))
    conn.execute("INSERT INTO articles VALUES (?, ?)", ("foo", "bar"))
    conn.commit()
    conn.close()

    records = ingest_sql(
        f"sqlite:///{db}",
        ["SELECT title, body FROM articles"],
        template="{title}\n{body}",
    )
    assert len(records) == 2
    texts = {r["text"] for r in records}
    assert "hola\nmundo" in texts and "foo\nbar" in texts


# ---------- Notion ----------


def test_notion_block_text_extracts_rich_text():
    from blackcode.ingest.notion import _block_text

    block = {
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"plain_text": "Hola "},
                {"plain_text": "mundo."},
            ]
        },
    }
    assert _block_text(block) == "Hola mundo."


def test_notion_block_text_handles_unknown_or_empty_block():
    from blackcode.ingest.notion import _block_text

    assert _block_text({}) == ""
    assert _block_text({"type": "image", "image": {}}) == ""


# ---------- Google Docs ----------


def test_gdocs_extract_doc_text_from_sample():
    from blackcode.ingest.gdocs import _extract_doc_text

    doc = {
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [
                            {"textRun": {"content": "Hola "}},
                            {"textRun": {"content": "mundo\n"}},
                        ]
                    }
                },
                {
                    "paragraph": {
                        "elements": [
                            {"textRun": {"content": "Segundo párrafo\n"}}
                        ]
                    }
                },
            ]
        }
    }
    assert _extract_doc_text(doc) == "Hola mundo\nSegundo párrafo\n"


def test_gdocs_extract_doc_text_handles_empty_doc():
    from blackcode.ingest.gdocs import _extract_doc_text

    assert _extract_doc_text({}) == ""
    assert _extract_doc_text({"body": {}}) == ""
