"""Tests del rastreador web.

El rastreo end-to-end se prueba contra un servidor HTTP local efímero, sin
dependencia de red. El rastreo necesita BeautifulSoup (extra `[web]`).
"""

import http.server
import socketserver
import threading
from pathlib import Path

import pytest


def test_same_site_ignores_www_prefix():
    from blackcode.ingest.crawl import _same_site

    assert _same_site("https://www.miempresa.cl/x", "miempresa.cl")
    assert _same_site("https://miempresa.cl/y", "miempresa.cl")
    assert not _same_site("https://otro.com/z", "miempresa.cl")


def test_extract_links_keeps_only_same_site():
    pytest.importorskip("bs4")
    from blackcode.ingest.crawl import _extract_links

    html = (
        '<a href="/a">a</a>'
        '<a href="https://otro.com/b">b</a>'
        '<a href="mailto:x@y.z">m</a>'
    )
    links = _extract_links(html, "https://miempresa.cl/", "miempresa.cl")
    assert links == ["https://miempresa.cl/a"]


@pytest.fixture
def site(tmp_path: Path):
    """Sirve un mini-sitio estático en un servidor HTTP local efímero."""
    pytest.importorskip("bs4")
    (tmp_path / "index.html").write_text(
        '<a href="page2.html">2</a><a href="https://example.com/">ext</a>',
        encoding="utf-8",
    )
    (tmp_path / "page2.html").write_text(
        '<p>pagina dos</p><a href="page3.html">3</a>', encoding="utf-8"
    )
    (tmp_path / "page3.html").write_text(
        '<p>pagina tres</p><a href="page2.html">vuelta</a>', encoding="utf-8"
    )

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, *args):  # silencia el log del servidor
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def test_crawl_follows_same_site_links(site):
    from blackcode.ingest.crawl import crawl

    records = crawl(site, max_depth=2)
    assert len(records) == 3  # index + page2 + page3
    assert not any("example.com" in r["source"] for r in records)


def test_crawl_respects_max_depth(site):
    from blackcode.ingest.crawl import crawl

    # Profundidad 1: solo la inicial y la que enlaza directamente.
    assert len(crawl(site, max_depth=1)) == 2


def test_crawl_respects_max_pages(site):
    from blackcode.ingest.crawl import crawl

    assert len(crawl(site, max_depth=5, max_pages=2)) == 2


def test_parse_sitemap_urlset():
    from blackcode.ingest.crawl import _parse_sitemap_xml

    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>https://x.com/a</loc></url>"
        b"<url><loc>https://x.com/b</loc></url>"
        b"</urlset>"
    )
    urls, subs = _parse_sitemap_xml(xml)
    assert urls == ["https://x.com/a", "https://x.com/b"]
    assert subs == []


def test_parse_sitemap_index_returns_subsitemaps():
    from blackcode.ingest.crawl import _parse_sitemap_xml

    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>https://x.com/sm1.xml</loc></sitemap>"
        b"<sitemap><loc>https://x.com/sm2.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    urls, subs = _parse_sitemap_xml(xml)
    assert urls == []
    assert subs == ["https://x.com/sm1.xml", "https://x.com/sm2.xml"]


def test_parse_sitemap_invalid_xml_returns_empty():
    from blackcode.ingest.crawl import _parse_sitemap_xml

    assert _parse_sitemap_xml(b"esto no es xml") == ([], [])
