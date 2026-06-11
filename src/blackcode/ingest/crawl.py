"""Rastreador web: sigue los enlaces de un sitio para ingerir varias páginas.

Recorre en anchura desde una URL inicial, siguiendo solo enlaces del mismo
dominio, hasta una profundidad y un número de páginas máximos. Respeta
robots.txt salvo que se desactive.

Extra requerido:  pip install 'blackcode[web]'
"""

from __future__ import annotations

import time
import urllib.request
import urllib.robotparser
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

from blackcode.ingest.web import html_to_text
from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("ingest.crawl")
_USER_AGENT = "blackcode-crawler"


def _site(url: str) -> str:
    """Dominio normalizado de una URL (sin el prefijo www.)."""
    return urlparse(url).netloc.removeprefix("www.")


def _same_site(url: str, netloc: str) -> bool:
    return _site(url) == netloc


def _safe_to_fetch(url: str, netloc: str) -> bool:
    """¿Es seguro descargar esta URL hallada en contenido remoto?

    Solo http/https y solo el mismo sitio. Sin esto, un sitemap (o HTML)
    malicioso podría listar `file:///etc/passwd` u hosts arbitrarios y el
    rastreador los leería al corpus: urlopen abre file:// sin quejarse.
    """
    scheme = urlparse(url).scheme.lower()
    return scheme in ("http", "https") and _same_site(url, netloc)


def _normalize(url: str) -> str:
    """Normaliza una URL para deduplicar variantes de la misma página.

    Quita el fragmento, garantiza la barra de la raíz y trata `/index.html`
    como equivalente a `/` (el caso de aliasing más común).
    """
    parsed = urlparse(urldefrag(url)[0])
    path = parsed.path
    for index in ("/index.html", "/index.htm"):
        if path.endswith(index):
            path = path[: -len(index) + 1]
            break
    return parsed._replace(path=path or "/").geturl()


def _fetch(url: str) -> str | None:
    """Descarga una URL; devuelve el HTML, o None si no es HTML o falla."""
    if urlparse(url).scheme.lower() not in ("http", "https"):
        _log.warning("Esquema no soportado (solo http/https): %s", url)
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            if content_type and "html" not in content_type.lower():
                return None
            return response.read().decode("utf-8", errors="ignore")
    except (OSError, ValueError) as exc:
        _log.warning("No se pudo descargar %s: %s", url, exc)
        return None


def _extract_links(html: str, base_url: str, netloc: str) -> list[str]:
    """Devuelve los enlaces del mismo sitio encontrados en el HTML."""
    ensure_extra("web", "bs4")
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = _normalize(urljoin(base_url, href))
        if absolute.startswith(("http://", "https://")) and _same_site(
            absolute, netloc
        ):
            links.append(absolute)
    return links


def _load_robots(start_url: str):
    """Carga robots.txt del sitio; None si no existe o no se puede leer."""
    parsed = urlparse(start_url)
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        parser.read()
    except (OSError, ValueError):
        return None
    return parser


def crawl(
    start_url: str,
    max_depth: int = 1,
    max_pages: int = 50,
    delay: float = 0.0,
    obey_robots: bool = True,
) -> list[dict]:
    """Rastrea un sitio desde `start_url` y devuelve registros `{text, source}`."""
    netloc = _site(start_url)
    robots = _load_robots(start_url) if obey_robots else None

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(_normalize(start_url), 0)])
    records: list[dict] = []

    while queue and len(seen) < max_pages:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        if robots is not None and not robots.can_fetch(_USER_AGENT, url):
            _log.warning("robots.txt no permite rastrear %s", url)
            continue
        html = _fetch(url)
        if html is None:
            continue
        text = html_to_text(html)
        if text.strip():
            records.append({"text": text.strip(), "source": url})
            _log.info("Rastreada %s (%d páginas)", url, len(records))
        if depth < max_depth:
            for link in _extract_links(html, url, netloc):
                if link not in seen:
                    queue.append((link, depth + 1))
        if delay:
            time.sleep(delay)
    return records


def _fetch_raw(url: str) -> bytes | None:
    """Descarga una URL devolviendo los bytes en bruto (sin filtro de tipo)."""
    if urlparse(url).scheme.lower() not in ("http", "https"):
        _log.warning("Esquema no soportado (solo http/https): %s", url)
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except (OSError, ValueError) as exc:
        _log.warning("No se pudo descargar %s: %s", url, exc)
        return None


def _parse_sitemap_xml(raw: bytes) -> tuple[list[str], list[str]]:
    """Devuelve (urls, sub_sitemaps) de un XML de sitemap."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return [], []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [
        loc.text.strip()
        for loc in root.findall(".//sm:url/sm:loc", ns)
        if loc.text
    ]
    subs = [
        loc.text.strip()
        for loc in root.findall(".//sm:sitemap/sm:loc", ns)
        if loc.text
    ]
    return urls, subs


def _read_sitemap(start_url: str, max_urls: int) -> list[str]:
    """Lee un sitemap (urlset o sitemapindex, recursivo) y devuelve las URLs.

    Las URLs y sub-sitemaps listados en el XML son contenido remoto: solo se
    aceptan si son http/https y del mismo sitio que el sitemap inicial (lo
    que además exige la propia especificación de sitemaps).
    """
    netloc = _site(start_url)
    pending: list[str] = [start_url]
    seen: set[str] = set()
    found: list[str] = []
    while pending and len(found) < max_urls:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        raw = _fetch_raw(current)
        if raw is None:
            continue
        urls, subs = _parse_sitemap_xml(raw)
        for url in urls:
            if not _safe_to_fetch(url, netloc):
                _log.warning(
                    "Ignorada URL del sitemap fuera del sitio o con esquema "
                    "no soportado: %s",
                    url,
                )
                continue
            found.append(url)
            if len(found) >= max_urls:
                return found
        for sub in subs:
            if _safe_to_fetch(sub, netloc):
                pending.append(sub)
            else:
                _log.warning(
                    "Ignorado sub-sitemap fuera del sitio o con esquema no "
                    "soportado: %s",
                    sub,
                )
    return found


def crawl_sitemap(
    start_url: str,
    max_pages: int = 50,
    delay: float = 0.0,
    obey_robots: bool = True,
) -> list[dict]:
    """Lee un `sitemap.xml` y devuelve `{text, source}` de cada URL listada."""
    urls = _read_sitemap(start_url, max_pages)
    _log.info("Sitemap %s: %d URLs", start_url, len(urls))
    robots = _load_robots(start_url) if obey_robots else None
    records: list[dict] = []
    for url in urls:
        if robots is not None and not robots.can_fetch(_USER_AGENT, url):
            _log.warning("robots.txt no permite rastrear %s", url)
            continue
        html = _fetch(url)
        if html is None:
            continue
        text = html_to_text(html)
        if text.strip():
            records.append({"text": text.strip(), "source": url})
            _log.info("Rastreada %s (%d páginas)", url, len(records))
        if delay:
            time.sleep(delay)
    return records
