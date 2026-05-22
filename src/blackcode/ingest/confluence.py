"""Conector de Confluence (Cloud).

Lee páginas de un espacio de Confluence y las convierte en registros
`{text, source}`. Auth por variables de entorno:

    export CONFLUENCE_URL=https://miempresa.atlassian.net
    export CONFLUENCE_USER=tu@email
    export CONFLUENCE_TOKEN=tu-api-token

Genera el token en https://id.atlassian.com/manage-profile/security/api-tokens.

Extra requerido:  pip install 'blackcode[confluence]'
"""

from __future__ import annotations

import os

from blackcode.ingest.web import html_to_text
from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("ingest.confluence")


def ingest_confluence(
    spaces: list[str], limit_per_space: int = 200
) -> list[dict]:
    """Ingiere las páginas de los espacios de Confluence indicados."""
    ensure_extra("confluence", "atlassian")
    from atlassian import Confluence  # type: ignore

    url = os.environ.get("CONFLUENCE_URL")
    user = os.environ.get("CONFLUENCE_USER")
    token = os.environ.get("CONFLUENCE_TOKEN")
    missing = [
        name
        for name, value in (
            ("CONFLUENCE_URL", url),
            ("CONFLUENCE_USER", user),
            ("CONFLUENCE_TOKEN", token),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Faltan variables de entorno para Confluence: " + ", ".join(missing)
        )

    conf = Confluence(url=url, username=user, password=token, cloud=True)
    records: list[dict] = []
    for space in spaces:
        _log.info("Leyendo páginas del espacio %s", space)
        pages = conf.get_all_pages_from_space(
            space, start=0, limit=limit_per_space, expand="body.storage"
        )
        for page in pages:
            html = page.get("body", {}).get("storage", {}).get("value", "")
            text = html_to_text(html) if html else ""
            if text.strip():
                title = page.get("title", page.get("id", "?"))
                records.append(
                    {
                        "text": text.strip(),
                        "source": f"confluence:{space}/{title}",
                    }
                )
    return records
