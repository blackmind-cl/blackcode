"""Conector de Notion.

Lee páginas de Notion (incluidos sub-bloques recursivos) y las convierte en
registros `{text, source}`. Requiere un token de integración interna:

    export NOTION_TOKEN=secret_...

Crea la integración en https://www.notion.so/my-integrations y compártela con
las páginas que quieres ingerir.

Extra requerido:  pip install 'blackcode[notion]'
"""

from __future__ import annotations

import os

from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("ingest.notion")


def _block_text(block: dict) -> str:
    """Extrae el texto plano de un bloque de Notion (paragraph, heading, …)."""
    btype = block.get("type")
    if not btype:
        return ""
    content = block.get(btype, {})
    if not isinstance(content, dict):
        return ""
    rich = content.get("rich_text", [])
    return "".join(item.get("plain_text", "") for item in rich)


def _collect_blocks(client, block_id: str) -> list[str]:
    """Recorre los bloques hijos de `block_id` recursivamente."""
    texts: list[str] = []
    cursor = None
    while True:
        kwargs = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        response = client.blocks.children.list(**kwargs)
        for block in response.get("results", []):
            text = _block_text(block)
            if text:
                texts.append(text)
            if block.get("has_children"):
                texts.extend(_collect_blocks(client, block["id"]))
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return texts


def ingest_notion(page_ids: list[str]) -> list[dict]:
    """Ingiere páginas de Notion por id y devuelve `{text, source}`."""
    ensure_extra("notion", "notion_client")
    from notion_client import Client  # type: ignore

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise ValueError(
            "Falta NOTION_TOKEN. Crea una integración interna en "
            "https://www.notion.so/my-integrations y exporta el token."
        )
    client = Client(auth=token)
    records: list[dict] = []
    for page_id in page_ids:
        _log.info("Leyendo página de Notion %s", page_id)
        texts = _collect_blocks(client, page_id)
        joined = "\n".join(t for t in texts if t.strip()).strip()
        if joined:
            records.append({"text": joined, "source": f"notion:{page_id}"})
    return records
