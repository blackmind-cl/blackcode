"""Conector de Google Docs.

Lista los documentos de una carpeta de Google Drive y extrae su texto. Se
autentica con una cuenta de servicio de Google (más simple para uso interno
que el flujo OAuth de usuario):

    export GOOGLE_APPLICATION_CREDENTIALS=/path/al/service-account.json

Crea la cuenta en https://console.cloud.google.com/iam-admin/serviceaccounts,
habilita las APIs de Drive y Docs, y comparte la carpeta del Drive con el
correo de la cuenta de servicio.

Extra requerido:  pip install 'blackcode[gdocs]'
"""

from __future__ import annotations

import os

from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("ingest.gdocs")


def _extract_doc_text(doc: dict) -> str:
    """Texto plano de un Google Doc representado como JSON."""
    parts: list[str] = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for run in paragraph.get("elements", []):
            text_run = run.get("textRun", {})
            content = text_run.get("content", "")
            if content:
                parts.append(content)
    return "".join(parts)


def ingest_gdocs(folder_ids: list[str], page_size: int = 100) -> list[dict]:
    """Ingiere todos los Google Docs dentro de las carpetas indicadas."""
    ensure_extra("gdocs", "google.oauth2", "googleapiclient")
    from google.oauth2 import service_account  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise ValueError(
            "Falta GOOGLE_APPLICATION_CREDENTIALS apuntando al JSON de la "
            "cuenta de servicio."
        )
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/documents.readonly",
        ],
    )
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    docs = build("docs", "v1", credentials=credentials, cache_discovery=False)

    records: list[dict] = []
    for folder_id in folder_ids:
        _log.info("Listando Google Docs en la carpeta %s", folder_id)
        query = (
            f"'{folder_id}' in parents and "
            "mimeType='application/vnd.google-apps.document' and trashed=false"
        )
        result = drive.files().list(
            q=query, pageSize=page_size, fields="files(id, name)"
        ).execute()
        for file in result.get("files", []):
            doc = docs.documents().get(documentId=file["id"]).execute()
            text = _extract_doc_text(doc).strip()
            if text:
                records.append(
                    {
                        "text": text,
                        "source": f"gdocs:{file['name']} ({file['id']})",
                    }
                )
    return records
