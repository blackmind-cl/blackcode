"""Conector de bases de datos SQL.

Ingiere filas de una BD como registros de texto. Motores soportados:

| Motor       | Esquema                | Extra                   |
|-------------|------------------------|-------------------------|
| SQLite      | `sqlite:///...`        | — (stdlib)              |
| PostgreSQL  | `postgresql://...`     | `blackcode[postgres]`   |
| MySQL       | `mysql://...`          | `blackcode[mysql]`      |
| SQL Server  | `mssql://...`          | `blackcode[mssql]`      |
| Oracle      | `oracle://...`         | `blackcode[oracle]`     |

Secretos: la cadena de conexión NO va en el YAML versionado. Se pasa por
`--sql-url` o por la variable de entorno `BLACKCODE_SQL_URL`.
"""

from __future__ import annotations

from urllib.parse import urlparse

from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("ingest.sql")


def _driver_for_scheme(scheme: str) -> str | None:
    """Devuelve el driver lógico (`sqlite`, `postgres`, …) o None si no soportado."""
    scheme = scheme.lower()
    if scheme.startswith("sqlite"):
        return "sqlite"
    if scheme.startswith(("postgres", "postgresql")):
        return "postgres"
    if scheme.startswith("mysql"):
        return "mysql"
    if scheme.startswith(("mssql", "sqlserver")):
        return "mssql"
    if scheme.startswith("oracle"):
        return "oracle"
    return None


def _connect(url: str):
    """Conecta a la base según el esquema de la URL."""
    scheme = urlparse(url).scheme.lower()
    driver = _driver_for_scheme(scheme)
    if driver is None:
        raise ValueError(
            f"Esquema SQL no soportado: '{scheme}'. Usa sqlite, postgresql, "
            "mysql, mssql u oracle."
        )

    if driver == "sqlite":
        import sqlite3

        # sqlite:///./mi.db  → "./mi.db";  sqlite:////abs/path → "/abs/path"
        return sqlite3.connect(url.replace("sqlite:///", "", 1))

    parsed = urlparse(url)

    if driver == "postgres":
        ensure_extra("postgres", "psycopg")
        import psycopg  # type: ignore

        return psycopg.connect(url)

    if driver == "mysql":
        ensure_extra("mysql", "pymysql")
        import pymysql  # type: ignore

        return pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
        )

    if driver == "mssql":
        ensure_extra("mssql", "pymssql")
        import pymssql  # type: ignore

        return pymssql.connect(
            server=parsed.hostname,
            port=parsed.port or 1433,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
        )

    if driver == "oracle":
        ensure_extra("oracle", "oracledb")
        import oracledb  # type: ignore

        # `oracle://user:pass@host:port/service_name` → DSN host:port/service.
        # oracledb usa modo `thin` por defecto: cero dependencias del sistema.
        dsn = f"{parsed.hostname}:{parsed.port or 1521}/{parsed.path.lstrip('/')}"
        return oracledb.connect(
            user=parsed.username, password=parsed.password, dsn=dsn
        )

    raise AssertionError(f"driver inesperado: {driver}")  # pragma: no cover


def _row_to_text(row: dict, template: str | None) -> str:
    """Convierte una fila (dict columna→valor) en texto plano."""
    if template:
        try:
            return template.format(**row).strip()
        except KeyError as exc:
            raise KeyError(
                f"La plantilla SQL referencia el campo {exc} ausente en la fila."
            ) from exc
    # Sin plantilla: concatena los valores no nulos.
    return "\n".join(str(v) for v in row.values() if v is not None).strip()


def _safe_url(url: str) -> str:
    """URL sin contraseña, para logs."""
    parsed = urlparse(url)
    if parsed.scheme.startswith("sqlite") or not parsed.hostname:
        return url
    return f"{parsed.scheme}://{parsed.hostname}{parsed.path}"


def ingest_sql(
    url: str, queries: list[str], template: str | None = None
) -> list[dict]:
    """Ejecuta `queries` y devuelve registros `{text, source}` por fila."""
    _log.info("Conectando a %s", _safe_url(url))
    conn = _connect(url)
    records: list[dict] = []
    try:
        for query in queries:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [d[0] for d in cursor.description or []]
            count = 0
            for row_tuple in cursor.fetchall():
                row = dict(zip(columns, row_tuple))
                text = _row_to_text(row, template)
                if text:
                    records.append(
                        {"text": text, "source": f"sql:{query[:80]}"}
                    )
                    count += 1
            _log.info("Query devolvió %d fila(s) con texto", count)
    finally:
        conn.close()
    return records
