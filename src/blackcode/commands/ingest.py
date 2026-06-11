"""`blackcode ingest` — extrae fuentes heterogéneas a un corpus JSONL.

Fuentes: archivos/carpetas/URLs/sitemaps posicionales, y conectores externos
(--sql, --notion, --confluence, --gdocs). Los secretos van por variables de
entorno, nunca en la línea de comandos versionable.
"""

from __future__ import annotations

import argparse
import sys


def _run(args: argparse.Namespace) -> int:
    import json
    import os
    from pathlib import Path

    from blackcode.ingest import ingest_paths

    sources_given = bool(
        args.paths or args.sql or args.notion or args.confluence or args.gdocs
    )
    if not sources_given:
        print(
            "Especifica al menos una fuente: rutas/URLs positionales, "
            "--sql, --notion, --confluence o --gdocs.",
            file=sys.stderr,
        )
        return 1

    records: list[dict] = []
    if args.paths:
        records.extend(
            ingest_paths(
                args.paths,
                crawl_depth=args.crawl_depth,
                max_pages=args.max_pages,
                delay=args.delay,
                obey_robots=not args.ignore_robots,
            )
        )
    if args.sql:
        from blackcode.ingest.sql import ingest_sql

        sql_url = args.sql_url or os.environ.get("BLACKCODE_SQL_URL")
        if not sql_url:
            print(
                "--sql requiere --sql-url o la variable BLACKCODE_SQL_URL.",
                file=sys.stderr,
            )
            return 1
        records.extend(ingest_sql(sql_url, args.sql, args.sql_template))
    if args.notion:
        from blackcode.ingest.notion import ingest_notion

        records.extend(ingest_notion(args.notion))
    if args.confluence:
        from blackcode.ingest.confluence import ingest_confluence

        records.extend(ingest_confluence(args.confluence))
    if args.gdocs:
        from blackcode.ingest.gdocs import ingest_gdocs

        records.extend(ingest_gdocs(args.gdocs))

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Ingesta completada: {len(records)} documento(s) -> {out}")
    if args.crawl_depth == 0 and any(
        p.startswith(("http://", "https://")) and not p.lower().endswith(".xml")
        for p in args.paths
    ):
        print("Sugerencia: usa --crawl-depth N para seguir los enlaces del sitio.")
    return 0


def register(sub) -> None:
    p = sub.add_parser("ingest", help="Ingerir documentos a un corpus JSONL")
    p.add_argument(
        "paths",
        nargs="*",
        help="Archivos, carpetas, URLs o sitemaps (.xml)",
    )
    p.add_argument("--output", default="corpus.jsonl", help="JSONL de salida")
    p.add_argument(
        "--crawl-depth",
        type=int,
        default=0,
        help="Profundidad de rastreo para URLs (0 = solo la página dada)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Máximo de páginas por sitio al rastrear",
    )
    p.add_argument(
        "--delay", type=float, default=0.0, help="Pausa (s) entre descargas"
    )
    p.add_argument(
        "--ignore-robots",
        action="store_true",
        help="No respetar robots.txt al rastrear",
    )
    p.add_argument(
        "--sql-url",
        help="URL de conexión SQL (o usa la variable BLACKCODE_SQL_URL)",
    )
    p.add_argument(
        "--sql",
        action="append",
        help="Consulta SQL a ejecutar (repetible)",
    )
    p.add_argument(
        "--sql-template",
        help="Plantilla para combinar columnas en texto, p.ej. '{titulo}\\n{cuerpo}'",
    )
    p.add_argument(
        "--notion",
        action="append",
        metavar="PAGE_ID",
        help="ID de página de Notion a ingerir (repetible)",
    )
    p.add_argument(
        "--confluence",
        action="append",
        metavar="SPACE",
        help="Clave de espacio de Confluence a ingerir (repetible)",
    )
    p.add_argument(
        "--gdocs",
        action="append",
        metavar="FOLDER_ID",
        help="ID de carpeta de Google Drive con Google Docs (repetible)",
    )
    p.set_defaults(func=_run)
