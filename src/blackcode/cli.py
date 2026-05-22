"""Interfaz de línea de comandos de Blackcode.

Usa solo argparse (sin dependencias) para que la CLI esté siempre disponible.
Comandos:
  blackcode doctor              Reporte de hardware y estrategia recomendada
  blackcode models              Catálogo de modelos recomendados por VRAM
  blackcode init [carpeta]      Crea un proyecto (carpeta con su configuración)
  blackcode ingest [fuentes]    Ingiere fuentes (archivos/web/sitemap/SQL/Notion/…) a un JSONL
  blackcode generate-qa <jsonl> Genera pares pregunta/respuesta desde un corpus
  blackcode trainers            Lista los trainers disponibles
  blackcode exporters           Lista los exportadores disponibles
  blackcode train  <config>     Ejecuta el pipeline completo
  blackcode eval   <config>     Evalúa un modelo ya entrenado (sin reentrenar)
  blackcode export <config>     Exporta el modelo (GGUF / ONNX / merge-lora)
  blackcode serve  <config>     Sirve el modelo entrenado (API estilo OpenAI)
  blackcode dashboard [rutas]   Genera un tablero HTML de métricas
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from blackcode import __version__
from blackcode.log import configure_logging, get_logger

# Archivo de configuración que identifica una carpeta como proyecto Blackcode.
_PROJECT_CONFIG = "blackcode.yaml"


def _cmd_doctor(_: argparse.Namespace) -> int:
    from blackcode.hardware import detect_hardware

    hw = detect_hardware()
    print("Blackcode · reporte de hardware")
    print("-" * 40)
    print(f"  SO            : {hw.os}")
    print(f"  Python        : {hw.python}")
    print(f"  CPUs          : {hw.cpu_count}")
    print(f"  RAM           : {hw.ram_gb} GB")
    print(f"  Acelerador    : {hw.accelerator.value}")
    if hw.gpus:
        for g in hw.gpus:
            print(f"  GPU {g.index}        : {g.name} ({g.total_memory_gb} GB)")
        print(f"  VRAM total    : {hw.total_vram_gb} GB")
        if hw.is_multi_gpu:
            print(f"  Multi-GPU     : sí ({len(hw.gpus)}) · FSDP automático")
    print("-" * 40)
    print(f"  Estrategia recomendada (sin tamaño de modelo): "
          f"{hw.recommend_strategy().value}")
    print("  Para un modelo de 7B :", hw.recommend_strategy(7).value)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from pathlib import Path

    from blackcode.templates import DEFAULT_TEMPLATE, TEMPLATES

    if args.template and args.template not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES))
        print(
            f"Plantilla desconocida: '{args.template}'. Disponibles: {available}",
            file=sys.stderr,
        )
        return 1

    # Sin argumento, el proyecto es el directorio actual; con argumento, se
    # crea esa carpeta. Cada proyecto es autónomo: config, datos y salidas.
    project_dir = Path(args.directory) if args.directory else Path.cwd()
    if project_dir.exists() and not project_dir.is_dir():
        print(f"{project_dir} existe y no es un directorio.", file=sys.stderr)
        return 1
    config_path = project_dir / _PROJECT_CONFIG
    if config_path.exists() and not args.force:
        print(
            f"Ya existe un proyecto en {project_dir} ({_PROJECT_CONFIG}). "
            "Usa --force para sobrescribir.",
            file=sys.stderr,
        )
        return 1

    if args.template:
        content = TEMPLATES[args.template]
        origen = f"plantilla '{args.template}'"
    elif sys.stdin.isatty():
        import yaml

        from blackcode.config import RunConfig
        from blackcode.wizard import run_wizard

        config = run_wizard(default_name=project_dir.resolve().name or "proyecto")
        if config is None:  # el usuario eligió «salir»
            print("Asistente cancelado; no se creó ningún proyecto.")
            return 0
        RunConfig.from_dict(config)  # valida antes de escribir
        content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        origen = "asistente interactivo"
    else:
        content = TEMPLATES[DEFAULT_TEMPLATE]
        origen = f"plantilla '{DEFAULT_TEMPLATE}' (sin terminal interactiva)"

    project_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    (project_dir / "data").mkdir(exist_ok=True)

    print(f"\nProyecto Blackcode creado en {project_dir} ({origen}).")
    print(f"  {_PROJECT_CONFIG}   configuración del proyecto")
    print("  data/            coloca aquí tus fuentes y datasets")
    if args.directory:
        print(f"\nSiguiente paso:  cd {args.directory} && blackcode train")
    else:
        print("\nSiguiente paso:  blackcode train")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from blackcode.catalog import recommend_models
    from blackcode.hardware import detect_hardware

    hw = detect_hardware()
    print(
        f"Modelos recomendados · VRAM detectada: {hw.total_vram_gb} GB · "
        f"acelerador: {hw.accelerator.value}"
    )
    print("-" * 60)
    for model, strategy in recommend_models(hw, args.task):
        print(f"  {model.id}  ({model.params_billions}B · {model.task})")
        print(f"    estrategia recomendada: {strategy} — {model.description}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from pathlib import Path

    from blackcode.dashboard import find_runs, render_dashboard

    runs = find_runs(args.paths or ["."])
    out = Path(args.output)
    out.write_text(render_dashboard(runs), encoding="utf-8")
    print(f"Tablero con {len(runs)} ejecución(es) escrito en {out}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
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


def _cmd_generate_qa(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from blackcode.generate import generate_qa

    corpus = Path(args.corpus)
    records = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = generate_qa(
        records,
        model_id=args.model,
        device=args.device,
        per_chunk=args.per_chunk,
        chunk_size=args.chunk_size,
    )
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        for row in dataset:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generados {len(dataset)} pares Q&A -> {out}")
    return 0


def _cmd_trainers(_: argparse.Namespace) -> int:
    from blackcode.registry import list_trainers
    import blackcode.trainers  # noqa: F401  (registra los de fábrica)

    print("Trainers disponibles:")
    for name in list_trainers():
        print(f"  - {name}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.evaluation.metrics import summarize
    from blackcode.pipeline import Pipeline
    import blackcode.trainers  # noqa: F401

    config = load_config(args.config)
    if args.resume:
        config.train.resume = True
    print(f"Ejecutando '{config.name}' (trainer={config.train.trainer})")
    result = Pipeline(config).run(do_eval=not args.no_eval)
    print("\nEntrenamiento completado. Métricas de entrenamiento:")
    print(summarize(result.train.metrics))
    if result.eval_metrics:
        print("\nMétricas de evaluación:")
        print(summarize(result.eval_metrics))
    print(f"\nModelo guardado en: {result.train.output_dir}")
    return 0


def _cmd_exporters(_: argparse.Namespace) -> int:
    from blackcode.registry import list_exporters
    import blackcode.exporters  # noqa: F401  (registra los de fábrica)

    print("Exportadores disponibles:")
    for name in list_exporters():
        print(f"  - {name}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.exporters import export_model

    config = load_config(args.config)
    results = export_model(config, args.to)
    print("Exportación completada:")
    for result in results:
        print(f"  {result.format:<12} -> {result.path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.evaluation.metrics import summarize
    from blackcode.pipeline import Pipeline
    import blackcode.trainers  # noqa: F401

    config = load_config(args.config)
    print(f"Evaluando '{config.name}' (modelo en {config.train.output_dir})")
    metrics = Pipeline(config).evaluate()
    print("\nMétricas de evaluación:")
    print(summarize(metrics))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from blackcode.config import load_config
    from blackcode.serving import serve_model

    serve_model(load_config(args.config))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackcode",
        description="Motor open source de entrenamiento e inferencia de IA local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Ejemplos:
              blackcode doctor
              blackcode init mi-proyecto       # crea la carpeta del proyecto
              cd mi-proyecto
              blackcode ingest https://miempresa.com --crawl-depth 2
              blackcode train                  # usa ./blackcode.yaml
              blackcode serve
            """
        ),
    )
    parser.add_argument("--version", action="version", version=f"blackcode {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Salida detallada (nivel DEBUG)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Reporte de hardware").set_defaults(func=_cmd_doctor)

    p_models = sub.add_parser("models", help="Catálogo de modelos por VRAM")
    p_models.add_argument(
        "--task",
        choices=["llm", "embeddings", "classifier"],
        help="Filtrar por tipo de tarea",
    )
    p_models.set_defaults(func=_cmd_models)

    p_init = sub.add_parser(
        "init", help="Crear un proyecto (asistente interactivo o plantilla)"
    )
    p_init.add_argument(
        "directory",
        nargs="?",
        help="Carpeta del proyecto (por defecto: el directorio actual)",
    )
    p_init.add_argument(
        "--template",
        help="Plantilla no interactiva: chatbot | clasificador | embeddings",
    )
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=_cmd_init)

    p_ingest = sub.add_parser("ingest", help="Ingerir documentos a un corpus JSONL")
    p_ingest.add_argument(
        "paths",
        nargs="*",
        help="Archivos, carpetas, URLs o sitemaps (.xml)",
    )
    p_ingest.add_argument(
        "--output", default="corpus.jsonl", help="JSONL de salida"
    )
    p_ingest.add_argument(
        "--crawl-depth",
        type=int,
        default=0,
        help="Profundidad de rastreo para URLs (0 = solo la página dada)",
    )
    p_ingest.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Máximo de páginas por sitio al rastrear",
    )
    p_ingest.add_argument(
        "--delay", type=float, default=0.0, help="Pausa (s) entre descargas"
    )
    p_ingest.add_argument(
        "--ignore-robots",
        action="store_true",
        help="No respetar robots.txt al rastrear",
    )
    p_ingest.add_argument(
        "--sql-url",
        help="URL de conexión SQL (o usa la variable BLACKCODE_SQL_URL)",
    )
    p_ingest.add_argument(
        "--sql",
        action="append",
        help="Consulta SQL a ejecutar (repetible)",
    )
    p_ingest.add_argument(
        "--sql-template",
        help="Plantilla para combinar columnas en texto, p.ej. '{titulo}\\n{cuerpo}'",
    )
    p_ingest.add_argument(
        "--notion",
        action="append",
        metavar="PAGE_ID",
        help="ID de página de Notion a ingerir (repetible)",
    )
    p_ingest.add_argument(
        "--confluence",
        action="append",
        metavar="SPACE",
        help="Clave de espacio de Confluence a ingerir (repetible)",
    )
    p_ingest.add_argument(
        "--gdocs",
        action="append",
        metavar="FOLDER_ID",
        help="ID de carpeta de Google Drive con Google Docs (repetible)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_qa = sub.add_parser("generate-qa", help="Generar pares Q&A desde un corpus")
    p_qa.add_argument(
        "corpus",
        nargs="?",
        default="corpus.jsonl",
        help="JSONL de corpus (de 'blackcode ingest')",
    )
    p_qa.add_argument("--output", default="dataset.jsonl", help="JSONL de salida")
    p_qa.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Modelo generador local",
    )
    p_qa.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo para la generación",
    )
    p_qa.add_argument(
        "--per-chunk", type=int, default=3, help="Pares Q&A por trozo de texto"
    )
    p_qa.add_argument(
        "--chunk-size", type=int, default=2000, help="Tamaño de trozo (caracteres)"
    )
    p_qa.set_defaults(func=_cmd_generate_qa)

    sub.add_parser("trainers", help="Listar trainers").set_defaults(func=_cmd_trainers)
    sub.add_parser("exporters", help="Listar exportadores").set_defaults(
        func=_cmd_exporters
    )

    p_train = sub.add_parser("train", help="Ejecutar el pipeline")
    p_train.add_argument("config", nargs="?", default=_PROJECT_CONFIG)
    p_train.add_argument("--no-eval", action="store_true", help="Omitir evaluación")
    p_train.add_argument(
        "--resume", action="store_true", help="Reanudar desde el último checkpoint"
    )
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluar un modelo ya entrenado")
    p_eval.add_argument("config", nargs="?", default=_PROJECT_CONFIG)
    p_eval.set_defaults(func=_cmd_eval)

    p_export = sub.add_parser("export", help="Exportar el modelo a despliegue")
    p_export.add_argument("config", nargs="?", default=_PROJECT_CONFIG)
    p_export.add_argument(
        "--to",
        action="append",
        metavar="FORMATO",
        help="Formato de exportación (repetible): merge-lora | gguf | onnx",
    )
    p_export.set_defaults(func=_cmd_export)

    p_serve = sub.add_parser("serve", help="Servir el modelo entrenado")
    p_serve.add_argument("config", nargs="?", default=_PROJECT_CONFIG)
    p_serve.set_defaults(func=_cmd_serve)

    p_dash = sub.add_parser("dashboard", help="Generar un tablero HTML de métricas")
    p_dash.add_argument(
        "paths", nargs="*", help="Carpetas a escanear (por defecto: el dir actual)"
    )
    p_dash.add_argument("--output", default="dashboard.html", help="HTML de salida")
    p_dash.set_defaults(func=_cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCancelado.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        get_logger("cli").error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
