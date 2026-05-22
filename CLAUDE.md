# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Blackcode (`blackcode`) is a local, private, plugin-based engine for training and serving AI models — LLM fine-tuning (QLoRA/LoRA), classic ML (scikit-learn), or custom backends. No telemetry, no outbound calls; data never leaves the machine.

Code docstrings, comments, and user docs are written in **Spanish**. Match that convention in any new code.

## Commands

```bash
pip install -e '.[all,dev]'   # dev setup (editable install with every extra)
pytest                        # run the test suite
pytest tests/test_smoke.py::test_template_rendering   # run a single test
ruff check src                # lint
```

The `blackcode` CLI (entry point `blackcode.cli:main`): `blackcode doctor` (hardware report), `blackcode models` (catalog filtered by detected VRAM), `blackcode init [dir]` (scaffold a project folder — interactive wizard on a TTY, or `--template chatbot|clasificador|embeddings` non-interactive; no arg uses the current directory), `blackcode ingest [sources]` (extract files/URLs/sitemaps/SQL/Notion/Confluence/Google Docs to a JSONL corpus), `blackcode generate-qa <corpus>` (generate instruction Q&A pairs with a local LLM, `--device cpu|cuda|auto`), `blackcode trainers` / `blackcode exporters` (list registered plugins), `blackcode train <config>` (`--resume` continues from a checkpoint, `--no-eval` skips evaluation), `blackcode eval <config>` (evaluate a trained model without retraining), `blackcode export <config>` (`--to merge-lora|gguf|onnx`, convert a trained model for deployment), `blackcode serve <config>`, `blackcode dashboard [paths]` (static HTML metrics dashboard). A global `-v`/`--verbose` flag raises log verbosity to DEBUG.

A **project** is a self-contained folder: `init` writes `blackcode.yaml` (the config) and a `data/` dir into it. `train`/`eval`/`export`/`serve` default their config argument to `./blackcode.yaml`, so commands run from inside a project need no path.

The smoke tests in `tests/` deliberately avoid torch/sklearn so they run on the lightweight core alone. New core tests should keep that property.

## Architecture

A run is one declarative YAML (`data` → `train` → `evaluate` → `serve` blocks) parsed into `RunConfig` (`src/blackcode/config.py`). The same flow serves every backend.

`Pipeline` (`src/blackcode/pipeline.py`) is the orchestrator: it resolves a trainer by name from the registry, loads the dataset, then calls `prepare → train → evaluate`. It is backend-agnostic — it only knows the `BaseTrainer` contract. When `data.augment > 0` it applies stdlib EDA augmentation (`src/blackcode/augment.py`) to the training split only — never validation.

**Plugin registry** (`src/blackcode/registry.py`): a generic `_Registry` backs three plugin kinds — trainers (`@register_trainer`), exporters (`@register_exporter`) and document extractors (`@register_extractor`), each with a matching entry-point group. The core never imports a concrete plugin directly; it resolves them by name at runtime. Importing the `blackcode.trainers` / `blackcode.exporters` / `blackcode.ingest` packages triggers registration of the built-ins as a side effect.

**Trainer contract** (`src/blackcode/trainers/base.py`): every backend subclasses `BaseTrainer` and implements `prepare`, `train`, `evaluate`. `resolve_strategy()` honors a user-declared strategy or falls back to hardware autodetection. Built-in trainers: `llm` (causal-LM fine-tuning), `sklearn` (classic ML), `transformer-classifier` (encoder classification), `embeddings` (sentence-transformers), `dpo` (DPO/ORPO preference alignment).

**Exporter contract** (`src/blackcode/exporters/`): each subclasses `BaseExporter` and implements `export()`, converting a trained model in `train.output_dir` into a deployment artifact. Built-ins: `merge-lora` (peft `merge_and_unload`, needs `[llm]`), `onnx` (optimum, needs `[onnx]`), `gguf` (shells out to llama.cpp's `convert_hf_to_gguf.py`; `llama-quantize` for the configurable `export.quantization`). `export_model()` orchestrates one or more formats.

**Hardware detection** (`src/blackcode/hardware.py`): inspects CUDA/MPS/CPU and VRAM, then `recommend_strategy()` picks `full`/`lora`/`qlora`/`cpu`. This module must NOT require torch — it uses torch if present, else `nvidia-smi`, else assumes CPU, so `import blackcode` works anywhere. `is_multi_gpu` (>1 GPU) makes the LLM trainer enable FSDP.

**Data loading** (`src/blackcode/data/base.py`): `Dataset` is a thin list-of-dicts wrapper with deterministic `split()` and `apply_template()`. JSONL/JSON/CSV use the stdlib only; the `hf` format imports `datasets` lazily.

**Logging** (`src/blackcode/log.py`): stdlib-only structured logging — `get_logger()` returns a `blackcode.*` child logger, `configure_logging()` wires a stderr handler, `track()` is a dependency-free progress helper. Logs go to stderr; results (metrics, reports) go to stdout via `print`.

**CLI helpers** (all stdlib-only): `templates.py` (project-template YAML for `init --template`), `wizard.py` (interactive `init` — an animated terminal console with ANSI color, simulated typing, spinners and arrow-key menus via `termios`/`tty`; `run_wizard()` takes injectable input/output and an `animate` flag that auto-disables for non-TTY output and tests, falling back to a numbered menu), `catalog.py` (recommended models annotated by size, for `models`), `dashboard.py` (collects `blackcode_metrics.json` files into a static HTML dashboard).

**Document ingestion** (`src/blackcode/ingest/`): file-based extractors (`@register_extractor`) for `text` (stdlib), `pdf`, `docx`, `html`/web, `image` (OCR). `ingest_paths()` walks files/dirs/URLs and dispatches by extension into `{text, source}` records; URLs with `crawl_depth > 0` are crawled by `crawl.py` (BFS, same-domain, depth/page limits, respects robots.txt). URLs ending in `.xml` are read as sitemaps via `crawl_sitemap()`. External-source connectors live alongside as standalone modules: `sql.py` (SQLite/Postgres/MySQL/SQL Server/Oracle, dispatched by URL scheme via `_driver_for_scheme()`; connection URL via `--sql-url` or `BLACKCODE_SQL_URL`), `notion.py` (`notion-client` + `NOTION_TOKEN`), `confluence.py` (`atlassian-python-api` + URL/user/token env vars), `gdocs.py` (`google-api-python-client` + `GOOGLE_APPLICATION_CREDENTIALS`). Each is an optional extra; secrets via env vars, never in the YAML. `generate.py` then turns the corpus into instruction Q&A pairs with a local LLM (`chunk_text` / `parse_qa` are pure helpers; `generate_qa` runs the model on a configurable device).

**Auto-install** (`src/blackcode/install.py`): `ensure_extra(extra, *modules)` checks each module with `importlib.util.find_spec` and, if any is missing, prints `⟫ Falta 'blackcode[X]'. Instalando…` to stderr and shells out to `pip install 'blackcode[X]'` — no prompt; the action the user already invoked is the consent. Failures inside a process are cached in `_attempted` so we don't retry pip on every call. Every lazy-import site (trainers, exporters, extractors, connectors, server, generator) calls `ensure_extra(...)` immediately before its lazy imports. This is the ONE sanctioned outbound: pip itself. Do not bypass the helper or import a heavy dep at module top.

**One-liner installer** (`install.sh`, repo root): bash script for end users who just want to *use* blackcode without cloning. Served from `https://blackmind.cl/blackcode/install.sh` and mirrored at `https://raw.githubusercontent.com/blackmind-cl/blackcode/main/install.sh`. Detects macOS/Linux + arch. **If something is missing, the installer installs it** — mirroring `ensure_extra`'s philosophy for system deps: on macOS it auto-installs Homebrew (`NONINTERACTIVE=1`) if absent; then installs Python ≥ 3.10 and git via the detected package manager (`brew` / `apt-get` / `dnf` / `yum` / `pacman` / `zypper` / `apk`). The `as_root` helper uses sudo when not running as root and aborts cleanly if neither is available. Then creates an isolated venv at `~/.blackcode/venv`, `pip install git+https://github.com/blackmind-cl/blackcode.git@main` (core only — ~20 MB), drops a shim at `~/.local/bin/blackcode`. Honors `BLACKCODE_HOME`, `BLACKCODE_REF`, `BLACKCODE_BIN` env overrides. Tested by `tests/test_install_sh.py` (syntax + structure; doesn't actually run the install). Heavy Python extras are NOT pre-installed: `ensure_extra` brings them on first use.

## Two hard constraints

1. **Lightweight core.** The only mandatory dependency is PyYAML. Everything heavy (`torch`, `transformers`, `peft`, `trl`, `sklearn`, `fastapi`) is an optional extra in `pyproject.toml` and must be imported lazily *inside the method that uses it* — never at module top level. Call `ensure_extra("<extra>", "<module>", …)` immediately before the lazy imports so a missing extra is auto-installed (it raises `ImportError` with the right `pip install` hint if pip itself fails).

2. **Privacy first.** No telemetry, no outbound calls, no "phone home". Trainers set `report_to=[]`.

Add a new backend by subclassing `BaseTrainer` in `src/blackcode/trainers/` (or `BaseExporter` in `src/blackcode/exporters/`), decorating with `@register_trainer` / `@register_exporter`, and importing it in the package `__init__.py` — never put backend-specific logic in the core.
