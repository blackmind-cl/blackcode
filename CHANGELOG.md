# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/).
Este proyecto sigue [versionado semántico](https://semver.org/lang/es/).

## [Sin publicar]

Nada todavía.

## [0.1.0] — 2026-05-21

Primera versión pública (primer release en PyPI).

### Añadido
- **Núcleo ligero**: una sola dependencia obligatoria (PyYAML). Todo lo
  pesado son extras opcionales que se **auto-instalan al primer uso**
  (`ensure_extra`).
- **Privacidad por diseño**: sin telemetría ni llamadas salientes;
  `report_to=[]` en todos los trainers; el server escucha solo en localhost.
- **Arquitectura de plugins**: trainers, exportadores y extractores se
  registran por decorador o entry point; el núcleo los resuelve por nombre.
- **Trainers**: `llm` (fine-tuning causal con QLoRA/LoRA/full según VRAM),
  `sklearn` (ML clásico), `transformer-classifier` (clasificación con
  encoders), `embeddings` (sentence-transformers), `dpo` (DPO/ORPO).
- **Detección de hardware**: CUDA / MPS / CPU, VRAM, multi-GPU (FSDP
  automático), estrategia y learning rate recomendados.
- **Exportadores**: `merge-lora`, `gguf` (llama.cpp, con cuantización),
  `onnx` (optimum).
- **Ingesta**: archivos (texto, PDF, DOCX, HTML, imágenes con OCR), rastreo
  web con respeto de robots.txt, sitemaps, SQL (SQLite/Postgres/MySQL/SQL
  Server/Oracle), Notion, Confluence y Google Docs. Secretos solo por
  variables de entorno.
- **Generación de datos**: `generate-qa` crea pares pregunta/respuesta desde
  un corpus con un LLM local.
- **Proyectos**: `blackcode init` con asistente interactivo animado o
  plantillas (`chatbot`, `clasificador`, `embeddings`).
- **Servidor local**: API compatible con OpenAI (`/v1/chat/completions`).
- **CLI completa**: doctor, models, init, ingest, generate-qa, train, eval,
  export, serve, dashboard, trainers, exporters.
- **Instalador de una línea** (`install.sh`): instala lo que falte (Homebrew,
  Python, git) y deja el motor en un entorno aislado.
- Tablero HTML estático de métricas, aumento de datos EDA, reanudación desde
  checkpoint, suite de tests.
- Helper compartido `trainers/_hf.py`: política única de checkpoints,
  `report_to=[]` y guardado de métricas para los trainers Hugging Face.
- El registro de plugins avisa por log cuando un entry point de terceros
  falla al cargar, en vez de omitirlo en silencio.

### Infraestructura
- Workflow de release (`release.yml`): al empujar un tag `vX.Y.Z` corre
  lint + tests, valida que el tag coincide con la versión y publica en PyPI
  vía Trusted Publishing (OIDC, sin tokens guardados).

### Seguridad
- El rastreador valida las URLs listadas en sitemaps (y sub-sitemaps):
  solo http/https y solo el mismo sitio, para que un sitemap malicioso no
  pueda hacer que el motor lea archivos locales (`file:///…`) ni hosts
  arbitrarios al corpus.
- La API local de inferencia acota `max_tokens` (1–8192) y `temperature`
  (0–2): una petición no puede dejar la máquina generando sin límite.
