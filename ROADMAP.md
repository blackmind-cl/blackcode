# Hoja de ruta

Estado actual: **v0.1 (alpha)** — el esqueleto y la API del core funcionan.

## v0.1 — Núcleo (hecho)
- [x] Detección de hardware (CUDA / MPS / CPU + VRAM)
- [x] Selección automática de estrategia (full / LoRA / QLoRA / CPU)
- [x] Configuración declarativa en YAML
- [x] Registro de plugins (decorador + entry points)
- [x] Carga de datos local-first (JSONL / JSON / CSV / HF)
- [x] Pipeline orquestador
- [x] Trainer LLM (transformers + peft + trl)
- [x] Trainer ML clásico (scikit-learn)
- [x] Servidor de inferencia compatible con la API de OpenAI
- [x] CLI (`doctor`, `init`, `trainers`, `train`, `serve`)

## v0.2 — Robustez (hecho)
- [x] Suite de tests amplia + CI (GitHub Actions)
- [x] Logging estructurado y barras de progreso
- [x] Reanudar desde checkpoint
- [x] Acumulación de gradiente y batch size automático por VRAM
- [x] `blackcode eval` independiente (evaluar sin reentrenar)
- [x] Validación de esquema de configuración con mensajes claros

## v0.3 — Despliegue local (hecho)
- [x] Exportar a **GGUF** (correr con llama.cpp / Ollama)
- [x] Exportar a **ONNX**
- [x] Fusionar adaptadores LoRA en el modelo base
- [x] Cuantización post-entrenamiento configurable
- [x] Imagen Docker para servir

## v0.4 — Más capacidades (hecho)
- [x] Trainer de **embeddings** (sentence-transformers)
- [x] Trainer de **clasificación con transformers** (no solo causal LM)
- [x] DPO / ORPO (alineación de preferencias)
- [x] Multi-GPU (FSDP) con autodetección
- [x] Generación / aumento de datos sintéticos local

## v0.5 — Experiencia (hecho)
- [x] Plantillas de proyecto (`blackcode init --template chatbot|clasificador`)
- [x] Catálogo de modelos recomendados por VRAM
- [x] Tablero de métricas local (sin servicios externos)
- [x] Documentación completa y tutoriales

## v0.6 — Ingesta de documentos (hecho)
- [x] Extractores de PDF, Word, HTML/web, imágenes (OCR) y texto plano
- [x] Registro de extractores extensible (plugins)
- [x] Comando `blackcode ingest` — documentos heterogéneos a un corpus JSONL

## v0.7 — Datos de instrucción (hecho)
- [x] Troceado de texto y generación de pares pregunta/respuesta con un LLM local
- [x] Comando `blackcode generate-qa` con dispositivo (CPU/GPU) configurable

## No objetivos
- Entrenar modelos fundacionales desde cero (coste prohibitivo, poca demanda).
- Telemetría, cuentas o cualquier dependencia de servicios en la nube.
