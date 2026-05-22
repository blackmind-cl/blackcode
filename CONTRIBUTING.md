# Contribuir a Blackcode

¡Gracias por tu interés! Blackcode es un proyecto comunitario.

## Entorno de desarrollo

```bash
git clone <repo>
cd blackcode
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
pytest
ruff check src
```

## Estructura del repositorio

```
src/blackcode/
  hardware.py        Detección de hardware y estrategia (sin deps pesadas)
  registry.py        Sistema de plugins (trainers, exportadores, extractores)
  config.py          Modelo de configuración + carga y validación YAML
  pipeline.py        Orquestador datos -> train -> eval
  log.py             Logging estructurado y progreso (solo stdlib)
  augment.py         Aumento de datos sintético (solo stdlib)
  generate.py        Generación de pares Q&A con un LLM local
  templates.py       Plantillas de proyecto para `init --template`
  wizard.py          Asistente interactivo de `init`
  catalog.py         Catálogo de modelos recomendados por VRAM
  dashboard.py       Tablero de métricas en HTML estático
  cli.py             Interfaz de línea de comandos (solo argparse)
  data/              Carga de datos local-first
  ingest/            Extractores de documentos y rastreador web
  trainers/          Backends de entrenamiento (llm, sklearn, ...)
  exporters/         Exportadores de despliegue (merge-lora, gguf, onnx)
  evaluation/        Formateo y comparación de métricas
  serving/           Servidor de inferencia local
```

La documentación de usuario (tutoriales y guías) está en `docs/`.

## Principios al contribuir

1. **El núcleo se mantiene ligero.** No añadas dependencias obligatorias.
   Toda librería pesada va como *extra* en `pyproject.toml` y se importa de
   forma perezosa dentro del método que la usa.
2. **Privacidad primero.** Nada de telemetría, llamadas salientes ni
   "phone home". `report_to=[]` siempre.
3. **Extiende vía el registro.** Un backend nuevo es una subclase de
   `BaseTrainer` decorada con `@register_trainer("nombre")`. No metas lógica
   específica de un backend en el core.
4. **Degradación elegante.** Si falta un extra, lanza un `ImportError` con el
   comando exacto de instalación.

## Añadir un trainer

1. Crea `src/blackcode/trainers/mi_backend.py` con una subclase de `BaseTrainer`.
2. Decórala con `@register_trainer("mi-backend")`.
3. Impórtala en `src/blackcode/trainers/__init__.py`.
4. Añade un ejemplo en `examples/` y un test en `tests/`.

## Estilo

- `ruff` para lint y formato (config en `pyproject.toml`).
- Type hints en las APIs públicas.
- Docstrings concisos; comentarios solo donde el "por qué" no sea obvio.

## Pull requests

- Una PR por cambio lógico, con tests.
- Describe el problema y la solución.
- Si cambias la API pública, actualiza el README y el ROADMAP.

## Código de conducta

Sé respetuoso y constructivo. Asumimos buena fe.
