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
  install.py         Auto-instalación de extras (`ensure_extra`)
  cli.py             Parser raíz de la CLI (solo argparse)
  commands/          Un módulo por subcomando de la CLI
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
4. **Los extras se instalan solos.** Antes de un import perezoso llama a
   `ensure_extra("<extra>", "<modulo>", …)`: si falta, se instala con pip e
   informa al usuario (y lanza `ImportError` con el comando exacto si pip
   falla).

## Añadir un trainer

1. Crea `src/blackcode/trainers/mi_backend.py` con una subclase de `BaseTrainer`.
2. Decórala con `@register_trainer("mi-backend")`.
3. Impórtala en `src/blackcode/trainers/__init__.py`.
4. Si usas el ecosistema Hugging Face, reutiliza los helpers de
   `trainers/_hf.py` (tokenizer, TrainingArguments comunes, guardado).
5. Añade un ejemplo en `examples/` y un test en `tests/`.

## Publicar un plugin como paquete externo

No hace falta tocar el core: cualquier paquete puede registrar sus plugins
vía entry points en su propio `pyproject.toml`:

```toml
[project.entry-points."blackcode.trainers"]
mi-backend = "mi_paquete.modulo:MiTrainer"

[project.entry-points."blackcode.exporters"]
mi-formato = "mi_paquete.modulo:MiExporter"

[project.entry-points."blackcode.extractors"]
mi-extension = "mi_paquete.modulo:MiExtractor"
```

Tras `pip install mi-paquete`, Blackcode los resuelve por nombre igual que
los integrados (`blackcode trainers` los lista). Si un plugin falla al
cargar, se omite con un aviso en el log — nunca tumba el motor.

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
