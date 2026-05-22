# Primeros pasos

Este tutorial te lleva de cero a un modelo entrenado y servido, todo en local.

## 1. Instalar

El núcleo es ligero (solo PyYAML). Las dependencias pesadas son *extras*
opcionales — instala solo las que necesites:

```bash
pip install blackcode              # núcleo: CLI, config, pipeline
pip install 'blackcode[llm]'       # fine-tuning de LLMs
pip install 'blackcode[sklearn]'   # ML clásico
pip install 'blackcode[embeddings]'# modelos de embeddings
pip install 'blackcode[serve]'     # servidor de inferencia
pip install 'blackcode[all]'       # todo
```

En desarrollo, desde la raíz del repositorio: `pip install -e '.[all,dev]'`.

## 2. Revisar tu hardware

```bash
blackcode doctor
```

Inspecciona CUDA / MPS / CPU y la VRAM, y recomienda una estrategia
(`full`, `lora`, `qlora` o `cpu`). Para ver qué modelos te convienen:

```bash
blackcode models --task llm
```

## 3. Crear un proyecto

Cada proyecto de Blackcode vive en su propia carpeta — con su configuración,
sus datos y sus salidas — y es autónomo. `blackcode init` lo crea:

```bash
blackcode init mi-proyecto    # crea la carpeta mi-proyecto/
cd mi-proyecto
```

Sin argumento, `blackcode init` usa el directorio actual como proyecto. En
ambos casos deja dentro un `blackcode.yaml` (la configuración) y una carpeta
`data/` para tus fuentes.

En una terminal interactiva, `init` lanza un asistente que te hace preguntas
guiadas con una breve explicación de cada opción. Para partir de una plantilla
sin interacción (útil en scripts), usa `--template`:

```bash
blackcode init mi-proyecto --template chatbot
```

Plantillas disponibles: `chatbot` (LLM), `clasificador` (clasificación de
texto), `embeddings` (búsqueda semántica). Ver la
[referencia de configuración](configuration.md).

Los comandos siguientes usan el `blackcode.yaml` del directorio actual, así que
basta con estar dentro de la carpeta del proyecto.

## 4. Entrenar

```bash
blackcode train
```

Carga los datos, divide en entrenamiento/validación, entrena y evalúa. El
modelo se guarda en `train.output_dir`. Usa `-v` para logs detallados y
`--resume` para continuar desde el último checkpoint.

## 5. Evaluar y exportar

```bash
blackcode eval                  # evalúa sin reentrenar
blackcode export --to gguf      # convierte para despliegue
```

Ver la [guía de exportadores](exporters.md).

## 6. Servir

```bash
blackcode serve
```

Expone el modelo en `http://localhost:8000` con una API compatible con la de
OpenAI (`/v1/chat/completions`). También hay un `Dockerfile` para empaquetarlo.

## 7. Revisar las métricas

```bash
blackcode dashboard ./blackcode-output
```

Genera un `dashboard.html` autocontenido que compara las métricas de tus
ejecuciones. Sin servicios externos: los datos nunca salen de la máquina.
