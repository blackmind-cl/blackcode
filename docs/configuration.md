# Configuración

Una ejecución de Blackcode se describe en un único archivo YAML, versionable
en git junto al resto de tu proyecto. Tiene cinco bloques; `data` y `train`
son obligatorios.

```yaml
name: mi-modelo
data: { ... }
train: { ... }
evaluate: { ... }   # opcional
serve: { ... }      # opcional
export: { ... }     # opcional
```

La configuración se valida al cargarse: claves desconocidas, tipos inválidos
o valores fuera de rango producen un error claro antes de entrenar.

## Bloque `data`

| Clave | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `path` | str | *(obligatorio)* | Ruta al dataset, o nombre del dataset HF si `format: hf`. |
| `format` | str | `auto` | `auto`, `jsonl`, `json`, `csv` o `hf`. |
| `text_field` | str | `text` | Campo con el texto principal. |
| `label_field` | str | — | Campo con la etiqueta (clasificación). |
| `instruction_template` | str | — | Plantilla que se renderiza por registro. |
| `validation_split` | float | `0.1` | Fracción reservada para validación. |
| `max_samples` | int | — | Límite de registros a cargar. |
| `augment` | int | `0` | Nº de variantes sintéticas por registro (solo entrenamiento). |
| `options` | dict | `{}` | Opciones específicas del trainer. |

## Bloque `train`

| Clave | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `trainer` | str | *(obligatorio)* | Trainer registrado (`llm`, `sklearn`, …). |
| `model` | str | *(obligatorio)* | Id de modelo (Hugging Face) o estimador. |
| `output_dir` | str | `./blackcode-output` | Dónde se guarda el modelo. |
| `strategy` | str | `auto` | `auto`, `full`, `lora`, `qlora` o `cpu`. |
| `epochs` | float | `3` | Épocas de entrenamiento. |
| `batch_size` | int | `0` | `0` = autodetectar por VRAM. |
| `learning_rate` | float | `0.0002` | Tasa de aprendizaje. |
| `max_seq_length` | int | `1024` | Longitud máxima de secuencia. |
| `seed` | int | `42` | Semilla (división de datos, entrenamiento). |
| `resume` | bool | `false` | Reanudar desde el último checkpoint. |
| `options` | dict | `{}` | Opciones específicas del trainer. |

## Bloque `evaluate`

| Clave | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `metrics` | list | `[loss]` | Métricas a calcular; dependen del trainer. |
| `options` | dict | `{}` | Opciones de evaluación. |

## Bloque `serve`

| Clave | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `host` | str | `127.0.0.1` | Interfaz de escucha. |
| `port` | int | `8000` | Puerto. |
| `api` | str | `openai` | Formato de API expuesto. |
| `options` | dict | `{}` | Opciones del servidor. |

## Bloque `export`

| Clave | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `formats` | list | `[]` | Formatos: `merge-lora`, `gguf`, `onnx`. |
| `output_dir` | str | `./blackcode-export` | Dónde se escriben los artefactos. |
| `quantization` | str | — | Tipo de cuantización (p.ej. `q4_k_m` en GGUF). |
| `options` | dict | `{}` | Opciones del exportador. |

Las opciones específicas de cada trainer o exportador (p.ej. `lora_r`,
`gradient_accumulation_steps`, `positive_field`, `method`, `llama_cpp_dir`)
van en el `options` del bloque correspondiente. Ver la
[guía de trainers](trainers.md) y la de [exportadores](exporters.md).
