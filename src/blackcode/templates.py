"""Plantillas de proyecto para `blackcode init --template`.

Cada plantilla es un archivo YAML completo y válido para un caso de uso
típico. Sirven de punto de partida: el usuario las edita y ejecuta.
"""

CHATBOT = """\
# Plantilla 'chatbot': fine-tuning de un LLM con datos de instrucciones.
name: mi-chatbot

data:
  path: ./data/instrucciones.jsonl
  format: auto              # auto | jsonl | json | csv | hf
  text_field: text
  validation_split: 0.1
  # Lista para el JSONL que produce `blackcode generate-qa` (instruction/output).
  # Si tus datos ya están en el campo `text`, quita esta línea.
  instruction_template: "### Pregunta: {instruction}\n### Respuesta: {output}"
  # augment: 0              # nº de variantes sintéticas por registro

train:
  trainer: llm
  model: Qwen/Qwen2.5-0.5B-Instruct
  output_dir: ./blackcode-output/chatbot
  strategy: auto            # auto detecta full/lora/qlora/cpu por hardware
  epochs: 3
  # learning_rate: 0.0002   # ausente = auto (LoRA→2e-4, full→2e-5)
  max_seq_length: 1024
  # resume: false           # reanudar desde el último checkpoint

evaluate:
  metrics: [loss, perplexity]

serve:
  host: 127.0.0.1
  port: 8000

# export:
#   formats: [merge-lora]    # merge-lora | gguf | onnx
#   quantization: q4_k_m
"""

CLASIFICADOR = """\
# Plantilla 'clasificador': clasificación de texto con un encoder transformer.
name: mi-clasificador

data:
  path: ./data/ejemplos.csv
  format: csv
  text_field: text
  label_field: label
  validation_split: 0.2

train:
  trainer: transformer-classifier
  model: distilbert-base-multilingual-cased
  output_dir: ./blackcode-output/clasificador
  epochs: 3
  # learning_rate: 0.00002  # ausente = 2e-5 (estándar para BERT)

evaluate:
  metrics: [accuracy]
"""

EMBEDDINGS = """\
# Plantilla 'embeddings': afinar un modelo de búsqueda semántica.
name: mi-buscador

data:
  path: ./data/pares.jsonl
  text_field: anchor
  validation_split: 0.1
  options:
    positive_field: positive   # campo con el texto positivo de cada par

train:
  trainer: embeddings
  model: sentence-transformers/all-MiniLM-L6-v2
  output_dir: ./blackcode-output/embeddings
  epochs: 1
  # learning_rate: 0.00002  # ausente = 2e-5 (estándar para sentence-transformers)

evaluate:
  metrics: [cosine_similarity]
"""

TEMPLATES: dict[str, str] = {
    "chatbot": CHATBOT,
    "clasificador": CLASIFICADOR,
    "embeddings": EMBEDDINGS,
}

DEFAULT_TEMPLATE = "chatbot"
