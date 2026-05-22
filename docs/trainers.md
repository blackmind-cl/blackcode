# Trainers

Un *trainer* es un backend de entrenamiento. El núcleo no los conoce: los
resuelve por nombre desde el registro de plugins. Elige uno con
`train.trainer`. Lista los disponibles con `blackcode trainers`.

## Trainers integrados

### `llm` — fine-tuning de LLMs

Causal LM con el ecosistema Hugging Face (transformers + peft + trl).
Autoselecciona `full` / `lora` / `qlora` según la VRAM. Si hay varias GPUs
activa FSDP (lánzalo con `accelerate launch`).

- Datos: campo de texto (`data.text_field`), o usa `instruction_template`.
- `options`: `lora_r`, `lora_alpha`, `lora_dropout`, `gradient_accumulation_steps`.
- Extra: `pip install 'blackcode[llm]'`.

### `sklearn` — ML clásico

Clasificación / regresión con scikit-learn (TF-IDF para texto). Funciona en CPU.

- Datos: `text_field` y `label_field`.
- `model`: `logreg` o `ridge`.
- Extra: `pip install 'blackcode[sklearn]'`.

### `transformer-classifier` — clasificación con transformers

Clasificación de texto con un encoder (BERT, RoBERTa…) y cabeza de
clasificación. Construye el mapa de etiquetas a partir de los datos.

- Datos: `text_field` y `label_field`.
- Extra: `pip install 'blackcode[llm]'`.

### `embeddings` — búsqueda semántica

Afina un bi-encoder con sentence-transformers sobre pares ancla/positivo.

- Datos: `text_field` (ancla) y `options.positive_field` (positivo).
- Extra: `pip install 'blackcode[embeddings]'`.

### `dpo` — alineación de preferencias

DPO u ORPO con trl. Cada registro tiene un prompt, una respuesta preferida y
una rechazada.

- Datos: campos `prompt`, `chosen`, `rejected` (renombrables vía `data.options`).
- `options`: `method` (`dpo` u `orpo`).
- Extra: `pip install 'blackcode[llm]'`.

## Crear tu propio trainer

Un trainer es una subclase de `BaseTrainer` con tres métodos:

```python
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult

@register_trainer("mi-backend")
class MiTrainer(BaseTrainer):
    def prepare(self, dataset):
        # valida y divide en (entrenamiento, validación)
        return dataset.split(self.config.data.validation_split,
                             self.config.train.seed)

    def train(self, train_ds, val_ds) -> TrainResult:
        # entrena y guarda el modelo en self.config.train.output_dir
        ...

    def evaluate(self, val_ds, metrics) -> dict:
        # devuelve un dict de métricas
        ...
```

Pautas:

1. Importa las dependencias pesadas **dentro** de los métodos, no arriba del
   módulo, para que el núcleo siga siendo ligero.
2. Si falta un extra, lanza `ImportError` con el comando exacto de instalación.
3. Regístralo con `@register_trainer` e impórtalo en
   `src/blackcode/trainers/__init__.py` (o publícalo como paquete aparte vía
   el entry point `blackcode.trainers`).

Úsalo con `train.trainer: mi-backend`.
