# Exportadores

Un *exportador* toma un modelo ya entrenado (en `train.output_dir`) y produce
un artefacto de despliegue. Igual que los trainers, son plugins resueltos por
nombre. Lista los disponibles con `blackcode exporters`.

Configura los destinos en el bloque `export` o pásalos con `--to`:

```bash
blackcode export config.yaml --to merge-lora --to gguf
```

```yaml
export:
  formats: [merge-lora, gguf]
  output_dir: ./blackcode-export
  quantization: q4_k_m
```

## Exportadores integrados

### `merge-lora` — fusionar adaptadores

Tras un fine-tuning con LoRA/QLoRA, lo guardado son los adaptadores más una
referencia al modelo base. Este exportador los fusiona en un modelo único y
autónomo, listo para servir o para convertir a otro formato.

- Requiere que el modelo se haya entrenado con estrategia `lora` o `qlora`.
- Extra: `pip install 'blackcode[llm]'`.

### `gguf` — para llama.cpp / Ollama

Convierte el modelo a GGUF. GGUF no tiene un conversor instalable vía pip:
se usa el script `convert_hf_to_gguf.py` de llama.cpp.

- Indica dónde está clonado llama.cpp con la variable de entorno
  `LLAMA_CPP_DIR` o con `export.options.llama_cpp_dir`.
- Si defines `export.quantization` (p.ej. `q4_k_m`) y `llama-quantize` está en
  el PATH, el modelo se cuantiza.

### `onnx` — para ONNX Runtime

Exporta a ONNX vía la librería `optimum`, para servir con ONNX Runtime
(CPU o GPU) sin arrastrar PyTorch.

- Extra: `pip install 'blackcode[onnx]'`.

## Crear tu propio exportador

Un exportador es una subclase de `BaseExporter` con un método `export()`:

```python
from blackcode.registry import register_exporter
from blackcode.exporters.base import BaseExporter, ExportResult

@register_exporter("mi-formato")
class MiExporter(BaseExporter):
    def export(self) -> ExportResult:
        # self.model_dir es la carpeta del modelo entrenado
        ...
        return ExportResult(format="mi-formato", path=destino, extra={})
```

Impórtalo en `src/blackcode/exporters/__init__.py`, o publícalo como paquete
aparte vía el entry point `blackcode.exporters`. Como en los trainers, importa
las dependencias pesadas de forma perezosa dentro de `export()`.
