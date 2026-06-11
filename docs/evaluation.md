# Evaluación y métricas

## `blackcode eval` — evaluar sin reentrenar

Evalúa el modelo ya entrenado que está en `train.output_dir`, reutilizando la
misma división de validación (determinista por `train.seed`) que se usó al
entrenar:

```bash
blackcode eval            # usa ./blackcode.yaml
blackcode eval ruta/a/blackcode.yaml
```

Las métricas se declaran en el bloque `evaluate` del YAML:

```yaml
evaluate:
  metrics:
    - loss
    - perplexity
```

Métricas disponibles por trainer:

| Trainer | Métricas |
|---|---|
| `llm` | `loss`, `perplexity` (sobre una muestra de validación) |
| `sklearn` | `accuracy`, `precision`, `recall`, `f1` |
| `transformer-classifier` | `accuracy` |
| `embeddings` | `cosine_similarity` (pares ancla/positivo) |
| `dpo` | — (la alineación de preferencias se evalúa con juicio humano) |

`blackcode train` ejecuta la evaluación automáticamente al terminar, salvo
que pases `--no-eval`.

## `blackcode dashboard` — tablero de métricas

Cada entrenamiento escribe un `blackcode_metrics.json` en su `output_dir`.
El comando `dashboard` los recolecta y genera un **HTML estático** (sin
servidor, sin dependencias) para comparar ejecuciones:

```bash
blackcode dashboard                  # escanea el directorio actual
blackcode dashboard run1/ run2/      # o carpetas concretas
blackcode dashboard --output informe.html
```

Abre el HTML resultante en el navegador. Útil para comparar estrategias
(LoRA vs QLoRA), learning rates o tamaños de modelo entre proyectos.
