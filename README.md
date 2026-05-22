# Blackcode

**Motor open source de entrenamiento e inferencia de IA, local y privado.**

Blackcode es para desarrolladores, equipos y empresas que **no quieren depender de
grandes LLMs en la nube**: quieren entrenar y servir sus propios modelos, con
sus propios datos, en su propio hardware. Todo corre en local. Cero telemetría.
Los datos nunca salen de la máquina.

---

## Por qué

Hoy entrenar o adaptar IA suele implicar pegar varias piezas (preparación de
datos, fine-tuning eficiente, evaluación, servidor de inferencia) y reescribir
ese pegamento en cada proyecto. Blackcode unifica ese flujo en **una arquitectura
de plugins** con una sola configuración declarativa:

```
datos  ->  entrenamiento  ->  evaluación  ->  servicio (API local)
```

El mismo flujo sirve para:

- **Fine-tuning de LLMs open source** (Llama, Mistral, Qwen…) con QLoRA / LoRA.
- **ML clásico** (clasificación / regresión con scikit-learn) — funciona en CPU.
- **Tus propios trainers**, registrados como plugins sin tocar el núcleo.

## Principios de diseño

- **Privacidad por defecto.** Sin llamadas salientes ni reporting. `report_to=[]`.
- **Núcleo ligero.** `pip install blackcode` solo trae PyYAML. Lo pesado
  (torch, transformers, sklearn) son *extras* opcionales.
- **Hardware autodetectado.** Blackcode inspecciona CUDA/MPS/CPU y VRAM y elige
  la estrategia (`full` / `lora` / `qlora` / `cpu`) — o impón la tuya.
- **Extensible.** Añadir un backend es registrar una clase con un decorador
  o un entry point.
- **Reproducible.** Una ejecución = un YAML versionable en git.

## Instalación

**Una sola línea** (macOS y Linux; en Windows usa WSL):

```bash
curl -fsSL https://blackmind.cl/blackcode/install.sh | bash
```

El instalador hace todo el trabajo: si te falta Homebrew (en macOS),
Python ≥ 3.10 o git, **los instala** con el gestor de paquetes de tu
sistema (`brew` / `apt` / `dnf` / `pacman` / `zypper` / `apk`). Luego
crea un entorno aislado en `~/.blackcode/venv`, deja el comando
`blackcode` en `~/.local/bin` y *no* baja dependencias pesadas: torch,
transformers y demás se auto-instalan la primera vez que los uses.
Si prefieres GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/blackmind-cl/blackcode/main/install.sh | bash
```

**Con pip** (si ya gestionas tus propios entornos):

```bash
# Núcleo (ligero)
pip install blackcode

# Con fine-tuning de LLMs
pip install 'blackcode[llm]'

# Con ML clásico y/o servidor
pip install 'blackcode[sklearn]' 'blackcode[serve]'

# Todo
pip install 'blackcode[all]'
```

(En desarrollo: `pip install -e '.[all,dev]'` desde la raíz del repo.)

## Uso rápido

```bash
# 1. ¿Qué hardware tengo y qué estrategia me conviene?
blackcode doctor

# 2. Crear un proyecto (carpeta autónoma con su configuración)
blackcode init mi-proyecto
cd mi-proyecto

# 3. Entrenar (carga datos -> entrena -> evalúa)
blackcode train

# 4. Servir el modelo con una API compatible con OpenAI
blackcode serve
```

### Como librería

```python
from blackcode import load_config, Pipeline

result = Pipeline(load_config("config.yaml")).run()
print(result.train.metrics, result.eval_metrics)
```

## Configuración

Todo se describe en un único YAML (ver `examples/`):

```yaml
name: soporte-bot
data:
  path: ./data/tickets.jsonl
  text_field: text
  validation_split: 0.1
train:
  trainer: llm
  model: Qwen/Qwen2.5-0.5B
  strategy: auto          # detecta full/lora/qlora/cpu por VRAM
  epochs: 3
evaluate:
  metrics: [loss, perplexity]
serve:
  port: 8000
```

## Extender con tu propio trainer

```python
from blackcode.registry import register_trainer
from blackcode.trainers.base import BaseTrainer, TrainResult

@register_trainer("mi-backend")
class MiTrainer(BaseTrainer):
    def prepare(self, dataset): ...
    def train(self, train_ds, val_ds) -> TrainResult: ...
    def evaluate(self, val_ds, metrics) -> dict: ...
```

Úsalo con `train.trainer: mi-backend`. También puedes publicarlo como paquete
independiente vía el entry point `blackcode.trainers`.

## Documentación

Guías y tutoriales en [`docs/`](docs/):

- [Primeros pasos](docs/getting-started.md) — instalar, entrenar y servir.
- [De documentos a un asistente](docs/ingest.md) — ingerir documentos y generar Q&A.
- [Configuración](docs/configuration.md) — referencia del archivo YAML.
- [Trainers](docs/trainers.md) — backends de entrenamiento y cómo crear el tuyo.
- [Exportadores](docs/exporters.md) — exportar a GGUF, ONNX o pesos fusionados.

## Estado

Alpha. La arquitectura y la API del core son estables; los trainers concretos
evolucionan. Ver [ROADMAP.md](ROADMAP.md).

## Contribuir

Lee [CONTRIBUTING.md](CONTRIBUTING.md). Las contribuciones de nuevos trainers,
exportadores (GGUF/ONNX) y métricas son especialmente bienvenidas.

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
