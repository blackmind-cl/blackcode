# Servir el modelo

`blackcode serve` expone el modelo entrenado en una API local **compatible con
el formato de OpenAI** (`/v1/chat/completions`). Cualquier herramienta que ya
hable ese protocolo (SDKs, interfaces de chat, editores) puede apuntar a tu
modelo sin reescribir nada. Todo corre en local; no hay llamadas salientes.

## Arrancar el servidor

Desde la carpeta del proyecto (usa `./blackcode.yaml`):

```bash
blackcode serve
```

O con una configuración explícita:

```bash
blackcode serve ruta/a/blackcode.yaml
```

El servidor carga el modelo de `train.output_dir` y queda escuchando:

```
Sirviendo 'mi-chatbot' en http://127.0.0.1:8000 · API estilo OpenAI · docs en /docs
```

La primera vez instalará solo el extra `serve` (FastAPI + Uvicorn) de forma
automática.

## Configuración

En el bloque `serve` del YAML:

```yaml
serve:
  host: 127.0.0.1   # por defecto solo localhost; usa 0.0.0.0 para exponer en la red
  port: 8000
```

> **Privacidad**: el valor por defecto `127.0.0.1` hace que el modelo solo sea
> accesible desde tu máquina. Cambia a `0.0.0.0` únicamente si quieres
> servirlo a otras máquinas de tu red, y entiende lo que eso implica.

## Usar la API

Con `curl`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "¿Qué servicios ofrece la empresa?"}],
    "max_tokens": 256
  }'
```

Con el SDK de OpenAI (Python):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="no-se-usa")
out = client.chat.completions.create(
    model="local",  # el nombre se ignora; siempre responde tu modelo
    messages=[{"role": "user", "content": "Hola"}],
)
print(out.choices[0].message.content)
```

También hay un endpoint de salud (`GET /health`) y la documentación
interactiva de FastAPI en `http://127.0.0.1:8000/docs`.

## Parámetros de generación

| Campo | Por defecto | Notas |
|---|---|---|
| `max_tokens` | 256 | Longitud máxima de la respuesta. |
| `temperature` | 0.0 | `0` = greedy (determinista). Sube a `0.7` para variedad. |

Por defecto la generación es **greedy**: determinista y más estable para
modelos pequeños o con poco entrenamiento. Si el modelo produce logits
inestables (NaN/inf, típico de un fine-tune con muy pocos datos), el servidor
reintenta automáticamente en modo greedy y lo deja registrado en el log.

## Detalles de implementación

- El prompt se construye con la **plantilla de chat nativa** del modelo
  (Qwen, Llama, …) vía `tokenizer.apply_chat_template`; si el modelo no tiene
  plantilla se usa un formato `role: contenido` simple.
- El modelo se carga en **fp32**: estable en Apple Silicon (donde bf16 puede
  dar NaN) y asumible en memoria para los tamaños que afina Blackcode.
- El dispositivo se autodetecta (CUDA / MPS / CPU), como en el resto del
  motor.
