"""Servidor de inferencia local.

Expone el modelo entrenado mediante una API compatible con el formato de
OpenAI (`/v1/chat/completions`) para que cualquier herramienta que ya hable
ese protocolo apunte a `http://localhost:8000` sin reescribir nada. Todo
corre en local; no hay llamadas salientes.

NOTA: este módulo NO usa `from __future__ import annotations`. Los modelos
Pydantic se definen dentro de `serve_model()` y necesitan que las anotaciones
se evalúen al definir las clases para que FastAPI las reconozca como cuerpo
de la petición (no como query params).

Extra requerido:  pip install 'blackcode[serve]'
"""

import time

from blackcode.config import RunConfig
from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("serving")


def serve_model(config: RunConfig) -> None:
    ensure_extra("serve", "uvicorn", "fastapi", "pydantic")
    import uvicorn  # type: ignore
    from fastapi import FastAPI  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore

    ensure_extra("llm", "torch", "transformers")
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    from blackcode.hardware import detect_hardware

    accelerator = detect_hardware().accelerator.value
    device = accelerator if accelerator in ("cuda", "mps") else "cpu"

    model_dir = config.train.output_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    # fp32 para inferencia: estable en MPS (donde bf16 puede dar NaN) y barato
    # en memoria para los tamaños que afina Blackcode.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    generator = pipeline(
        "text-generation", model=model, tokenizer=tok, device=device
    )
    _log.info("Modelo cargado en %s (fp32)", device)

    app = FastAPI(title=f"Blackcode · {config.name}", docs_url="/docs")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[Message]
        # Acotado: sin tope, una petición podría pedir miles de millones de
        # tokens y dejar la máquina generando indefinidamente.
        max_tokens: int = Field(default=256, ge=1, le=8192)
        # Greedy por defecto: determinista y más estable para modelos
        # pequeños o poco entrenados que pueden producir logits con NaN/inf.
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": config.name}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        messages_payload = [
            {"role": m.role, "content": m.content} for m in req.messages
        ]
        try:
            # Plantilla de chat del modelo (Qwen, Llama, …): formato nativo,
            # mucho más estable que concatenar 'user:\nassistant:'.
            prompt = tok.apply_chat_template(
                messages_payload, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - fallback para modelos sin plantilla
            prompt = (
                "\n".join(f"{m.role}: {m.content}" for m in req.messages)
                + "\nassistant:"
            )
        gen_kwargs = dict(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            return_full_text=False,
        )
        try:
            out = generator(prompt, **gen_kwargs)
        except RuntimeError as exc:
            # Si los logits salen con NaN/inf (modelo mal entrenado), la
            # multinomial revienta. Reintentamos greedy: argmax tolera NaN.
            if "inf" in str(exc).lower() or "nan" in str(exc).lower():
                _log.warning(
                    "El modelo produjo logits inestables (inf/NaN); probable "
                    "fine-tune con muy pocos datos. Reintento con greedy; la "
                    "respuesta puede ser incoherente."
                )
                gen_kwargs["do_sample"] = False
                out = generator(prompt, **gen_kwargs)
            else:
                raise
        text = out[0]["generated_text"].strip()
        return {
            "id": f"blackcode-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    _log.info(
        "Sirviendo '%s' en http://%s:%s · API estilo OpenAI · docs en /docs",
        config.name,
        config.serve.host,
        config.serve.port,
    )
    uvicorn.run(app, host=config.serve.host, port=config.serve.port, log_level="info")
