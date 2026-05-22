"""Generación de datos de instrucción (pares pregunta/respuesta).

A partir de un corpus de texto (el JSONL de `blackcode ingest`), un LLM
generador local produce pares pregunta/respuesta que sirven de datos de
instruction tuning. Todo corre en local; el dispositivo (CPU/GPU) es
configurable.

El modelo generador requiere el extra:  pip install 'blackcode[llm]'
"""

from __future__ import annotations

import re

from blackcode.hardware import Accelerator, detect_hardware
from blackcode.install import ensure_extra
from blackcode.log import get_logger

_log = get_logger("generate")

_PROMPT = """\
A partir del siguiente texto, genera {n} pares pregunta–respuesta basados
únicamente en él. Usa exactamente este formato, dos líneas por par:

P: <pregunta>
R: <respuesta>

No añadas nada más. Texto:

{texto}
"""

# Prefijos aceptados al parsear la salida del modelo. Aceptamos varios
# alias y formatos (numeración, viñetas) para tolerar la variabilidad de
# modelos pequeños.
_Q_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?"
    r"(?:pregunta|question|p|q)\b\s*[:.\-]\s*",
    re.IGNORECASE,
)
_A_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?"
    r"(?:respuesta|answer|r|a)\b\s*[:.\-]\s*",
    re.IGNORECASE,
)


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Divide el texto en trozos de hasta `max_chars`, respetando párrafos."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def parse_qa(response: str) -> list[dict]:
    """Extrae pares pregunta/respuesta de la salida del modelo.

    Tolerante con variantes habituales: `P:`/`R:`, `Pregunta:`/`Respuesta:`,
    `Q:`/`A:`, con o sin numeración (`1.`, `1)`) o viñetas (`-`, `*`).
    """
    pairs: list[dict] = []
    question: str | None = None
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        q_match = _Q_PREFIX.match(line)
        a_match = _A_PREFIX.match(line)
        if q_match:
            question = line[q_match.end():].strip()
        elif a_match and question:
            answer = line[a_match.end():].strip()
            if question and answer:
                pairs.append({"instruction": question, "output": answer})
            question = None
    return pairs


def _estimate_model_size(model_id: str) -> float | None:
    """Estima los miles de millones de parámetros del modelo.

    Busca primero en el catálogo; si no aparece usa una heurística sobre el
    propio id (`7B`, `1.5B`, …). Devuelve `None` si no se puede determinar.
    """
    from blackcode.catalog import CATALOG

    for model in CATALOG:
        if model.id == model_id:
            return model.params_billions
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    return float(match.group(1)) if match else None


def _should_warn_slow(model_id: str, device: str) -> bool:
    """¿Conviene avisar de generación lenta para esta combinación?"""
    if device != "cpu":
        return False
    size = _estimate_model_size(model_id)
    return size is not None and size > 1.0


def _maybe_warn_slow_inference(model_id: str, device: str) -> None:
    """Avisa si el modelo es grande y vamos a correr en CPU."""
    if not _should_warn_slow(model_id, device):
        return
    size = _estimate_model_size(model_id)
    _log.warning(
        "Vas a generar Q&A con un modelo de ~%sB en CPU. Esto es lento "
        "(horas, no minutos). Considera un modelo más pequeño "
        "(p.ej. Qwen/Qwen2.5-0.5B-Instruct) o correr con --device cuda.",
        f"{size:g}",
    )


def resolve_device(device: str) -> str:
    """Resuelve `auto` al dispositivo detectado; respeta cpu/cuda/mps."""
    if device != "auto":
        return device
    accelerator = detect_hardware().accelerator
    return {Accelerator.CUDA: "cuda", Accelerator.MPS: "mps"}.get(
        accelerator, "cpu"
    )


def generate_qa(
    records: list[dict],
    model_id: str,
    device: str = "auto",
    per_chunk: int = 3,
    chunk_size: int = 2000,
) -> list[dict]:
    """Genera pares Q&A desde registros `{text, source}` con un LLM local."""
    ensure_extra("llm", "transformers")
    from transformers import pipeline  # type: ignore

    resolved = resolve_device(device)
    _maybe_warn_slow_inference(model_id, resolved)
    _log.info("Generador: %s · dispositivo: %s", model_id, resolved)
    generator = pipeline("text-generation", model=model_id, device=resolved)

    dataset: list[dict] = []
    for record in records:
        for chunk in chunk_text(record["text"], chunk_size):
            prompt = _format_chat_prompt(
                generator.tokenizer, _PROMPT.format(n=per_chunk, texto=chunk)
            )
            output = generator(
                prompt, max_new_tokens=512, do_sample=False, return_full_text=False
            )
            raw_text = output[0]["generated_text"]
            pairs = parse_qa(raw_text)
            if not pairs:
                _log.warning(
                    "El modelo no devolvió pares en el formato esperado para "
                    "un fragmento. Ejecuta con -v para ver la salida cruda."
                )
                _log.debug("Salida cruda del modelo:\n%s", raw_text)
            for pair in pairs:
                dataset.append({**pair, "source": record.get("source", "")})
    _log.info("Generados %d pares Q&A desde %d documentos", len(dataset), len(records))
    return dataset


def _format_chat_prompt(tokenizer, user_message: str) -> str:
    """Aplica la plantilla de chat del modelo si está disponible.

    Los modelos Instruct/Chat siguen instrucciones mucho mejor cuando el
    prompt se formatea con su plantilla nativa (`<|im_start|>…` en Qwen,
    etc.). Si el tokenizer no tiene plantilla, devolvemos el texto plano.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 - cualquier fallo cae al modo texto plano
        return user_message
