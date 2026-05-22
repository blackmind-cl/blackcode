"""Catálogo de modelos recomendados, anotados por tamaño.

Permite a `blackcode models` sugerir qué modelos puedes entrenar con tu
hardware: para cada modelo se calcula la estrategia recomendada con la misma
heurística de VRAM que usa `HardwareProfile.recommend_strategy`.
"""

from __future__ import annotations

from dataclasses import dataclass

from blackcode.hardware import HardwareProfile


@dataclass(frozen=True)
class ModelInfo:
    id: str
    params_billions: float
    task: str                # "llm" | "embeddings" | "classifier"
    description: str


CATALOG: list[ModelInfo] = [
    ModelInfo("Qwen/Qwen2.5-0.5B-Instruct", 0.5, "llm",
              "LLM diminuto: entrena hasta en CPU."),
    ModelInfo("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "llm",
              "LLM pequeño y equilibrado."),
    ModelInfo("Qwen/Qwen2.5-3B-Instruct", 3.0, "llm",
              "LLM mediano para tareas más exigentes."),
    ModelInfo("Qwen/Qwen2.5-7B-Instruct", 7.0, "llm",
              "LLM grande: QLoRA en una GPU de consumo."),
    ModelInfo("meta-llama/Llama-3.2-1B-Instruct", 1.0, "llm",
              "Llama compacto, buen punto de partida."),
    ModelInfo("meta-llama/Llama-3.2-3B-Instruct", 3.0, "llm",
              "Llama mediano."),
    ModelInfo("mistralai/Mistral-7B-Instruct-v0.3", 7.0, "llm",
              "Mistral 7B, sólido para chat."),
    ModelInfo("sentence-transformers/all-MiniLM-L6-v2", 0.02, "embeddings",
              "Embeddings rápidos y ligeros."),
    ModelInfo("sentence-transformers/all-mpnet-base-v2", 0.11, "embeddings",
              "Embeddings de mayor calidad."),
    ModelInfo("distilbert-base-multilingual-cased", 0.13, "classifier",
              "Clasificador multilingüe ligero."),
    ModelInfo("FacebookAI/xlm-roberta-base", 0.28, "classifier",
              "Clasificador multilingüe robusto."),
]


def recommend_models(
    hardware: HardwareProfile, task: str | None = None
) -> list[tuple[ModelInfo, str]]:
    """Devuelve (modelo, estrategia recomendada) para el hardware dado.

    Filtra opcionalmente por `task` ("llm", "embeddings" o "classifier").
    """
    return [
        (m, hardware.recommend_strategy(m.params_billions).value)
        for m in CATALOG
        if task is None or m.task == task
    ]
