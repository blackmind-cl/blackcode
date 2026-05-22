"""Blackcode: motor open source de entrenamiento e inferencia de IA local.

Diseñado para correr íntegramente en la máquina del usuario, sin telemetría
ni dependencia de APIs externas. Arquitectura de plugins: el mismo flujo
(datos -> entrenamiento -> evaluación -> servicio) sirve para fine-tuning de
LLMs, modelos pequeños o ML clásico.
"""

from blackcode.config import ConfigError, RunConfig, load_config
from blackcode.hardware import HardwareProfile, detect_hardware
from blackcode.pipeline import Pipeline
from blackcode.registry import (
    get_exporter,
    get_extractor,
    get_trainer,
    list_exporters,
    list_extractors,
    list_trainers,
    register_exporter,
    register_extractor,
    register_trainer,
)

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "load_config",
    "ConfigError",
    "HardwareProfile",
    "detect_hardware",
    "Pipeline",
    "register_trainer",
    "get_trainer",
    "list_trainers",
    "register_exporter",
    "get_exporter",
    "list_exporters",
    "register_extractor",
    "get_extractor",
    "list_extractors",
    "__version__",
]
