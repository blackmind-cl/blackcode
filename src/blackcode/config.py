"""Configuración declarativa de una ejecución (run).

Una ejecución se describe en un único archivo YAML con cuatro bloques:
`data`, `train`, `evaluate` y `serve`. Esto mantiene el flujo reproducible
y versionable en git junto al resto del proyecto del usuario.
"""

from __future__ import annotations

import types
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """Error de configuración con un mensaje accionable para el usuario."""


@dataclass
class DataConfig:
    path: str
    format: str = "auto"          # auto | jsonl | json | csv | hf
    text_field: str = "text"
    label_field: str | None = None
    instruction_template: str | None = None  # p.ej. "### {instruction}\n{output}"
    validation_split: float = 0.1
    max_samples: int | None = None
    augment: int = 0              # nº de variantes sintéticas por registro
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    trainer: str                  # nombre registrado, p.ej. "llm" o "sklearn"
    model: str                    # id de modelo (HF hub) o estimador
    output_dir: str = "./blackcode-output"
    strategy: str = "auto"        # auto | full | lora | qlora | cpu
    epochs: float = 3.0
    batch_size: int = 0           # 0 = autodetectar
    learning_rate: float | None = None  # None = autodetectar según estrategia
    max_seq_length: int = 1024
    seed: int = 42
    resume: bool = False          # reanudar desde el último checkpoint
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluateConfig:
    metrics: list[str] = field(default_factory=lambda: ["loss"])
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api: str = "openai"           # API local compatible con el formato OpenAI
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    formats: list[str] = field(default_factory=list)  # p.ej. ["merge-lora", "gguf"]
    output_dir: str = "./blackcode-export"
    quantization: str | None = None  # tipo de cuantización (p.ej. "q4_k_m" en GGUF)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    name: str
    data: DataConfig
    train: TrainConfig
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        if not isinstance(raw, dict):
            raise ConfigError(
                "La configuración debe ser un mapa YAML en el nivel superior."
            )
        unknown = sorted(set(raw) - _VALID_BLOCKS)
        if unknown:
            raise ConfigError(
                f"Bloque(s) no reconocido(s): {', '.join(unknown)}. "
                f"Bloques válidos: {', '.join(sorted(_VALID_BLOCKS))}."
            )
        if "data" not in raw or "train" not in raw:
            raise ConfigError(
                "La config debe incluir al menos los bloques 'data' y 'train'."
            )
        name = raw.get("name", "unnamed-run")
        if not isinstance(name, str):
            raise ConfigError("La clave 'name' debe ser una cadena de texto.")
        config = cls(
            name=name,
            data=_build(DataConfig, raw["data"], "data"),
            train=_build(TrainConfig, raw["train"], "train"),
            evaluate=_build(EvaluateConfig, raw.get("evaluate", {}), "evaluate"),
            serve=_build(ServeConfig, raw.get("serve", {}), "serve"),
            export=_build(ExportConfig, raw.get("export", {}), "export"),
        )
        _check_choice("data.format", config.data.format, _DATA_FORMATS)
        _check_choice("train.strategy", config.train.strategy, _STRATEGIES)
        return config


_VALID_BLOCKS = {"name", "data", "train", "evaluate", "serve", "export"}
_DATA_FORMATS = {"auto", "jsonl", "json", "csv", "hf"}
_STRATEGIES = {"auto", "full", "lora", "qlora", "cpu"}


def _type_name(expected: Any) -> str:
    """Nombre legible de un tipo para los mensajes de error."""
    return getattr(expected, "__name__", str(expected).replace("typing.", ""))


def _type_ok(value: Any, expected: Any) -> bool:
    """Comprueba (de forma laxa) que `value` encaja en la anotación `expected`."""
    if expected is Any:
        return True
    origin = get_origin(expected)
    if origin is Union or origin is types.UnionType:  # X | None y Union[X, None]
        return any(_type_ok(value, arg) for arg in get_args(expected))
    if origin is not None:  # genéricos: list[...], dict[...], etc.
        return isinstance(value, origin)
    if expected is type(None):
        return value is None
    if expected is float:  # un entero de YAML es un float aceptable
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, type):
        return isinstance(value, expected)
    return True


def _build(dc_type: type, raw: Any, block: str):
    """Instancia un dataclass de config validando claves y tipos."""
    if not isinstance(raw, dict):
        raise ConfigError(
            f"El bloque '{block}' debe ser un mapa de clave/valor, "
            f"no {type(raw).__name__}."
        )
    spec = {f.name: f for f in fields(dc_type)}
    hints = get_type_hints(dc_type)

    unknown = sorted(set(raw) - set(spec))
    if unknown:
        raise ConfigError(
            f"Clave(s) no reconocida(s) en '{block}': {', '.join(unknown)}. "
            f"Claves válidas: {', '.join(sorted(spec))}."
        )
    missing = sorted(
        name
        for name, f in spec.items()
        if name not in raw
        and f.default is MISSING
        and f.default_factory is MISSING
    )
    if missing:
        raise ConfigError(
            f"Falta(n) clave(s) obligatoria(s) en '{block}': {', '.join(missing)}."
        )
    for key, value in raw.items():
        expected = hints.get(key)
        if expected is not None and not _type_ok(value, expected):
            raise ConfigError(
                f"La clave '{block}.{key}' debe ser de tipo {_type_name(expected)}, "
                f"no {type(value).__name__}."
            )
    return dc_type(**raw)


def _check_choice(key: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ConfigError(
            f"Valor inválido para '{key}': '{value}'. "
            f"Opciones: {', '.join(sorted(choices))}."
        )


def load_config(path: str | Path) -> RunConfig:
    """Carga y valida un archivo de configuración YAML."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Falta PyYAML. Instala con: pip install blackcode (incluye PyYAML)"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de config: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RunConfig.from_dict(raw)
