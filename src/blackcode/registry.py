"""Registro de plugins.

Hace el motor extensible: cualquier paquete puede registrar un trainer o un
exportador con un decorador (`@register_trainer` / `@register_exporter`) o vía
entry points (`blackcode.trainers` / `blackcode.exporters`). El núcleo no conoce los
backends concretos; los resuelve por nombre en tiempo de ejecución.
"""

from __future__ import annotations

from importlib import metadata
from typing import Callable


class _Registry:
    """Registro genérico de plugins resolubles por nombre."""

    def __init__(self, kind: str, entrypoint_group: str):
        self._kind = kind
        self._group = entrypoint_group
        self._items: dict[str, type] = {}
        self._entrypoints_loaded = False

    def register(self, name: str) -> Callable[[type], type]:
        """Decorador que registra una clase bajo `name`."""

        def decorator(cls: type) -> type:
            key = name.lower()
            if key in self._items and self._items[key] is not cls:
                raise ValueError(
                    f"Ya existe un {self._kind} registrado como '{name}'"
                )
            self._items[key] = cls
            return cls

        return decorator

    def _load_entrypoints(self) -> None:
        """Carga plugins expuestos por otros paquetes vía entry points."""
        if self._entrypoints_loaded:
            return
        self._entrypoints_loaded = True
        try:
            eps = metadata.entry_points()
            group = (
                eps.select(group=self._group)
                if hasattr(eps, "select")
                else eps.get(self._group, [])  # type: ignore[attr-defined]
            )
            for ep in group:
                try:
                    self._items.setdefault(ep.name.lower(), ep.load())
                except Exception:  # noqa: BLE001 - un plugin roto no tumba el core
                    continue
        except Exception:  # noqa: BLE001
            pass

    def get(self, name: str) -> type:
        """Devuelve la clase registrada bajo `name`."""
        self._load_entrypoints()
        key = name.lower()
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "(ninguno)"
            raise KeyError(
                f"{self._kind.capitalize()} '{name}' no encontrado. "
                f"Disponibles: {available}"
            )
        return self._items[key]

    def names(self) -> list[str]:
        """Lista los nombres registrados."""
        self._load_entrypoints()
        return sorted(self._items)


_trainers = _Registry("trainer", "blackcode.trainers")
_exporters = _Registry("exporter", "blackcode.exporters")
_extractors = _Registry("extractor", "blackcode.extractors")


def register_trainer(name: str) -> Callable[[type], type]:
    """Decorador que registra una clase de trainer bajo `name`."""
    return _trainers.register(name)


def get_trainer(name: str) -> type:
    """Devuelve la clase de trainer registrada bajo `name`."""
    return _trainers.get(name)


def list_trainers() -> list[str]:
    """Lista los nombres de trainers disponibles."""
    return _trainers.names()


def register_exporter(name: str) -> Callable[[type], type]:
    """Decorador que registra una clase de exportador bajo `name`."""
    return _exporters.register(name)


def get_exporter(name: str) -> type:
    """Devuelve la clase de exportador registrada bajo `name`."""
    return _exporters.get(name)


def list_exporters() -> list[str]:
    """Lista los nombres de exportadores disponibles."""
    return _exporters.names()


def register_extractor(name: str) -> Callable[[type], type]:
    """Decorador que registra una clase de extractor de documentos bajo `name`."""
    return _extractors.register(name)


def get_extractor(name: str) -> type:
    """Devuelve la clase de extractor registrada bajo `name`."""
    return _extractors.get(name)


def list_extractors() -> list[str]:
    """Lista los nombres de extractores disponibles."""
    return _extractors.names()
