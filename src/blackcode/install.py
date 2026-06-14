"""Instala extras de Blackcode bajo demanda.

Cuando una acción necesita una dependencia opcional (`blackcode[notion]`,
`blackcode[llm]`, …) que no está instalada, este módulo la trata como una
dependencia de la propia función: la instala con pip e informa al usuario.
No pregunta — la acción que el usuario inició YA implica el consentimiento.

Esta es la única ruta del proyecto que hace una llamada saliente a PyPI, y
solo en respuesta a una acción que la necesita.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import threading

# Extras ya intentados en este proceso: si la instalación falló una vez no
# volvemos a intentarla en la misma corrida. El lock serializa las llamadas
# concurrentes (p. ej. workers del server) para no lanzar dos pip a la vez.
_attempted: set[str] = set()
_lock = threading.Lock()


def ensure_extra(extra: str, *modules: str) -> None:
    """Si alguno de los `modules` falta, instala `blackcode[extra]` con pip.

    Imprime un aviso a stderr antes de instalar; pip emite su propio output.
    Lanza `ImportError` si la instalación falla.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return
    with _lock:
        # Reverifica dentro del lock: otro hilo pudo instalarlo mientras
        # esperábamos.
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        if not missing:
            return
        if extra in _attempted:
            raise ImportError(
                f"Falta blackcode[{extra}] y un intento previo en este proceso "
                f"falló. Instálalo manualmente: pip install 'blackcode[{extra}]'"
            )
        _attempted.add(extra)
        print(
            f"⟫ Falta 'blackcode[{extra}]'. Instalando…",
            file=sys.stderr,
            flush=True,
        )
        # `blackcode[extra]` funciona aunque blackcode NO esté en PyPI: pip ve
        # que el core ya está instalado (lo haya traído PyPI, git o un install
        # editable), da el requisito `blackcode` por satisfecho y solo baja las
        # dependencias del extra. No reinstala ni vuelve a clonar el core.
        # (No cambiar a `... @ git+URL`: eso forzaría un re-clone en cada extra.)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"blackcode[{extra}]"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ImportError(
                f"pip install 'blackcode[{extra}]' falló."
            ) from exc
        importlib.invalidate_caches()
