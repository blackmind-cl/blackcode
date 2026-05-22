"""Asistente interactivo de configuración para `blackcode init`.

Hace preguntas guiadas — con una breve explicación de cada opción — y devuelve
un diccionario de configuración. Sin dependencias: solo `input()` y códigos
ANSI. En una terminal interactiva muestra color y animaciones simuladas
(tecleo, spinners); al redirigir la salida o en los tests escribe texto plano
e instantáneo.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

try:  # entrada de teclado cruda (flechas); solo disponible en Unix
    import termios
    import tty

    _RAW_INPUT = True
except ImportError:  # pragma: no cover - Windows u otras plataformas
    _RAW_INPUT = False

# Metadatos por tipo de proyecto: trainer, modelo base, métricas y la `task`
# que se usa para filtrar el catálogo de modelos.
_PROJECT_TYPES: dict[str, dict] = {
    "chatbot": {
        "trainer": "llm",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "metrics": ["loss", "perplexity"],
        "desc": "afina un LLM conversacional con tus datos",
        "catalog_task": "llm",
    },
    "clasificador": {
        "trainer": "transformer-classifier",
        "model": "distilbert-base-multilingual-cased",
        "metrics": ["accuracy"],
        "desc": "clasifica texto en categorías",
        "catalog_task": "classifier",
    },
    "embeddings": {
        "trainer": "embeddings",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "metrics": ["cosine_similarity"],
        "desc": "afina un modelo de búsqueda semántica",
        "catalog_task": "embeddings",
    },
}

# Valor que devuelve el menú cuando el usuario elige cancelar el asistente.
_EXIT = "salir"

# Códigos ANSI (solo se emiten en modo animado).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"


class _Console:
    """Capa de presentación: color y animación si hay una terminal."""

    def __init__(
        self,
        animate: bool,
        output_fn: Callable[[str], None],
        input_fn: Callable[[str], str],
    ):
        self.animate = animate
        self._out = output_fn
        self._in = input_fn

    def _write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def line(self, text: str = "") -> None:
        """Una línea de texto plano, instantánea."""
        if self.animate:
            self._write(text + "\n")
        else:
            self._out(text)

    def typed(self, text: str, delay: float = 0.014) -> None:
        """Una línea con efecto de tecleo simulado."""
        if not self.animate:
            self._out(text)
            return
        for char in text:
            self._write(char)
            time.sleep(delay)
        self._write("\n")

    def spin(self, text: str, seconds: float = 0.9) -> None:
        """Spinner simulado durante `seconds`, rematado con un check."""
        if not self.animate:
            self._out(text)
            return
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        deadline, i = time.time() + seconds, 0
        while time.time() < deadline:
            frame = frames[i % len(frames)]
            self._write(f"\r  {_CYAN}{frame}{_RESET} {_DIM}{text}…{_RESET}")
            time.sleep(0.07)
            i += 1
        self._write(f"\r  {_GREEN}✓{_RESET} {_DIM}{text}{_RESET}    \n")

    def banner(self) -> None:
        if not self.animate:
            self._out("Asistente de configuración de Blackcode")
            return
        self.line()
        self.line(
            f"  {_BOLD}{_CYAN}✦ Blackcode{_RESET}"
            f"  {_DIM}·  asistente de proyecto{_RESET}"
        )
        self.line(f"  {_CYAN}{'─' * 38}{_RESET}")

    def heading(self, text: str) -> None:
        if self.animate:
            self.line(f"\n{_BOLD}{_MAGENTA}▸{_RESET} {_BOLD}{text}{_RESET}")
        else:
            self.line(f"\n· {text}")

    def hint(self, text: str) -> None:
        self.line(f"  {_DIM}{text}{_RESET}" if self.animate else f"  {text}")

    def option(
        self, index: int, value: str, desc: str, marker: str = " "
    ) -> None:
        starred = marker.strip() == "★"
        if self.animate:
            mark = f"{_YELLOW}★{_RESET}" if starred else " "
            self.line(
                f"  {mark} {_CYAN}{index}{_RESET}) {_BOLD}{value}{_RESET} "
                f"{_DIM}— {desc}{_RESET}"
            )
        else:
            self.line(f"  {'★' if starred else ' '} {index}) {value} — {desc}")

    def warn(self, text: str) -> None:
        self.line(f"  {_YELLOW}{text}{_RESET}" if self.animate else f"  {text}")

    def ask(self, default: str) -> str:
        if self.animate:
            prompt = f"  {_CYAN}❯{_RESET} {_DIM}[{default}]{_RESET} "
        else:
            prompt = f"  [{default}] > "
        return self._in(prompt)

    def menu(self, options: list) -> str:
        """Elige una opción: con flechas en terminal, por número si no."""
        if self.animate and _RAW_INPUT and sys.stdin.isatty():
            return self._arrow_menu(options)
        return self._numbered_menu(options)

    def _numbered_menu(self, options: list) -> str:
        for index, (value, desc, marker) in enumerate(options, 1):
            self.option(index, value, desc, marker)
        while True:
            raw = self.ask("1").strip()
            if not raw:
                return options[0][0]
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1][0]
            self.warn("Opción no válida; intenta de nuevo.")

    def _draw_menu(self, options: list, selected: int, first: bool) -> None:
        if not first:
            self._write(f"\033[{len(options)}A")  # subir hasta la 1ª opción
        for index, (value, desc, marker) in enumerate(options):
            mark = f"{_YELLOW}★{_RESET}" if marker.strip() == "★" else " "
            if index == selected:
                row = (
                    f"  {mark} {_CYAN}❯{_RESET} {_BOLD}{_CYAN}{value}{_RESET} "
                    f"{_DIM}— {desc}{_RESET}"
                )
            else:
                row = f"  {mark}   {_DIM}{value} — {desc}{_RESET}"
            self._write(f"\r\033[K{row}\n")

    def _arrow_menu(self, options: list) -> str:
        self.line(f"  {_DIM}↑/↓ para moverte · Enter para elegir{_RESET}")
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        selected = 0
        self._write("\033[?25l")  # ocultar el cursor
        try:
            tty.setcbreak(fd)
            self._draw_menu(options, selected, first=True)
            while True:
                key = _read_key(sys.stdin.read)
                if key == "up":
                    selected = (selected - 1) % len(options)
                elif key == "down":
                    selected = (selected + 1) % len(options)
                elif key == "enter":
                    break
                self._draw_menu(options, selected, first=False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self._write("\033[?25h")  # restaurar el cursor
        return options[selected][0]


def _read_key(read_fn: Callable[[int], str]) -> str:
    """Interpreta una pulsación: 'up', 'down', 'enter' u 'other'.

    Asume que la terminal ya está en modo cbreak. `read_fn(n)` lee n caracteres.
    Acepta las flechas ↑/↓ y, como alias, las teclas k/j.
    """
    char = read_fn(1)
    if char in ("\r", "\n"):
        return "enter"
    if char == "\x1b":  # secuencia de escape de las flechas
        return {"[A": "up", "[B": "down"}.get(read_fn(2), "other")
    if char == "k":
        return "up"
    if char == "j":
        return "down"
    return "other"


def _as_number(value: str) -> float | int:
    """Convierte el texto en número; vuelve a 3 si no es válido."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return 3


def _ask(console: _Console, title: str, explanation: str, default: str) -> str:
    """Pregunta de texto libre, con explicación y valor por defecto."""
    console.heading(title)
    console.hint(explanation)
    return console.ask(default).strip() or default


def _ask_choice(
    console: _Console, title: str, explanation: str, options: list
) -> str:
    """Pregunta de opción múltiple. `options` = [(valor, descripción, marca)]."""
    console.heading(title)
    console.hint(explanation)
    return console.menu(options)


def _is_recommended(model_info, hardware) -> bool:
    """Si el modelo encaja cómodamente con el hardware del usuario."""
    if hardware.accelerator.value in ("cpu", "mps"):
        # Sin VRAM medible: solo los modelos pequeños son cómodos.
        return model_info.params_billions <= 1.5
    return hardware.recommend_strategy(model_info.params_billions).value != "cpu"


def _model_menu_options(task: str, hardware) -> list:
    """Construye las opciones de menú (valor, desc, marca) desde el catálogo."""
    from blackcode.catalog import CATALOG

    catalog = sorted(
        (m for m in CATALOG if m.task == task), key=lambda m: m.params_billions
    )
    options = []
    for m in catalog:
        marker = "★" if _is_recommended(m, hardware) else " "
        strategy = hardware.recommend_strategy(m.params_billions).value
        desc = f"{m.params_billions}B · estrategia {strategy} · {m.description}"
        options.append((m.id, desc, marker))
    options.append(
        ("personalizado", "escribir el id de otro modelo de Hugging Face", " ")
    )
    return options


def _ask_model(console: _Console, task: str, hardware, default: str) -> str:
    """Elige un modelo del catálogo o, si se prefiere, escribe uno personalizado."""
    choice = _ask_choice(
        console,
        "Modelo base",
        "Elige un modelo. ★ marca los recomendados para tu hardware.",
        _model_menu_options(task, hardware),
    )
    if choice == "personalizado":
        return _ask(
            console,
            "Modelo personalizado",
            "Escribe el id del modelo en Hugging Face.",
            default,
        )
    return choice


def run_wizard(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    hardware=None,
    default_name: str | None = None,
    animate: bool | None = None,
) -> dict | None:
    """Ejecuta el asistente y devuelve un diccionario de configuración.

    Devuelve `None` si el usuario elige la opción «salir». `default_name` (la
    carpeta del proyecto) se sugiere como nombre. `animate` se autodetecta: hay
    animación solo si la salida es una terminal.
    """
    if animate is None:
        animate = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    console = _Console(animate, output_fn, input_fn)

    console.banner()
    console.typed(
        "  Te haré unas preguntas para configurar tu proyecto. "
        "Pulsa Enter para aceptar el valor sugerido."
    )

    if hardware is None:
        from blackcode.hardware import detect_hardware

        hardware = detect_hardware()
    console.spin("Analizando tu hardware")
    console.hint(
        f"Detectado: {hardware.accelerator.value}, "
        f"{hardware.total_vram_gb} GB de VRAM · "
        f"estrategia sugerida: {hardware.recommend_strategy().value}"
    )

    choices = [
        (name, meta["desc"], " ") for name, meta in _PROJECT_TYPES.items()
    ]
    choices.append((_EXIT, "cancelar sin crear el proyecto", " "))
    ptype = _ask_choice(
        console,
        "Tipo de proyecto",
        "Qué quieres entrenar. Define el trainer y el modelo base por defecto.",
        choices,
    )
    if ptype == _EXIT:
        console.line()
        console.line("Asistente cancelado.")
        return None
    meta = _PROJECT_TYPES[ptype]

    name = _ask(
        console, "Nombre del proyecto",
        "Identifica esta ejecución; aparece en los logs y en las rutas.",
        default_name or f"mi-{ptype}",
    )
    data_path = _ask(
        console, "Ruta de los datos",
        "Archivo JSONL, CSV o JSON con tus ejemplos de entrenamiento.",
        "./data/train.jsonl",
    )
    model = _ask_model(console, meta["catalog_task"], hardware, meta["model"])
    epochs = _ask(
        console, "Épocas",
        "Cuántas pasadas completas se harán sobre los datos.",
        "3",
    )
    output_dir = _ask(
        console, "Carpeta de salida",
        "Dónde se guardará el modelo entrenado.",
        f"./blackcode-output/{name}",
    )

    data: dict = {"path": data_path, "validation_split": 0.1}
    if ptype == "clasificador":
        data["text_field"] = "text"
        data["label_field"] = _ask(
            console, "Campo de etiqueta",
            "Nombre del campo de cada registro que contiene la categoría.",
            "label",
        )
    elif ptype == "embeddings":
        data["text_field"] = "anchor"
        data["options"] = {"positive_field": "positive"}
    else:  # chatbot
        data["text_field"] = "text"
        # Compatibilidad lista con el JSONL que produce `generate-qa`
        # (campos `instruction` / `output`). Si tus datos vienen ya en `text`,
        # quita esta línea del YAML.
        data["instruction_template"] = (
            "### Pregunta: {instruction}\n### Respuesta: {output}"
        )

    config = {
        "name": name,
        "data": data,
        "train": {
            "trainer": meta["trainer"],
            "model": model,
            "output_dir": output_dir,
            "epochs": _as_number(epochs),
        },
        "evaluate": {"metrics": meta["metrics"]},
    }

    console.line()
    console.typed(
        f"  Voy a crear el proyecto '{name}' ({ptype}) con el modelo {model}."
    )
    console.spin("Generando la configuración del proyecto")
    return config
