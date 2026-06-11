# Instalación

## Instalador de una línea (recomendado)

Para macOS y Linux (en Windows, usa WSL):

```bash
curl -fsSL https://blackmind.cl/blackcode/install.sh | bash
```

Espejo en GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/blackmind-cl/blackcode/main/install.sh | bash
```

El instalador hace todo el trabajo, **incluyendo instalar lo que falte**:

1. Detecta tu sistema (macOS / Linux) y arquitectura.
2. Si falta Homebrew (en macOS), lo instala.
3. Si falta Python ≥ 3.10 o git, los instala con el gestor de paquetes de tu
   sistema (`brew` / `apt` / `dnf` / `yum` / `pacman` / `zypper` / `apk`).
4. Crea un entorno aislado en `~/.blackcode/venv` (no toca tu Python).
5. Instala el núcleo de Blackcode (~20 MB).
6. Deja el comando `blackcode` en `~/.local/bin`.

Variables opcionales:

| Variable | Por defecto | Para qué |
|---|---|---|
| `BLACKCODE_HOME` | `~/.blackcode` | Dónde vive el entorno aislado. |
| `BLACKCODE_REF` | `main` | Rama, tag o commit a instalar. |
| `BLACKCODE_BIN` | `~/.local/bin` | Dónde dejar el ejecutable. |

## Con pip

Si ya gestionas tus propios entornos:

```bash
pip install blackcode              # núcleo ligero (solo PyYAML)
pip install 'blackcode[llm]'       # + fine-tuning de LLMs
pip install 'blackcode[all]'       # todo
```

Para desarrollo, desde la raíz del repo: `pip install -e '.[all,dev]'`.

## Los extras se instalan solos

El núcleo de Blackcode es deliberadamente ligero. Las dependencias pesadas
(torch, transformers, scikit-learn, FastAPI, …) son **extras opcionales** que
se instalan automáticamente la primera vez que una acción los necesita:

```
$ blackcode train
⟫ Falta 'blackcode[llm]'. Instalando…
```

No hay que preguntarse "¿qué tengo que instalar para X?": invocar la acción
ya es el consentimiento, y Blackcode se encarga. Es la única salida a
internet del motor (pip), siempre en respuesta a una acción tuya.

Extras disponibles:

| Extra | Habilita |
|---|---|
| `llm` | Fine-tuning de LLMs (torch, transformers, peft, trl) y DPO/ORPO |
| `sklearn` | ML clásico (clasificación/regresión) |
| `embeddings` | Modelos de embeddings (sentence-transformers) |
| `serve` | Servidor de inferencia (FastAPI + Uvicorn) |
| `onnx` | Exportación a ONNX (optimum) |
| `pdf`, `docx`, `web`, `ocr` | Extractores de documentos (`ingest` los agrupa) |
| `postgres`, `mysql`, `mssql`, `oracle` | Conectores SQL |
| `notion`, `confluence`, `gdocs` | Conectores de plataformas de conocimiento |
| `dev` | pytest + ruff |
| `all` | Todo lo anterior |

## Desinstalar

```bash
rm -rf ~/.blackcode ~/.local/bin/blackcode
```

(Si instalaste con pip: `pip uninstall blackcode`.)
