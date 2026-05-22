#!/usr/bin/env bash
# Instalador de Blackcode.
#
# Uso:
#     curl -fsSL https://blackmind.cl/blackcode/install.sh | bash
#
# O equivalente, desde el repo:
#     curl -fsSL https://raw.githubusercontent.com/blackmind-cl/blackcode/main/install.sh | bash
#
# Que hace:
#   1. Detecta SO y arquitectura (macOS y Linux; Windows queda fuera).
#   2. Si falta Homebrew (en macOS) lo instala.
#   3. Si falta Python >= 3.10 lo instala con el gestor de paquetes
#      detectado (brew / apt / dnf / yum / pacman / zypper / apk).
#   4. Si falta git lo instala.
#   5. Crea un entorno virtual aislado en ~/.blackcode/venv.
#   6. Instala el core de blackcode desde GitHub (~20 MB).
#   7. Coloca un ejecutable 'blackcode' en ~/.local/bin.
#   8. Avisa si ~/.local/bin no esta en PATH y como anadirlo.
#
# Filosofia: si algo no esta, lo instalamos. La accion de correr este
# script YA implica el consentimiento. Para paquetes del sistema usamos
# sudo cuando hace falta. Si no hay sudo, abortamos con un mensaje claro.
#
# Los extras pesados (torch, transformers, etc.) NO se bajan aqui.
# Se auto-instalan cuando los necesitas, via blackcode.install.ensure_extra.
#
# Este archivo es ASCII puro a proposito: un instalador servido por
# 'curl | bash' debe sobrevivir cualquier servidor y codificacion.
#
# Variables de entorno opcionales:
#   BLACKCODE_HOME   raiz del install (por defecto: $HOME/.blackcode)
#   BLACKCODE_REF    rama, tag o commit del repo (por defecto: main)
#   BLACKCODE_BIN    directorio para el shim (por defecto: $HOME/.local/bin)

set -euo pipefail

# --- Configuracion ---------------------------------------------------------
BLACKCODE_HOME="${BLACKCODE_HOME:-$HOME/.blackcode}"
BLACKCODE_REF="${BLACKCODE_REF:-main}"
BLACKCODE_BIN="${BLACKCODE_BIN:-$HOME/.local/bin}"
REPO_URL="https://github.com/blackmind-cl/blackcode.git"
VENV_DIR="${BLACKCODE_HOME}/venv"

# --- UI --------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

info()    { printf "%s>>%s %s\n" "$BOLD" "$RESET" "$1"; }
success() { printf "%s ok%s %s\n" "$GREEN" "$RESET" "$1"; }
warn()    { printf "%s ! %s %s\n" "$YELLOW" "$RESET" "$1" >&2; }
err()     { printf "%s xx%s %s\n" "$RED" "$RESET" "$1" >&2; }

banner() {
    printf "%sBlackcode%s %s- instalador%s\n" "$BOLD" "$RESET" "$DIM" "$RESET"
    printf "%sMotor local de entrenamiento e inferencia de modelos IA.%s\n\n" \
        "$DIM" "$RESET"
}

# --- Privilegios -----------------------------------------------------------
# Ejecuta un comando como root si hace falta. Si ya somos root, lo corre
# directo. Si no, usa sudo. Si no hay sudo y no somos root, falla.
as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        err "Necesito instalar paquetes del sistema y no encuentro 'sudo'."
        err "Ejecuta este script como root o instala sudo primero."
        exit 1
    fi
}

# --- Deteccion de SO -------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)
            err "Sistema operativo no soportado: $(uname -s)."
            err "Blackcode corre en macOS y Linux. En Windows usa WSL."
            exit 1
            ;;
    esac
}

# --- Homebrew (solo macOS) -------------------------------------------------
ensure_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        return 0
    fi
    info "Homebrew no esta instalado. Lo instalo (gestor de paquetes de macOS)."
    # El script oficial soporta NONINTERACTIVE=1 para evitar prompts.
    NONINTERACTIVE=1 /bin/bash -c \
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Asegura que brew este en PATH para esta sesion.
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    if ! command -v brew >/dev/null 2>&1; then
        err "La instalacion de Homebrew no dejo 'brew' en PATH."
        err "Reabre la terminal y vuelve a ejecutar el instalador."
        exit 1
    fi
    success "Homebrew instalado."
}

# --- Gestor de paquetes (Linux) --------------------------------------------
detect_pkg_manager() {
    for pm in apt-get dnf yum pacman zypper apk; do
        if command -v "$pm" >/dev/null 2>&1; then
            echo "$pm"
            return 0
        fi
    done
    echo "unknown"
}

pkg_install() {
    # Instala uno o mas paquetes con el gestor detectado.
    local pm="$1"; shift
    case "$pm" in
        brew)
            brew install "$@"
            ;;
        apt-get)
            as_root apt-get update -qq
            as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
            ;;
        dnf)
            as_root dnf install -y "$@"
            ;;
        yum)
            as_root yum install -y "$@"
            ;;
        pacman)
            as_root pacman -Sy --noconfirm "$@"
            ;;
        zypper)
            as_root zypper --non-interactive install "$@"
            ;;
        apk)
            as_root apk add --no-cache "$@"
            ;;
        *)
            err "No reconozco el gestor de paquetes '$pm'."
            err "Instala manualmente: $*"
            exit 1
            ;;
    esac
}

# Devuelve el nombre del paquete de Python para el gestor dado.
python_pkg_name() {
    case "$1" in
        brew)    echo "python@3.12" ;;
        apt-get) echo "python3 python3-venv python3-pip" ;;
        dnf|yum) echo "python3 python3-pip" ;;
        pacman)  echo "python python-pip" ;;
        zypper)  echo "python3 python3-pip python3-venv" ;;
        apk)     echo "python3 py3-pip" ;;
    esac
}

# --- Deteccion de Python ---------------------------------------------------
find_python() {
    # Buscamos un interprete >= 3.10. Aceptamos python3.13, python3.12,
    # python3.11, python3.10 y python3 (si reporta >= 3.10).
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

ensure_python() {
    local pm="$1"
    if PYTHON_BIN="$(find_python)"; then
        local version
        version="$("$PYTHON_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')"
        success "Python encontrado: ${PYTHON_BIN} (version ${version})"
        return 0
    fi
    info "No hay Python >= 3.10. Lo instalo con ${pm}."
    # shellcheck disable=SC2046  # queremos word-splitting aqui
    pkg_install "$pm" $(python_pkg_name "$pm")
    if ! PYTHON_BIN="$(find_python)"; then
        err "La instalacion termino pero sigo sin encontrar Python >= 3.10."
        err "Revisa la salida del gestor de paquetes y vuelve a intentarlo."
        exit 1
    fi
    success "Python instalado: ${PYTHON_BIN}"
}

# --- Deteccion de git ------------------------------------------------------
ensure_git() {
    local pm="$1"
    if command -v git >/dev/null 2>&1; then
        success "git ya esta instalado."
        return 0
    fi
    info "git no esta instalado. Lo instalo con ${pm}."
    pkg_install "$pm" git
    if ! command -v git >/dev/null 2>&1; then
        err "git no quedo disponible despues de instalar."
        exit 1
    fi
    success "git instalado."
}

# --- PATH ------------------------------------------------------------------
shell_rc() {
    case "${SHELL:-}" in
        */zsh)  echo "${HOME}/.zshrc" ;;
        */bash) echo "${HOME}/.bashrc" ;;
        */fish) echo "${HOME}/.config/fish/config.fish" ;;
        *)      echo "${HOME}/.profile" ;;
    esac
}

path_export_line() {
    case "${SHELL:-}" in
        */fish) echo "fish_add_path ${BLACKCODE_BIN}" ;;
        *)      echo "export PATH=\"${BLACKCODE_BIN}:\$PATH\"" ;;
    esac
}

# --- Pasos -----------------------------------------------------------------
step_create_venv() {
    info "Preparando entorno aislado en ${VENV_DIR}"
    mkdir -p "$BLACKCODE_HOME"
    if [ -d "$VENV_DIR" ] && [ -x "${VENV_DIR}/bin/python" ]; then
        success "Entorno ya existe, lo reutilizo."
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "Entorno virtual creado."
    fi
    "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
}

step_install_blackcode() {
    info "Instalando blackcode (core) desde ${REPO_URL}@${BLACKCODE_REF}"
    info "Los extras pesados (torch, transformers) se bajan luego, al usarlos."
    "${VENV_DIR}/bin/python" -m pip install --quiet \
        "git+${REPO_URL}@${BLACKCODE_REF}"
    success "blackcode instalado en ${VENV_DIR}"
}

step_install_shim() {
    info "Colocando ejecutable en ${BLACKCODE_BIN}/blackcode"
    mkdir -p "$BLACKCODE_BIN"
    # Usamos un shim simple en lugar de un symlink: asi no se rompe si el
    # venv se mueve y deja al usuario un archivo legible.
    cat > "${BLACKCODE_BIN}/blackcode" <<EOF
#!/usr/bin/env bash
# Generado por install.sh de Blackcode.
exec "${VENV_DIR}/bin/blackcode" "\$@"
EOF
    chmod +x "${BLACKCODE_BIN}/blackcode"
    success "Comando 'blackcode' disponible en ${BLACKCODE_BIN}/blackcode"
}

step_check_path() {
    case ":${PATH}:" in
        *":${BLACKCODE_BIN}:"*)
            success "${BLACKCODE_BIN} ya esta en PATH."
            return 0
            ;;
    esac
    warn "${BLACKCODE_BIN} no esta en tu PATH."
    printf "\n  Anade esta linea a %s:\n\n      %s\n\n" \
        "$(shell_rc)" "$(path_export_line)"
    printf "  Y recarga la sesion (o abre una terminal nueva).\n\n"
}

step_done() {
    printf "\n%s%sInstalacion lista.%s\n\n" "$GREEN" "$BOLD" "$RESET"
    printf "Siguientes pasos:\n\n"
    printf "  %sblackcode doctor%s              %s# verifica hardware (CPU/GPU/MPS)%s\n" \
        "$BOLD" "$RESET" "$DIM" "$RESET"
    printf "  %sblackcode init mi-proyecto%s    %s# crea un proyecto nuevo%s\n" \
        "$BOLD" "$RESET" "$DIM" "$RESET"
    printf "  %sblackcode --help%s              %s# todos los comandos%s\n\n" \
        "$BOLD" "$RESET" "$DIM" "$RESET"
    printf "Los extras opcionales (LLM, sklearn, embeddings, conectores) se\n"
    printf "instalan solos la primera vez que los uses.\n\n"
}

# --- Main ------------------------------------------------------------------
main() {
    banner

    local os pm
    os="$(detect_os)"
    info "Sistema: ${os} ($(uname -m))"

    # 1. Gestor de paquetes (en macOS: Homebrew; en Linux: el que detectemos).
    if [ "$os" = "macos" ]; then
        ensure_homebrew
        pm="brew"
    else
        pm="$(detect_pkg_manager)"
        if [ "$pm" = "unknown" ]; then
            err "No reconozco el gestor de paquetes de tu distribucion."
            err "Instala manualmente: python3 (>= 3.10) y git, y vuelve a correr."
            exit 1
        fi
        info "Gestor de paquetes detectado: ${pm}"
    fi

    # 2. Dependencias del sistema: Python y git.
    ensure_python "$pm"
    ensure_git "$pm"

    # 3. Blackcode propiamente.
    step_create_venv
    step_install_blackcode
    step_install_shim
    step_check_path
    step_done
}

main "$@"
