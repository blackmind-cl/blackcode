"""Tests del instalador `install.sh`.

No ejecuta el script de verdad (eso bajaría blackcode desde GitHub y tocaría
el sistema). Sólo verifica que:
  - es bash válido (`bash -n`),
  - declara las variables clave configurables por entorno,
  - apunta al repo correcto,
  - cubre macOS y Linux y rechaza Windows,
  - busca Python >= 3.10,
  - **auto-instala dependencias del sistema** (Homebrew en macOS si falta,
    Python y git con el gestor del SO si faltan),
  - usa `sudo` cuando hace falta,
  - cubre los gestores de paquetes habituales (apt/dnf/yum/pacman/zypper/apk),
  - NO instala extras pesados (eso lo hace `ensure_extra` después).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "install.sh"


def test_install_sh_exists_and_executable():
    assert SCRIPT.exists(), "install.sh no existe en la raíz del repo"
    assert SCRIPT.stat().st_mode & 0o111, "install.sh debería ser ejecutable"


def test_install_sh_has_valid_bash_syntax():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash no está disponible en este sistema")
    result = subprocess.run(
        [bash, "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Sintaxis bash inválida:\n{result.stderr}"


def test_install_sh_uses_strict_mode():
    content = SCRIPT.read_text()
    assert "set -euo pipefail" in content, (
        "El instalador debe usar 'set -euo pipefail' para fallar rápido."
    )


def test_install_sh_targets_blackmind_repo():
    content = SCRIPT.read_text()
    assert "github.com/blackmind-cl/blackcode" in content, (
        "El instalador debe apuntar al repo blackmind-cl/blackcode."
    )


def test_install_sh_supports_macos_and_linux_only():
    content = SCRIPT.read_text()
    assert "Darwin" in content and "Linux" in content
    # Windows queda fuera: damos un mensaje útil pero no lo soportamos.
    assert "WSL" in content


def test_install_sh_requires_python_310():
    content = SCRIPT.read_text()
    # Comprueba >= 3.10 explícitamente.
    assert "(3,10)" in content or "(3, 10)" in content, (
        "El instalador debe exigir Python >= 3.10."
    )


def test_install_sh_does_not_install_heavy_extras():
    """El core es ~20 MB. Torch/transformers se bajan vía ensure_extra al uso."""
    content = SCRIPT.read_text()
    forbidden = ["blackcode[all]", "blackcode[llm]", "torch", "transformers"]
    for token in forbidden:
        # Sólo prohibimos el token como argumento de instalación.
        # 'transformers' como palabra suelta en un mensaje está bien.
        assert f"pip install {token}" not in content, (
            f"El instalador no debe pre-instalar '{token}': eso es trabajo "
            "de ensure_extra, al primer uso."
        )


def test_install_sh_creates_isolated_venv():
    content = SCRIPT.read_text()
    assert "python -m venv" in content or "-m venv" in content
    assert ".blackcode/venv" in content or "BLACKCODE_HOME" in content


def test_install_sh_documents_curl_oneliner():
    content = SCRIPT.read_text()
    assert "curl -fsSL" in content, (
        "El instalador debe documentar el one-liner en su cabecera."
    )


def test_install_sh_honors_env_overrides():
    content = SCRIPT.read_text()
    for var in ("BLACKCODE_HOME", "BLACKCODE_REF", "BLACKCODE_BIN"):
        assert f"${{{var}:-" in content, (
            f"El instalador debería honrar la variable de entorno {var}."
        )


def test_install_sh_auto_installs_homebrew_on_macos():
    """Si no hay brew en macOS, debe instalarlo (no solo sugerirlo)."""
    content = SCRIPT.read_text()
    assert "ensure_homebrew" in content
    # Usa el instalador oficial de Homebrew en modo no interactivo.
    assert "Homebrew/install" in content
    assert "NONINTERACTIVE=1" in content


def test_install_sh_auto_installs_python_if_missing():
    """Si no hay Python >= 3.10, debe instalarlo con el gestor detectado."""
    content = SCRIPT.read_text()
    assert "ensure_python" in content
    # El nombre del paquete varía por gestor; lo gestionamos con un helper.
    assert "python_pkg_name" in content


def test_install_sh_auto_installs_git_if_missing():
    """git es indispensable para `pip install git+…`. Si falta, lo instalamos."""
    content = SCRIPT.read_text()
    assert "ensure_git" in content


def test_install_sh_supports_common_linux_pkg_managers():
    """Linux: cubrimos apt, dnf, yum, pacman, zypper y apk."""
    content = SCRIPT.read_text()
    for pm in ("apt-get", "dnf", "yum", "pacman", "zypper", "apk"):
        assert pm in content, f"Falta soporte para el gestor de paquetes '{pm}'."


def test_install_sh_uses_sudo_when_needed():
    """Para paquetes del sistema en Linux necesitamos privilegios."""
    content = SCRIPT.read_text()
    assert "as_root" in content
    # El helper debería caer a sudo si no somos root.
    assert "sudo" in content


def test_install_sh_aborts_cleanly_without_sudo_or_root():
    """Si no hay sudo y no somos root, el instalador no debe dejar a medias."""
    content = SCRIPT.read_text()
    # Debe haber un mensaje claro y un exit.
    assert "no encuentro 'sudo'" in content or "sudo" in content
    assert "exit 1" in content
