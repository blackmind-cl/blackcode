"""Detección de hardware y selección automática de estrategia.

Este módulo NO requiere torch. Si torch está instalado se usa para una
detección más precisa de GPU; si no, se recurre a `nvidia-smi` o se asume CPU.
El objetivo es que `import blackcode` funcione en cualquier máquina.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum


class Accelerator(str, Enum):
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


class Strategy(str, Enum):
    """Estrategia de entrenamiento recomendada según recursos."""

    FULL = "full"          # fine-tuning completo (mucha VRAM)
    LORA = "lora"          # adaptadores LoRA (VRAM media)
    QLORA = "qlora"        # LoRA + cuantización 4-bit (VRAM baja)
    CPU = "cpu"            # sin GPU; solo modelos pequeños / ML clásico


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float


@dataclass
class HardwareProfile:
    os: str
    python: str
    cpu_count: int
    ram_gb: float
    accelerator: Accelerator
    gpus: list[GPUInfo] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        return round(sum(g.total_memory_gb for g in self.gpus), 1)

    @property
    def is_multi_gpu(self) -> bool:
        """True si hay más de una GPU: habilita entrenamiento distribuido."""
        return len(self.gpus) > 1

    def recommend_strategy(self, model_billions: float | None = None) -> Strategy:
        """Recomienda una estrategia.

        `model_billions` = tamaño del modelo en miles de millones de params.
        Heurística por GPU (reglas aproximadas para fine-tuning en bf16 con
        Adam): full ~16 GB/B (pesos+gradientes+estados del optimizador),
        LoRA ~4 GB/B (modelo base + adaptadores), QLoRA ~1.2 GB/B (base en
        4-bit + adaptadores). Conservadora a propósito para dejar margen a
        activaciones y batch.
        """
        if self.accelerator == Accelerator.CPU:
            return Strategy.CPU
        if self.accelerator == Accelerator.MPS:
            # Apple Silicon: memoria unificada (sin VRAM medible). LoRA es la
            # opción estable — afina solo los adaptadores, no desestabiliza
            # el modelo base. QLoRA depende de bitsandbytes que no soporta MPS.
            return Strategy.LORA
        vram = self.total_vram_gb
        if model_billions is None:
            # Sin información del modelo: elegir por VRAM disponible.
            if vram >= 48:
                return Strategy.FULL
            if vram >= 16:
                return Strategy.LORA
            return Strategy.QLORA
        if vram >= model_billions * 16:
            return Strategy.FULL
        if vram >= model_billions * 4:
            return Strategy.LORA
        if vram >= model_billions * 1.2:
            return Strategy.QLORA
        return Strategy.CPU


def _ram_gb() -> float:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError):
        pass
    return 0.0


def _detect_with_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=round(props.total_memory / (1024 ** 3), 1),
                )
            )
        return Accelerator.CUDA, gpus
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Accelerator.MPS, []
    return Accelerator.CPU, []


def _detect_with_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        idx, name, mem = (p.strip() for p in line.split(","))
        gpus.append(
            GPUInfo(
                index=int(idx),
                name=name,
                total_memory_gb=round(float(mem) / 1024, 1),
            )
        )
    if gpus:
        return Accelerator.CUDA, gpus
    return None


def detect_hardware() -> HardwareProfile:
    """Inspecciona la máquina y devuelve un perfil de hardware."""
    accelerator, gpus = Accelerator.CPU, []
    for detector in (_detect_with_torch, _detect_with_nvidia_smi):
        result = detector()
        if result is not None:
            accelerator, gpus = result
            break

    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        accelerator=accelerator,
        gpus=gpus,
    )
