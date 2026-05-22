"""Tests de humo del núcleo (no requieren torch ni sklearn)."""

import json
from pathlib import Path

import pytest

from blackcode import detect_hardware, list_trainers
from blackcode.config import RunConfig
from blackcode.data.base import Dataset, load_dataset
from blackcode.config import DataConfig
from blackcode.hardware import Accelerator, HardwareProfile, Strategy
from blackcode.registry import get_trainer, register_trainer


def test_hardware_detection_runs():
    hw = detect_hardware()
    assert hw.cpu_count >= 1
    assert hw.accelerator in Accelerator
    assert isinstance(hw.recommend_strategy(), Strategy)


def test_strategy_recommendation_by_vram():
    cpu = HardwareProfile("Linux", "3.11", 8, 16.0, Accelerator.CPU)
    assert cpu.recommend_strategy(7) == Strategy.CPU

    # 80 GB: full FT de un 3B (3*16=48 <= 80), pero un 7B (112) cae a LoRA.
    big = HardwareProfile("Linux", "3.11", 32, 128.0, Accelerator.CUDA)
    big.gpus = [type("G", (), {"total_memory_gb": 80})()]
    assert big.recommend_strategy(3) == Strategy.FULL
    assert big.recommend_strategy(7) == Strategy.LORA

    # 12 GB con un 7B: QLoRA (7*1.2=8.4 <= 12), su caso de uso típico.
    small = HardwareProfile("Linux", "3.11", 16, 32.0, Accelerator.CUDA)
    small.gpus = [type("G", (), {"total_memory_gb": 12})()]
    assert small.recommend_strategy(7) == Strategy.QLORA


def test_builtin_trainers_registered():
    import blackcode.trainers  # noqa: F401

    names = set(list_trainers())
    assert {"llm", "sklearn", "embeddings", "transformer-classifier", "dpo"} <= names
    assert get_trainer("LLM") is not None


def test_register_duplicate_raises():
    with pytest.raises(ValueError):

        @register_trainer("llm")
        class _Dup:  # pragma: no cover
            pass


def test_get_unknown_trainer_raises_with_available_list():
    with pytest.raises(KeyError, match="llm"):
        get_trainer("inexistente")


def test_strategy_recommendation_without_model_size():
    cuda = HardwareProfile("Linux", "3.11", 16, 64.0, Accelerator.CUDA)
    cuda.gpus = [type("G", (), {"total_memory_gb": 24})()]
    assert cuda.total_vram_gb == 24.0
    # 24 GB sin tamaño de modelo: rango LoRA (16 <= 24 < 48).
    assert cuda.recommend_strategy() == Strategy.LORA


def test_strategy_recommendation_for_mps_is_lora():
    mps = HardwareProfile("Darwin", "3.13", 8, 32.0, Accelerator.MPS)
    # Apple Silicon: sin VRAM medible. Antes caía a CPU (full fine-tune,
    # inestable con LR alta). Ahora recomienda LoRA, mucho más estable.
    assert mps.recommend_strategy() == Strategy.LORA
    assert mps.recommend_strategy(7) == Strategy.LORA


def test_is_multi_gpu_detection():
    g = type("G", (), {"total_memory_gb": 24})
    one = HardwareProfile("Linux", "3.11", 16, 64.0, Accelerator.CUDA, [g()])
    two = HardwareProfile("Linux", "3.11", 16, 64.0, Accelerator.CUDA, [g(), g()])
    assert one.is_multi_gpu is False
    assert two.is_multi_gpu is True
    assert HardwareProfile("Linux", "3.11", 8, 16.0, Accelerator.CPU).is_multi_gpu is False


def test_dataset_split_is_deterministic():
    ds = Dataset([{"text": str(i)} for i in range(100)])
    a1, b1 = ds.split(0.2, seed=42)
    a2, b2 = ds.split(0.2, seed=42)
    assert len(a1) == 80 and len(b1) == 20
    assert [r["text"] for r in b1] == [r["text"] for r in b2]


def test_dataset_split_keeps_at_least_one_in_train():
    train, val = Dataset([{"text": "x"}]).split(0.1, seed=42)
    assert len(train) == 1
    assert len(val) == 0


def test_template_rendering():
    ds = Dataset([{"instruction": "hola", "output": "mundo"}])
    out = ds.apply_template("{instruction} -> {output}", "text")
    assert out[0]["text"] == "hola -> mundo"


def test_load_jsonl(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n")
    ds = load_dataset(DataConfig(path=str(f)))
    assert len(ds) == 2 and ds[0]["text"] == "a"


def test_config_requires_data_and_train():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"name": "x"})
