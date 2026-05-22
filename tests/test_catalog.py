"""Tests del catálogo de modelos recomendados."""

from blackcode.catalog import CATALOG, recommend_models
from blackcode.hardware import Accelerator, HardwareProfile


def _cpu() -> HardwareProfile:
    return HardwareProfile("Linux", "3.11", 8, 16.0, Accelerator.CPU)


def _gpu(vram_gb: float) -> HardwareProfile:
    hw = HardwareProfile("Linux", "3.11", 16, 64.0, Accelerator.CUDA)
    hw.gpus = [type("G", (), {"total_memory_gb": vram_gb})()]
    return hw


def test_catalog_is_not_empty():
    assert len(CATALOG) > 0


def test_recommend_models_returns_one_strategy_per_model():
    out = recommend_models(_cpu())
    assert len(out) == len(CATALOG)
    assert all(strategy == "cpu" for _, strategy in out)


def test_recommend_models_filters_by_task():
    llms = recommend_models(_gpu(24.0), task="llm")
    assert llms and all(model.task == "llm" for model, _ in llms)


def test_recommend_strategy_scales_with_vram():
    by_id = {m.id: s for m, s in recommend_models(_gpu(80.0))}
    small = next(m for m in CATALOG if m.task == "llm" and m.params_billions <= 1)
    assert by_id[small.id] == "full"  # 1B en 80 GB: fine-tuning completo
