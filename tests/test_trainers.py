"""Tests de los trainers que no requieren dependencias pesadas.

`blackcode.trainers.llm` importa torch de forma perezosa, así que el heurístico
de batch (`_auto_batch_config`) es importable y testeable sin instalarlo.
"""

from blackcode.hardware import Strategy
from blackcode.trainers.llm import _auto_batch_config, _default_learning_rate


def test_auto_batch_respects_declared_values():
    assert _auto_batch_config(Strategy.LORA, 24.0, 8, 2) == (8, 2)


def test_auto_batch_qlora_uses_single_sample_batch():
    batch, accum = _auto_batch_config(Strategy.QLORA, 12.0, 0, None)
    assert batch == 1
    assert batch * accum == 16


def test_auto_batch_scales_with_vram():
    assert _auto_batch_config(Strategy.LORA, 48.0, 0, None)[0] == 8
    assert _auto_batch_config(Strategy.LORA, 24.0, 0, None)[0] == 4
    assert _auto_batch_config(Strategy.LORA, 12.0, 0, None)[0] == 2
    assert _auto_batch_config(Strategy.LORA, 8.0, 0, None)[0] == 1


def test_auto_batch_targets_stable_effective_batch():
    for vram in (8.0, 12.0, 24.0, 48.0, 80.0):
        batch, accum = _auto_batch_config(Strategy.LORA, vram, 0, None)
        assert batch * accum == 16


def test_auto_batch_declared_accumulation_overrides_auto():
    _, accum = _auto_batch_config(Strategy.LORA, 48.0, 0, 7)
    assert accum == 7


def test_default_learning_rate_is_conservative_for_full():
    # Full fine-tune (todos los pesos): LR baja para no desestabilizar.
    assert _default_learning_rate(Strategy.FULL) == 2e-5
    assert _default_learning_rate(Strategy.CPU) == 2e-5
    # LoRA/QLoRA: solo afinan adaptadores → toleran LR más alta.
    assert _default_learning_rate(Strategy.LORA) == 2e-4
    assert _default_learning_rate(Strategy.QLORA) == 2e-4
