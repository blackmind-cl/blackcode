"""Tests del asistente de configuración (`blackcode init`).

El asistente recibe funciones de entrada/salida inyectables, y los tests lo
ejecutan con `animate=False` para que sea instantáneo y sin códigos ANSI.
"""

from blackcode.config import RunConfig
from blackcode.hardware import Accelerator, HardwareProfile
from blackcode.wizard import _read_key, run_wizard

_HW = HardwareProfile("Linux", "3.11", 8, 16.0, Accelerator.CPU)


def _feed(answers):
    """Devuelve una función de entrada que va entregando `answers` en orden."""
    items = iter(answers)
    return lambda _prompt: next(items)


def _silent(_message):
    pass


def _run(answers, **kwargs):
    return run_wizard(
        _feed(answers), _silent, hardware=_HW, animate=False, **kwargs
    )


def test_wizard_builds_chatbot_config():
    # El 4º valor "1" es la primera opción del menú de modelo: Qwen 0.5B-Instruct.
    config = _run(["1", "soporte", "./d.jsonl", "1", "2", "./out"])
    assert config["name"] == "soporte"
    assert config["train"]["trainer"] == "llm"
    assert config["train"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["train"]["epochs"] == 2
    assert config["data"]["path"] == "./d.jsonl"


def test_wizard_empty_answers_use_defaults():
    config = _run([""] * 6)
    assert config["name"] == "mi-chatbot"
    assert config["train"]["trainer"] == "llm"
    assert config["data"]["path"] == "./data/train.jsonl"


def test_wizard_classifier_asks_for_label_field():
    config = _run(["2", "tickets", "./d.csv", "1", "3", "./out", "categoria"])
    assert config["train"]["trainer"] == "transformer-classifier"
    assert config["data"]["label_field"] == "categoria"


def test_wizard_choice_rejects_invalid_then_accepts():
    config = _run(["99", "1", "", "", "", "", ""])
    assert config["train"]["trainer"] == "llm"


def test_wizard_uses_project_dir_as_default_name():
    config = _run([""] * 6, default_name="mi-empresa")
    assert config["name"] == "mi-empresa"


def test_wizard_output_is_a_valid_config():
    config = _run(["3"] + [""] * 5)
    run_config = RunConfig.from_dict(config)
    assert run_config.train.trainer == "embeddings"
    assert run_config.data.options["positive_field"] == "positive"


def test_wizard_exit_option_returns_none():
    # La 4ª opción del menú de tipo de proyecto es «salir».
    assert _run(["4"]) is None


def test_wizard_custom_model_falls_back_to_free_text():
    # LLM tiene 7 modelos en el catálogo; la 8ª opción del menú es «custom»
    # y se pregunta el id como texto libre.
    config = _run(["1", "demo", "./d.jsonl", "8", "mi/modelo-x", "3", "./out"])
    assert config["train"]["model"] == "mi/modelo-x"


def test_wizard_chatbot_includes_instruction_template():
    config = _run(["1", "demo", "./d.jsonl", "1", "3", "./out"])
    assert "instruction_template" in config["data"]
    assert "{instruction}" in config["data"]["instruction_template"]
    assert "{output}" in config["data"]["instruction_template"]


def test_wizard_non_chatbot_has_no_instruction_template():
    classifier = _run(["2", "c", "./d.csv", "1", "3", "./out", "label"])
    embeddings = _run(["3"] + [""] * 5)
    assert "instruction_template" not in classifier["data"]
    assert "instruction_template" not in embeddings["data"]


def test_is_recommended_marks_small_models_on_cpu():
    from blackcode.catalog import ModelInfo
    from blackcode.wizard import _is_recommended

    small = ModelInfo("a", 0.5, "llm", "")
    big = ModelInfo("b", 7.0, "llm", "")
    assert _is_recommended(small, _HW) is True
    assert _is_recommended(big, _HW) is False


def test_is_recommended_gpu_uses_strategy():
    from blackcode.catalog import ModelInfo
    from blackcode.hardware import Accelerator, HardwareProfile
    from blackcode.wizard import _is_recommended

    gpu = HardwareProfile("Linux", "3.11", 16, 64.0, Accelerator.CUDA)
    gpu.gpus = [type("G", (), {"total_memory_gb": 24})()]
    assert _is_recommended(ModelInfo("x", 7.0, "llm", ""), gpu) is True


def _reader(text):
    """Lector de caracteres para probar `_read_key` sin una terminal real."""
    box = {"i": 0}

    def read(n):
        chunk = text[box["i"] : box["i"] + n]
        box["i"] += n
        return chunk

    return read


def test_read_key_parses_arrows_enter_and_aliases():
    assert _read_key(_reader("\r")) == "enter"
    assert _read_key(_reader("\n")) == "enter"
    assert _read_key(_reader("\x1b[A")) == "up"
    assert _read_key(_reader("\x1b[B")) == "down"
    assert _read_key(_reader("k")) == "up"
    assert _read_key(_reader("j")) == "down"
    assert _read_key(_reader("x")) == "other"
