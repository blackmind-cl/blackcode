"""Tests de validación de esquema de configuración."""

from pathlib import Path

import pytest

from blackcode.config import ConfigError, RunConfig

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _minimal(**overrides):
    raw = {
        "data": {"path": "./d.jsonl"},
        "train": {"trainer": "llm", "model": "Qwen/Qwen2.5-0.5B"},
    }
    raw.update(overrides)
    return raw


def test_minimal_config_is_valid():
    cfg = RunConfig.from_dict(_minimal())
    assert cfg.name == "unnamed-run"
    assert cfg.train.trainer == "llm"
    assert cfg.data.format == "auto"


def test_example_configs_validate():
    for path in sorted(_EXAMPLES.glob("*.yaml")):
        import yaml

        RunConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_missing_data_and_train_blocks():
    with pytest.raises(ConfigError, match="data.*train"):
        RunConfig.from_dict({"name": "x"})


def test_unknown_top_level_block():
    with pytest.raises(ConfigError, match="no reconocido"):
        RunConfig.from_dict(_minimal(extra={}))


def test_unknown_key_in_block_lists_valid_keys():
    with pytest.raises(ConfigError, match="text_field"):
        RunConfig.from_dict(_minimal(data={"path": "d", "txt_field": "t"}))


def test_missing_required_key():
    with pytest.raises(ConfigError, match="model"):
        RunConfig.from_dict(_minimal(train={"trainer": "llm"}))


def test_wrong_scalar_type():
    with pytest.raises(ConfigError, match="epochs"):
        RunConfig.from_dict(
            _minimal(train={"trainer": "llm", "model": "m", "epochs": "tres"})
        )


def test_block_must_be_a_mapping():
    with pytest.raises(ConfigError, match="mapa"):
        RunConfig.from_dict({"data": "oops", "train": {"trainer": "llm", "model": "m"}})


def test_invalid_format_choice():
    with pytest.raises(ConfigError, match="format"):
        RunConfig.from_dict(_minimal(data={"path": "d", "format": "xml"}))


def test_invalid_strategy_choice():
    with pytest.raises(ConfigError, match="strategy"):
        RunConfig.from_dict(
            _minimal(train={"trainer": "llm", "model": "m", "strategy": "turbo"})
        )


def test_optional_field_accepts_none_and_value():
    assert RunConfig.from_dict(
        _minimal(data={"path": "d", "label_field": None})
    ).data.label_field is None
    assert RunConfig.from_dict(
        _minimal(data={"path": "d", "label_field": "y"})
    ).data.label_field == "y"


def test_integer_is_accepted_where_float_expected():
    cfg = RunConfig.from_dict(
        _minimal(train={"trainer": "llm", "model": "m", "epochs": 3})
    )
    assert cfg.train.epochs == 3


def test_name_must_be_a_string():
    with pytest.raises(ConfigError, match="name"):
        RunConfig.from_dict(_minimal(name=123))


def test_config_error_is_a_value_error():
    # La CLI captura ValueError; ConfigError debe seguir siéndolo.
    assert issubclass(ConfigError, ValueError)


def test_resume_defaults_false_and_accepts_bool():
    assert RunConfig.from_dict(_minimal()).train.resume is False
    cfg = RunConfig.from_dict(
        _minimal(train={"trainer": "llm", "model": "m", "resume": True})
    )
    assert cfg.train.resume is True


def test_export_block_is_optional_and_validated():
    assert RunConfig.from_dict(_minimal()).export.formats == []
    cfg = RunConfig.from_dict(
        _minimal(export={"formats": ["gguf"], "quantization": "q4_k_m"})
    )
    assert cfg.export.formats == ["gguf"]
    assert cfg.export.quantization == "q4_k_m"


def test_unknown_key_in_export_block():
    with pytest.raises(ConfigError, match="formats"):
        RunConfig.from_dict(_minimal(export={"formatos": []}))


def test_augment_field_defaults_and_validates():
    assert RunConfig.from_dict(_minimal()).data.augment == 0
    cfg = RunConfig.from_dict(_minimal(data={"path": "d", "augment": 3}))
    assert cfg.data.augment == 3
