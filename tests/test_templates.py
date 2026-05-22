"""Tests de las plantillas de proyecto."""

import yaml

from blackcode.config import RunConfig
from blackcode.templates import DEFAULT_TEMPLATE, TEMPLATES


def test_default_template_exists():
    assert DEFAULT_TEMPLATE in TEMPLATES


def test_expected_use_cases_are_covered():
    assert {"chatbot", "clasificador", "embeddings"} <= set(TEMPLATES)


def test_every_template_is_a_valid_config():
    for name, text in TEMPLATES.items():
        config = RunConfig.from_dict(yaml.safe_load(text))
        assert config.name, f"la plantilla '{name}' no define 'name'"
        assert config.train.trainer, f"la plantilla '{name}' no define trainer"
