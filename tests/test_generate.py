"""Tests de la generación de datos de instrucción (sin dependencias pesadas)."""

from blackcode.generate import (
    _estimate_model_size,
    _should_warn_slow,
    chunk_text,
    parse_qa,
    resolve_device,
)


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("texto corto", max_chars=2000) == ["texto corto"]


def test_chunk_text_respects_max_chars():
    text = "\n\n".join("parrafo" + str(i) * 50 for i in range(10))
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_hard_splits_huge_paragraph():
    chunks = chunk_text("x" * 5000, max_chars=1000)
    assert len(chunks) == 5
    assert all(len(c) <= 1000 for c in chunks)


def test_parse_qa_extracts_pairs():
    pairs = parse_qa("P: ¿Qué es?\nR: Un motor.\nP: ¿Dónde corre?\nR: En local.")
    assert pairs == [
        {"instruction": "¿Qué es?", "output": "Un motor."},
        {"instruction": "¿Dónde corre?", "output": "En local."},
    ]


def test_parse_qa_ignores_incomplete_pairs():
    assert parse_qa("P: pregunta sin respuesta\ntexto suelto") == []


def test_parse_qa_accepts_long_form_prefixes():
    pairs = parse_qa("Pregunta: ¿Qué es?\nRespuesta: Un motor.")
    assert pairs == [{"instruction": "¿Qué es?", "output": "Un motor."}]


def test_parse_qa_accepts_q_a_in_english():
    pairs = parse_qa("Q: what is it?\nA: an engine.")
    assert pairs == [{"instruction": "what is it?", "output": "an engine."}]


def test_parse_qa_accepts_numbered_lists():
    text = "1. P: Hola\n   R: Hola.\n2. P: Adiós\nR: Chau."
    assert len(parse_qa(text)) == 2


def test_parse_qa_is_case_insensitive():
    pairs = parse_qa("pregunta: hi\nrespuesta: hello")
    assert pairs == [{"instruction": "hi", "output": "hello"}]


def test_resolve_device_passthrough_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in ("cpu", "cuda", "mps")


def test_estimate_model_size_from_catalog():
    assert _estimate_model_size("Qwen/Qwen2.5-7B-Instruct") == 7.0


def test_estimate_model_size_from_regex():
    assert _estimate_model_size("alguien/Modelo-13B-loquesea") == 13.0


def test_estimate_model_size_unknown_returns_none():
    assert _estimate_model_size("nada/sin-tamano-detectable") is None


def test_should_warn_slow_cpu_with_large_model():
    assert _should_warn_slow("Qwen/Qwen2.5-7B-Instruct", "cpu") is True


def test_should_not_warn_cpu_with_small_model():
    assert _should_warn_slow("Qwen/Qwen2.5-0.5B-Instruct", "cpu") is False


def test_should_not_warn_on_gpu():
    assert _should_warn_slow("Qwen/Qwen2.5-7B-Instruct", "cuda") is False
