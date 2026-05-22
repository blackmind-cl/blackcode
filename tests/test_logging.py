"""Tests del módulo de logging y progreso."""

import logging

from blackcode.log import configure_logging, get_logger, track


def test_get_logger_is_child_of_blackcode():
    assert get_logger("pipeline").name == "blackcode.pipeline"


def test_track_yields_all_items_unchanged():
    assert list(track([1, 2, 3], "test")) == [1, 2, 3]


def test_track_handles_empty_iterable():
    assert list(track([])) == []


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging(verbose=True)
    logger = logging.getLogger("blackcode")
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
