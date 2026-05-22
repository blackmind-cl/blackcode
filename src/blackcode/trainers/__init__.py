"""Trainers incluidos en el núcleo.

Importar este módulo registra los trainers de fábrica en el registro global.
Cada trainer importa sus dependencias pesadas de forma perezosa, por lo que
basta tener instalado el extra correspondiente para usarlo.
"""

from blackcode.trainers import (  # noqa: F401  (efecto: registro)
    classifier,
    dpo,
    embeddings,
    llm,
    sklearn,
)
from blackcode.trainers.base import BaseTrainer, TrainResult

__all__ = ["BaseTrainer", "TrainResult"]
