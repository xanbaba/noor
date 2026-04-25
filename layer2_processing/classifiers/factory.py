"""Classifier factory — string → :class:`AbstractClassifier` instance."""

from __future__ import annotations

from layer2_processing.classifiers.base import AbstractClassifier
from layer2_processing.classifiers.fbcca import FBCCAClassifier
from layer2_processing.config import ProcessingConfig

# Only FBCCA is registered for the MVP.  TRCA / xDAWN go here later.
_REGISTRY: dict[str, type[AbstractClassifier]] = {
    "fbcca": FBCCAClassifier,
}


def create_classifier(
    cfg: ProcessingConfig, name: str | None = None
) -> AbstractClassifier:
    """Instantiate the classifier requested by ``name`` (or ``cfg.classifier``)."""
    key = (name or cfg.classifier).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown classifier '{key}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](cfg)


def registered_classifiers() -> list[str]:
    return sorted(_REGISTRY)
