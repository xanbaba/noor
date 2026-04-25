"""SSVEP classifier implementations.

Only :class:`FBCCAClassifier` is registered for the Layer 2 MVP.  The
``AbstractClassifier`` interface and the YAML ``classifier:`` field exist so
TRCA / xDAWN can be added in later iterations without touching the pipeline.
"""

from layer2_processing.classifiers.base import (
    AbstractClassifier,
    ClassifierResult,
)
from layer2_processing.classifiers.factory import create_classifier
from layer2_processing.classifiers.fbcca import FBCCAClassifier

__all__ = [
    "AbstractClassifier",
    "ClassifierResult",
    "FBCCAClassifier",
    "create_classifier",
]
