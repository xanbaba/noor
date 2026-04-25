"""Abstract base class for SSVEP classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassifierResult:
    """Output of :meth:`AbstractClassifier.predict`."""

    frequency_hz: float
    confidence: float           # in [0, 1] — softmax-style normalisation
    raw_scores: np.ndarray      # (n_freqs,) per-class scores, debug only


class AbstractClassifier(ABC):
    """Contract every SSVEP classifier in the pipeline must satisfy.

    Implementations are stateful in their constructor (filters, references,
    weights are pre-computed once) but :meth:`predict` is a pure function of
    the input epoch.
    """

    @abstractmethod
    def predict(self, epoch: np.ndarray) -> ClassifierResult:
        """Classify a single epoch of shape ``(channels, n_samples)``."""
