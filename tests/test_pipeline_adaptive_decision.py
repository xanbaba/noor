"""Tests for confidence-gated adaptive decision extension in Layer 2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from layer2_processing.classifiers.base import AbstractClassifier, ClassifierResult
from layer2_processing.config import load_config
from layer2_processing.pipeline import Pipeline


class _FakeInlet:
    channel_count = 8


class _SequenceClassifier(AbstractClassifier):
    """Returns a predefined sequence of classifier outputs."""

    def __init__(self, outputs: list[ClassifierResult]) -> None:
        self._outputs = list(outputs)
        self._i = 0

    def predict(self, epoch: np.ndarray) -> ClassifierResult:
        if self._i >= len(self._outputs):
            return self._outputs[-1]
        out = self._outputs[self._i]
        self._i += 1
        return out


def _minimal_cfg(overrides: dict[str, Any]):
    base = Path("configs/layer2_minimal.yaml")
    if not base.exists():
        pytest.skip("configs/layer2_minimal.yaml not found - run from project root.")
    default_overrides = {
        "stimulus_frequencies_hz": [6.0, 20.0],
        "snr_gate_enabled": False,
        "artefact_policy": "ignore",
        "prediction_smoothing_window": 0,
    }
    default_overrides.update(overrides)
    return load_config(base, overrides=default_overrides)


def test_adaptive_decision_defers_then_emits_select():
    cfg = _minimal_cfg(
        {
            "decision_confidence_min": 0.80,
            "decision_max_extra_epochs": 1,
            "decision_emit_no_decision": False,
        }
    )

    classifier = _SequenceClassifier(
        [
            ClassifierResult(
                frequency_hz=6.0,
                confidence=0.55,
                raw_scores=np.array([0.55, 0.45], dtype=np.float32),
            ),
            ClassifierResult(
                frequency_hz=6.0,
                confidence=0.95,
                raw_scores=np.array([0.95, 0.05], dtype=np.float32),
            ),
        ]
    )

    emitted: list[dict[str, Any]] = []

    def _on_emit(p: dict[str, Any]) -> None:
        emitted.append(p)

    pipeline = Pipeline(
        cfg=cfg,
        inlet=_FakeInlet(),  # type: ignore[arg-type]
        classifier=classifier,
        ws_emitter=None,
        osc_emitter=None,
        on_emit=_on_emit,
    )

    epoch = np.random.default_rng(0).standard_normal((8, 1000)).astype(np.float32)
    pipeline._handle_epoch(epoch)
    assert emitted == []

    pipeline._handle_epoch(epoch)
    assert len(emitted) == 1
    assert emitted[0]["command"] == "SELECT"
    assert emitted[0]["frequency"] == 6.0


def test_adaptive_decision_emits_no_decision_after_max_extra_epochs():
    cfg = _minimal_cfg(
        {
            "decision_confidence_min": 0.90,
            "decision_max_extra_epochs": 1,
            "decision_emit_no_decision": True,
            "decision_no_decision_command": "NO_DECISION",
        }
    )

    classifier = _SequenceClassifier(
        [
            ClassifierResult(
                frequency_hz=6.0,
                confidence=0.55,
                raw_scores=np.array([0.55, 0.45], dtype=np.float32),
            ),
            ClassifierResult(
                frequency_hz=6.0,
                confidence=0.56,
                raw_scores=np.array([0.56, 0.44], dtype=np.float32),
            ),
        ]
    )

    emitted: list[dict[str, Any]] = []

    def _on_emit(p: dict[str, Any]) -> None:
        emitted.append(p)

    pipeline = Pipeline(
        cfg=cfg,
        inlet=_FakeInlet(),  # type: ignore[arg-type]
        classifier=classifier,
        ws_emitter=None,
        osc_emitter=None,
        on_emit=_on_emit,
    )

    epoch = np.random.default_rng(1).standard_normal((8, 1000)).astype(np.float32)
    pipeline._handle_epoch(epoch)
    assert emitted == []

    pipeline._handle_epoch(epoch)
    assert len(emitted) == 1
    assert emitted[0]["command"] == "NO_DECISION"
