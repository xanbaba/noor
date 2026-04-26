"""Pipeline behaviour when SNR gate and artefact veto are disabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from layer2_processing.classifiers.factory import create_classifier
from layer2_processing.config import load_config
from layer2_processing.pipeline import Pipeline


class _FakeInlet:
    channel_count = 8


def test_handle_epoch_emits_when_snr_gate_disabled_despite_low_snr():
    """With snr_gate_enabled false, an epoch must not be dropped for low SNR."""
    base = Path("configs/layer2_minimal.yaml")
    if not base.exists():
        pytest.skip("configs/layer2_minimal.yaml not found — run from project root.")
    cfg = load_config(
        base,
        overrides={
            "snr_min_db": 100.0,
            "snr_gate_enabled": False,
            "artefact_policy": "ignore",
        },
    )
    assert cfg.snr_gate_enabled is False

    rng = np.random.default_rng(0)
    epoch = (rng.standard_normal((8, 1000)) * 5.0).astype(np.float32)

    classifier = create_classifier(cfg)
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
    pipeline._handle_epoch(epoch)

    assert len(emitted) == 1
    assert emitted[0]["command"] == "SELECT"
    assert pipeline.stats.commands_emitted == 1
    assert pipeline.stats.epochs_below_snr == 0
