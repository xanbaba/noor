"""Smoke test: sklearn train + eval + inference wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.ssvep_ml.inference_wrapper import load_bundle, predict_proba
from experiments.ssvep_ml.train import main as train_main


@pytest.fixture
def tiny_epochs_npz(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n, c, t = 40, 8, 375
    X = rng.standard_normal((n, c, t)).astype(np.float32) * 10.0
    y = np.array([0, 1] * 20, dtype=np.int64)
    path = tmp_path / "epochs.npz"
    np.savez_compressed(
        path,
        X=X,
        y=y,
        trial_id=np.arange(n, dtype=np.int64),
        session=np.array(["s0"] * n, dtype=object),
        channel_names=np.array([f"ch{i}" for i in range(c)], dtype=object),
        sample_rate_hz=np.int32(500),
        onset_delay_s=np.float32(0.1),
        window_s=np.float32(0.75),
    )
    return path


def test_train_eval_inference(tmp_path: Path, tiny_epochs_npz: Path) -> None:
    model_dir = tmp_path / "model"
    rc = train_main(
        [
            "--npz",
            str(tiny_epochs_npz),
            "--output-dir",
            str(model_dir),
            "--backend",
            "sklearn",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "sklearn"
    assert "test_metrics" in manifest

    from experiments.ssvep_ml.eval import main as eval_main

    rc2 = eval_main(["--model-dir", str(model_dir), "--epochs-npz", str(tiny_epochs_npz)])
    assert rc2 == 0

    bundle = load_bundle(model_dir)
    prob = predict_proba(bundle, np.zeros((8, 375), dtype=np.float32))
    assert prob.shape == (2,)
    assert np.allclose(prob.sum(), 1.0, atol=1e-5)
