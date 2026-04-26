"""Load a saved bundle and run ``predict_proba`` on one epoch ``(n_channels, n_samples)``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ssvep_ml.features import bandpower_feature_matrix


def load_bundle(model_dir: Path | str) -> dict[str, Any]:
    """Return a dict with ``kind``, ``fs``, and model-specific handles."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    kind = str(manifest.get("model", "sklearn"))
    fs = float(manifest["sample_rate_hz"])
    if kind == "sklearn":
        import joblib

        pipe = joblib.load(model_dir / "sklearn_model.joblib")
        bands = [(11.0, 13.0), (14.0, 16.0), (22.0, 26.0), (28.0, 32.0)]
        return {"kind": "sklearn", "fs": fs, "pipe": pipe, "bands": bands}
    if kind == "cnn":
        import torch

        from experiments.ssvep_ml.eeg_cnn import TinyEegCNN

        cfg = json.loads((model_dir / "cnn_config.json").read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyEegCNN(cfg["n_channels"], cfg["n_time"], cfg["n_classes"]).to(device)
        model.load_state_dict(torch.load(model_dir / "cnn_weights.pt", map_location=device))
        model.eval()
        return {"kind": "cnn", "fs": fs, "model": model, "device": device}
    raise ValueError(f"Unsupported model kind {kind!r}")


def predict_proba(bundle: dict[str, Any], epoch_ct: np.ndarray) -> np.ndarray:
    """Return shape ``(n_classes,)`` probabilities (softmax for CNN; sklearn decision_function sigmoid hack)."""
    x = np.asarray(epoch_ct, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("epoch_ct must be (n_channels, n_samples)")
    if bundle["kind"] == "sklearn":
        pipe = bundle["pipe"]
        bands = bundle["bands"]
        fs = bundle["fs"]
        xf = bandpower_feature_matrix(x[np.newaxis, ...], fs, bands)
        if hasattr(pipe, "predict_proba"):
            return np.asarray(pipe.predict_proba(xf)[0], dtype=np.float64)
        z = pipe.decision_function(xf)[0]
        p1 = 1.0 / (1.0 + np.exp(-float(z)))
        return np.array([1.0 - p1, p1], dtype=np.float64)
    import torch

    model = bundle["model"]
    device = bundle["device"]
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return np.asarray(prob, dtype=np.float64)


def export_notes_text() -> str:
    return (
        "Real-time integration (future): load_bundle(model_dir) once, then call "
        "predict_proba(bundle, epoch) on each Layer-2-sized window. For Layer 2, "
        "implement a new AbstractClassifier that wraps predict_proba or TorchScript "
        "export of TinyEegCNN."
    )
