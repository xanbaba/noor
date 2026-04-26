"""Evaluate a saved model bundle on a held-out ``epochs.npz``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from experiments.ssvep_ml.features import bandpower_feature_matrix


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SSVEP classifier bundle.")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--epochs-npz", type=Path, required=True)
    args = p.parse_args(argv)

    manifest = json.loads((args.model_dir / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(args.epochs_npz, allow_pickle=True)
    X, y = data["X"], data["y"]
    fs = float(data["sample_rate_hz"])

    model_type = manifest.get("model", "sklearn")
    if model_type == "sklearn":
        bands = [(11.0, 13.0), (14.0, 16.0), (22.0, 26.0), (28.0, 32.0)]
        pipe = joblib.load(args.model_dir / "sklearn_model.joblib")
        Xf = bandpower_feature_matrix(X, fs, bands)
        pred = pipe.predict(Xf)
    elif model_type == "cnn":
        import torch

        from experiments.ssvep_ml.eeg_cnn import TinyEegCNN

        cfg = json.loads((args.model_dir / "cnn_config.json").read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyEegCNN(cfg["n_channels"], cfg["n_time"], cfg["n_classes"]).to(device)
        model.load_state_dict(
            torch.load(args.model_dir / "cnn_weights.pt", map_location=device)
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X).float().to(device)).argmax(dim=1).cpu().numpy()
    else:
        raise SystemExit(f"Unknown model type {model_type!r}")

    classes = list(manifest.get("classes_hz", [str(i) for i in range(int(y.max()) + 1)]))
    label_ids = list(range(len(classes)))
    cm = confusion_matrix(y, pred, labels=label_ids)

    def _pretty_class(c: str) -> str:
        s = str(c)
        return f"{s} Hz" if s.replace(".", "", 1).isdigit() else s

    pred_headers = [_pretty_class(c) for c in classes]

    print("balanced_accuracy:", balanced_accuracy_score(y, pred))
    print("macro_f1:", f1_score(y, pred, average="macro"))
    print("n_samples:", len(y))
    print()
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"{'':>22}" + "".join(f"{h:>14}" for h in pred_headers))
    for i, row in enumerate(cm):
        print(f"true {_pretty_class(classes[i]):>12}" + "".join(f"{int(v):>14}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
