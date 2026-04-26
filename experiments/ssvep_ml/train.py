"""Train 6 Hz vs 15 Hz classifier (sklearn band-power baseline or tiny CNN)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.ssvep_ml.features import bandpower_feature_matrix
from experiments.ssvep_ml.inference_wrapper import export_notes_text


def _load_merged(npz_paths: list[Path]) -> dict[str, Any]:
    Xs, ys, sessions = [], [], []
    fs_ref: float | None = None
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        s = np.asarray(d["session"], dtype=object).reshape(-1)
        if s.size != len(d["y"]):
            s = np.array([str(path)] * len(d["y"]), dtype=object)
        sessions.append(s)
        fs = float(d["sample_rate_hz"])
        fs_ref = fs if fs_ref is None else fs_ref
        if float(d["sample_rate_hz"]) != fs_ref:
            raise ValueError(f"sample_rate_hz mismatch in {path}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    session = np.concatenate(sessions, axis=0)
    return {"X": X, "y": y, "session": session, "fs": float(fs_ref)}


def _split_by_session(
    data: dict[str, Any],
    val_session: str | None,
    test_session: str | None,
) -> tuple[np.ndarray, ...]:
    X, y, session = data["X"], data["y"], data["session"]
    if val_session is None and test_session is None:
        return tuple()  # signal caller to use random split
    if val_session is None or test_session is None:
        raise ValueError("Session split requires both --val-session and --test-session")

    def mask(s: str) -> np.ndarray:
        return np.array([str(t) == s for t in session], dtype=bool)
    m_test = mask(test_session)
    m_val = mask(val_session)
    m_train = ~(m_test | m_val)
    if not np.any(m_train):
        raise ValueError("No training samples after session split")
    return X[m_train], y[m_train], X[m_val], y[m_val], X[m_test], y[m_test]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _train_sklearn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    fs: float,
) -> Pipeline:
    bands = [(4.0, 8.5), (10.0, 14.0), (14.0, 17.0), (28.0, 32.0)]
    Xf_tr = bandpower_feature_matrix(X_tr, fs, bands)
    Xf_va = bandpower_feature_matrix(X_va, fs, bands)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(Xf_tr, y_tr)
    pred_va = pipe.predict(Xf_va)
    print("val:", _metrics(y_va, pred_va))
    return pipe


def _train_cnn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    out_dir: Path,
    epochs_max: int,
) -> None:
    import torch
    from torch import nn

    from experiments.ssvep_ml.eeg_cnn import TinyEegCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, n_ch, n_t = X_tr.shape
    model = TinyEegCNN(n_ch, n_t, n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    def to_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=shuffle)

    dl_tr = to_loader(X_tr, y_tr, True)
    dl_va = to_loader(X_va, y_va, False)

    best_state = None
    best_val = 0.0
    bad = 0
    for ep in range(int(epochs_max)):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                pred = model(xb).argmax(dim=1).cpu().numpy()
                correct += int((pred == yb.numpy()).sum())
                total += len(yb)
        acc = correct / max(total, 1)
        if acc > best_val + 1e-4:
            best_val = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 25:
            break
        if ep % 10 == 0:
            print(f"epoch {ep:4d} val_acc={acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "cnn_weights.pt")
    (out_dir / "cnn_config.json").write_text(
        json.dumps({"n_channels": n_ch, "n_time": n_t, "n_classes": 2}),
        encoding="utf-8",
    )
    print("CNN best val acc (proxy):", best_val)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train SSVEP 6 vs 15 Hz classifier.")
    p.add_argument("--npz", type=Path, nargs="+", required=True, help="epochs.npz files")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-session", type=str, default=None)
    p.add_argument("--test-session", type=str, default=None)
    p.add_argument(
        "--backend",
        choices=["sklearn", "cnn"],
        default="sklearn",
    )
    p.add_argument("--cnn-epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_merged(list(args.npz))
    X, y, session = data["X"], data["y"], data["session"]
    fs = data["fs"]

    split = _split_by_session(data, args.val_session, args.test_session)
    if len(split) == 0:
        X_tr, x_tmp, y_tr, y_tmp = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=int(args.seed),
            stratify=y,
        )
        X_va, X_te, y_va, y_te = train_test_split(
            x_tmp,
            y_tmp,
            test_size=0.5,
            random_state=int(args.seed),
            stratify=y_tmp,
        )
    else:
        X_tr, y_tr, X_va, y_va, X_te, y_te = split

    manifest: dict[str, Any] = {
        "model": args.backend,
        "classes_hz": ["6", "15"],
        "sample_rate_hz": fs,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
    }

    if args.backend == "sklearn":
        bands = [(4.0, 8.5), (10.0, 14.0), (14.0, 17.0), (28.0, 32.0)]
        pipe = _train_sklearn(X_tr, y_tr, X_va, y_va, fs)
        joblib.dump(pipe, out_dir / "sklearn_model.joblib")
        Xf_te = bandpower_feature_matrix(X_te, fs, bands)
        pred_te = pipe.predict(Xf_te)
        manifest["test_metrics"] = _metrics(y_te, pred_te)
        manifest["confusion_matrix"] = confusion_matrix(y_te, pred_te).tolist()
        print("test:", manifest["test_metrics"])
    else:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise SystemExit("CNN backend requires: pip install 'neuron[experiments]'") from exc
        _train_cnn(X_tr, y_tr, X_va, y_va, out_dir, args.cnn_epochs)
        manifest["cnn_bundle"] = "cnn_weights.pt + cnn_config.json"
        # quick test eval
        import torch
        from torch import nn

        from experiments.ssvep_ml.eeg_cnn import TinyEegCNN

        cfg = json.loads((out_dir / "cnn_config.json").read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyEegCNN(cfg["n_channels"], cfg["n_time"], cfg["n_classes"]).to(device)
        model.load_state_dict(torch.load(out_dir / "cnn_weights.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_te).float().to(device))
            pred_te = logits.argmax(dim=1).cpu().numpy()
        manifest["test_metrics"] = _metrics(y_te, pred_te)
        manifest["confusion_matrix"] = confusion_matrix(y_te, pred_te).tolist()
        print("test:", manifest["test_metrics"])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "EXPORT_NOTES.txt").write_text(export_notes_text() + "\n", encoding="utf-8")
    print("Wrote", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
