"""Build epochs.npz from synthetic CSV + events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ssvep_ml.build_dataset import main as build_main


def _write_synthetic_session(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    fs = 500
    n = 2500
    ch = 8
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((n, ch)).astype(np.float64) * 5.0
    csv_path = d / "raw_eeg.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("sample_index,monotonic_s," + ",".join(f"ch{i}" for i in range(ch)) + "\n")
        for i in range(n):
            row = [str(i), f"{i / fs:.6f}"] + [f"{eeg[i, j]:.6f}" for j in range(ch)]
            fh.write(",".join(row) + "\n")
    events = [
        {"event": "stim_on", "trial_id": 0, "label": "6hz", "frequency_hz": 6.0, "sample_index": 200},
        {"event": "stim_off", "trial_id": 0, "sample_index": 1200},
        {"event": "stim_on", "trial_id": 1, "label": "20hz", "frequency_hz": 20.0, "sample_index": 1300},
        {"event": "stim_off", "trial_id": 1, "sample_index": 2200},
    ]
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def test_build_dataset_npz(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "sess"
    _write_synthetic_session(root)
    out = root / "epochs.npz"
    code = build_main(
        [
            "--eeg-csv",
            str(root / "raw_eeg.csv"),
            "--events-jsonl",
            str(root / "events.jsonl"),
            "--output",
            str(out),
            "--session-id",
            "test_sess",
            "--onset-delay-s",
            "0.1",
            "--window-s",
            "0.5",
            "--layer2-config",
            str(repo / "configs" / "layer2_default.yaml"),
        ]
    )
    assert code == 0
    d = np.load(out, allow_pickle=True)
    assert d["X"].shape[0] == 2
    assert d["y"].tolist() == [0, 1]
    assert d["X"].shape[1] == 8
    assert d["X"].shape[2] == int(0.5 * 500)
