"""Build labelled epoch arrays from ``raw_eeg.csv`` + ``events.jsonl``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from layer2_processing.config import load_config as load_layer2_config
from layer2_processing.preprocessing import Preprocessor


def load_raw_eeg_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``eeg`` (n_channels, n_samples), ``sample_index`` (n_samples,), channel names."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    if len(header) < 3:
        raise ValueError(f"Unexpected CSV header in {path}")
    ch_names = header[2:]
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    sample_index = data[:, 0].astype(np.int64)
    eeg = np.ascontiguousarray(data[:, 2:].T, dtype=np.float32)
    if sample_index.size != eeg.shape[1]:
        raise ValueError("sample_index length mismatch vs EEG columns")
    diffs = np.diff(sample_index)
    if not np.all(diffs == 1):
        raise ValueError(
            "sample_index must be contiguous with step 1; "
            f"first bad step at position {int(np.argmax(diffs != 1))}"
        )
    return eeg, sample_index, ch_names


def load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def label_to_y(label: str) -> int:
    if label in ("6hz", "12hz"):
        return 0
    if label in ("15hz", "20hz"):
        return 1
    raise ValueError(
        f"Unknown label {label!r}; expected '6hz' or '15hz' (legacy: 12hz, 20hz)"
    )


def pair_trials(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair ``stim_on`` / ``stim_off`` by ``trial_id``."""
    on_by_tid: dict[int, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for ev in events:
        tid = int(ev["trial_id"])
        kind = ev["event"]
        if kind == "stim_on":
            on_by_tid[tid] = ev
        elif kind == "stim_off":
            if tid not in on_by_tid:
                raise ValueError(f"stim_off for trial_id={tid} without stim_on")
            on = on_by_tid.pop(tid)
            pairs.append(
                {
                    "trial_id": tid,
                    "label": str(on["label"]),
                    "sample_on": int(on["sample_index"]),
                    "sample_off": int(ev["sample_index"]),
                }
            )
        else:
            raise ValueError(f"Unknown event type {kind!r}")
    if on_by_tid:
        raise ValueError(f"Unmatched stim_on trials: {sorted(on_by_tid)}")
    pairs.sort(key=lambda p: p["trial_id"])
    return pairs


def build_epochs(
    *,
    eeg: np.ndarray,
    n_samples_total: int,
    trials: list[dict[str, Any]],
    sample_rate_hz: int,
    onset_delay_s: float,
    window_s: float,
    preprocessor: Preprocessor | None,
    session_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, C, T), y (n,), trial_id (n,), session (n,) as object array."""
    win = int(round(window_s * sample_rate_hz))
    delay = int(round(onset_delay_s * sample_rate_hz))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    tids: list[int] = []
    for tr in trials:
        s0 = tr["sample_on"] + delay
        s1 = s0 + win
        if s0 < 0 or s1 > n_samples_total:
            raise ValueError(
                f"trial {tr['trial_id']} window [{s0}, {s1}) out of bounds "
                f"(n_samples={n_samples_total})"
            )
        epoch = eeg[:, s0:s1].astype(np.float32, copy=False)
        if preprocessor is not None:
            epoch = preprocessor.filter(epoch)
        xs.append(epoch)
        ys.append(label_to_y(tr["label"]))
        tids.append(int(tr["trial_id"]))
    if not xs:
        raise ValueError("No epochs extracted")
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    trial_id = np.asarray(tids, dtype=np.int64)
    session = np.array([session_id] * len(ys), dtype=object)
    return X, y, trial_id, session


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build epochs.npz from raw CSV + events.")
    p.add_argument("--eeg-csv", type=Path, required=True)
    p.add_argument("--events-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Path to epochs.npz")
    p.add_argument("--session-id", type=str, default="session")
    p.add_argument("--onset-delay-s", type=float, default=0.75)
    p.add_argument("--window-s", type=float, default=1.5)
    p.add_argument(
        "--layer2-config",
        type=Path,
        default=Path("configs/layer2_default.yaml"),
        help="YAML for notch/bandpass/CAR (use_car forced True).",
    )
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip Layer 2-style filtering (raw µV windows only).",
    )
    args = p.parse_args(argv)

    eeg, sample_index, ch_names = load_raw_eeg_csv(args.eeg_csv)
    events = load_events_jsonl(args.events_jsonl)
    trials = pair_trials(events)

    l2 = load_layer2_config(args.layer2_config)
    fs = l2.sample_rate_hz
    if args.no_preprocess:
        pre = None
    else:
        pre = Preprocessor(replace(l2, use_car=True))

    X, y, trial_id, session = build_epochs(
        eeg=eeg,
        n_samples_total=eeg.shape[1],
        trials=trials,
        sample_rate_hz=fs,
        onset_delay_s=float(args.onset_delay_s),
        window_s=float(args.window_s),
        preprocessor=pre,
        session_id=str(args.session_id),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        trial_id=trial_id,
        session=session,
        channel_names=np.array(ch_names, dtype=object),
        sample_rate_hz=np.int32(fs),
        onset_delay_s=np.float32(args.onset_delay_s),
        window_s=np.float32(args.window_s),
    )
    print(f"Wrote {args.output} | X shape={X.shape} | trials={len(trials)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
