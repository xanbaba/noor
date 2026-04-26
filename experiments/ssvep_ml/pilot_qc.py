"""Quick PSD sanity check on saved epochs (no GUI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal


def dominant_frequency(epoch_1d: np.ndarray, fs: float) -> float:
    nperseg = min(len(epoch_1d), int(fs))
    f, pxx = signal.welch(epoch_1d, fs=fs, nperseg=nperseg)
    k = int(np.argmax(pxx))
    return float(f[k])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print dominant Welch frequency per epoch.")
    p.add_argument("--epochs-npz", type=Path, required=True)
    p.add_argument("--channel", type=int, default=0, help="Channel index for PSD peak.")
    p.add_argument("--max-trials", type=int, default=12)
    args = p.parse_args(argv)

    data = np.load(args.epochs_npz, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    fs = float(data["sample_rate_hz"])

    n = min(int(args.max_trials), X.shape[0])
    print(f"fs={fs} Hz | showing first {n} epochs | ch={args.channel}")
    for i in range(n):
        dom = dominant_frequency(X[i, int(args.channel)], fs)
        print(f"  epoch={i:3d}  y={int(y[i])}  peak_hz={dom:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
