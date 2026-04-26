"""Append raw EEG chunks (µV, float32) to a CSV file — same data as LSL."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np


class RawEegFileLogger:
    """One row per sample: sample_index, monotonic_s, <ch0>..<chN-1>."""

    def __init__(
        self,
        path: Path,
        channel_labels: list[str],
        sample_rate_hz: int,
        log_format: str = "csv",
    ) -> None:
        fmt = (log_format or "csv").lower()
        if fmt != "csv":
            raise ValueError(f"Unsupported raw_eeg_log_format: {log_format!r}")
        self._path = Path(path).expanduser()
        self._labels = list(channel_labels)
        self._fs = float(sample_rate_hz)
        self._fh: object | None = None  # TextIOWrapper
        self._writer: csv.writer | None = None
        self._sample_index = 0
        self._closed = False

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        header = ["sample_index", "monotonic_s"] + self._labels
        self._writer.writerow(header)
        self._fh.flush()

    def append_chunk(self, chunk: np.ndarray) -> None:
        if self._closed or self._fh is None or self._writer is None:
            return
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"chunk must be 2-D; got shape {x.shape}")
        n_ch, n = x.shape
        if n_ch != len(self._labels):
            raise ValueError(
                f"chunk has {n_ch} channels but {len(self._labels)} labels configured"
            )
        if n == 0:
            return
        t0 = time.monotonic()
        inv_fs = 1.0 / self._fs
        for j in range(n):
            mono = t0 + j * inv_fs
            row = [self._sample_index, f"{mono:.6f}"]
            row.extend(f"{float(x[i, j]):.6f}" for i in range(n_ch))
            self._writer.writerow(row)
            self._sample_index += 1
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None
                self._writer = None
