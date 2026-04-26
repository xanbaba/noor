"""Configuration loading and validation for Layer 1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AcquisitionConfig:
    board: str
    serial_port: str
    sample_rate_hz: int
    channel_count: int
    channel_labels: list[str]
    impedance_max_kohm: float
    lsl_stream_name: str
    lsl_stream_type: str
    pull_interval_ms: int
    log_interval_s: int
    # Optional: append same float32 µV chunks as LSL to this file (CSV). None = off.
    raw_eeg_log_path: Optional[str] = None
    raw_eeg_log_format: str = "csv"

    @property
    def active_channel_indices(self) -> list[int]:
        """Return 0-based indices of channels that have real electrode labels."""
        return [i for i, lbl in enumerate(self.channel_labels) if lbl != "--"]

    def validate(self) -> None:
        if self.sample_rate_hz < 250:
            raise ValueError(
                f"sample_rate_hz={self.sample_rate_hz} is below the 250 Hz minimum "
                "required for reliable harmonic detection."
            )
        if len(self.channel_labels) != self.channel_count:
            raise ValueError(
                f"channel_labels length ({len(self.channel_labels)}) must equal "
                f"channel_count ({self.channel_count})."
            )
        if self.impedance_max_kohm <= 0:
            raise ValueError("impedance_max_kohm must be > 0.")
        if self.board not in {"cyton", "synthetic"}:
            raise ValueError(
                f"Unknown board '{self.board}'. Valid values: cyton, synthetic."
            )
        if self.pull_interval_ms < 1:
            raise ValueError("pull_interval_ms must be >= 1.")
        fmt = (self.raw_eeg_log_format or "csv").lower()
        if fmt not in {"csv"}:
            raise ValueError(
                f"raw_eeg_log_format must be 'csv' for now; got {self.raw_eeg_log_format!r}"
            )
        if self.raw_eeg_log_path is not None and not str(self.raw_eeg_log_path).strip():
            raise ValueError("raw_eeg_log_path must be non-empty when set.")


def load_config(
    path: str | os.PathLike,
    overrides: Optional[dict] = None,
) -> AcquisitionConfig:
    """Load YAML config and apply any CLI overrides.

    Args:
        path: Path to a YAML configuration file.
        overrides: Dict of field-name → value to override after loading.

    Returns:
        Validated AcquisitionConfig instance.
    """
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    rlp = raw.get("raw_eeg_log_path")
    if rlp is None or (isinstance(rlp, str) and not rlp.strip()):
        raw_eeg_log_path: Optional[str] = None
    else:
        raw_eeg_log_path = str(rlp)

    cfg = AcquisitionConfig(
        board=str(raw["board"]),
        serial_port=str(raw["serial_port"]),
        sample_rate_hz=int(raw["sample_rate_hz"]),
        channel_count=int(raw["channel_count"]),
        channel_labels=[str(lbl) for lbl in raw["channel_labels"]],
        impedance_max_kohm=float(raw["impedance_max_kohm"]),
        lsl_stream_name=str(raw["lsl_stream_name"]),
        lsl_stream_type=str(raw["lsl_stream_type"]),
        pull_interval_ms=int(raw["pull_interval_ms"]),
        log_interval_s=int(raw["log_interval_s"]),
        raw_eeg_log_path=raw_eeg_log_path,
        raw_eeg_log_format=str(raw.get("raw_eeg_log_format", "csv")),
    )
    cfg.validate()
    return cfg
