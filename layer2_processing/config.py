"""Configuration loading and validation for Layer 2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import yaml

SnrAggregate = Literal["single", "max", "mean", "median"]
ArtefactPolicy = Literal["drop", "penalize", "ignore"]


@dataclass
class ProcessingConfig:
    # Stream
    lsl_stream_name: str
    sample_rate_hz: int

    # Preprocessing
    notch_freq_hz: float
    notch_q: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    bandpass_order: int
    artefact_threshold_uv: float

    # Epoching
    epoch_length_s: float
    epoch_step_s: float

    # Classifier
    classifier: str
    sub_bands_hz: list[list[float]]
    sub_band_filter_order: int
    weight_a: float
    weight_b: float
    n_harmonics: int
    stimulus_frequencies_hz: list[float]

    # SNR gate
    snr_min_db: float
    snr_noise_band_hz: float
    snr_channel_index: int

    # Outputs
    websocket_host: str
    websocket_port: int
    osc_host: str
    osc_port: int
    osc_address: str

    # Loop
    inlet_resolve_timeout_s: float = 5.0
    inlet_pull_timeout_s: float = 0.05
    log_interval_s: int = 5
    # ``single`` uses ``snr_channel_index``; else aggregate over ``snr_channel_indices``
    # (None = all rows at runtime).
    snr_aggregate: SnrAggregate = "single"
    snr_channel_indices: Optional[list[int]] = None
    snr_adaptive_relax: bool = False
    snr_adaptive_floor_db: float = 2.5
    # When false, SNR is still computed for the payload but never drops an epoch.
    snr_gate_enabled: bool = True
    # If set, artefact gate uses peak-to-peak only on these LSL row indices; None = all.
    artefact_channel_indices: Optional[list[int]] = None
    # ``drop`` skips classification; ``penalize`` scales confidence; ``ignore`` skips both.
    artefact_policy: ArtefactPolicy = "penalize"
    artefact_penalty: float = 0.65
    # Majority vote window over frequencies that passed SNR (0 or 1 = off).
    prediction_smoothing_window: int = 4
    # Extra mains notches (Hz), applied after ``notch_freq_hz`` (e.g. 50 + 60).
    additional_notch_freqs_hz: list[float] = field(default_factory=list)
    # Common average reference after bandpass (optional; off by default).
    use_car: bool = False

    @property
    def epoch_length_samples(self) -> int:
        return int(round(self.epoch_length_s * self.sample_rate_hz))

    @property
    def epoch_step_samples(self) -> int:
        return int(round(self.epoch_step_s * self.sample_rate_hz))

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate_hz / 2.0

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        nyq = self.nyquist_hz

        if not self.stimulus_frequencies_hz:
            raise ValueError("stimulus_frequencies_hz must be non-empty")
        if any(f <= 0 or f >= nyq for f in self.stimulus_frequencies_hz):
            raise ValueError(
                f"All stimulus_frequencies_hz must be in (0, {nyq}); "
                f"got {self.stimulus_frequencies_hz}"
            )

        if not (0 < self.bandpass_low_hz < self.bandpass_high_hz < nyq):
            raise ValueError(
                f"bandpass must satisfy 0 < low < high < {nyq}; "
                f"got [{self.bandpass_low_hz}, {self.bandpass_high_hz}]"
            )
        if not (0 < self.notch_freq_hz < nyq):
            raise ValueError(
                f"notch_freq_hz must be in (0, {nyq}); got {self.notch_freq_hz}"
            )
        for f in self.additional_notch_freqs_hz:
            if not (0 < f < nyq):
                raise ValueError(
                    f"each additional_notch_freqs_hz must be in (0, {nyq}); got {f}"
                )

        if not self.sub_bands_hz:
            raise ValueError("sub_bands_hz must be non-empty")
        for lo, hi in self.sub_bands_hz:
            if not (0 < lo < hi < nyq):
                raise ValueError(
                    f"Each sub_band must satisfy 0 < low < high < {nyq}; "
                    f"got [{lo}, {hi}]"
                )

        if self.epoch_length_s <= 0 or self.epoch_step_s <= 0:
            raise ValueError("epoch_length_s and epoch_step_s must be > 0")
        if self.epoch_step_s > self.epoch_length_s:
            raise ValueError("epoch_step_s must be <= epoch_length_s")

        if self.n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1")
        if self.artefact_threshold_uv <= 0:
            raise ValueError("artefact_threshold_uv must be > 0")
        if self.artefact_channel_indices is not None:
            if not self.artefact_channel_indices:
                raise ValueError("artefact_channel_indices must be non-empty when set")
            for i in self.artefact_channel_indices:
                if i < 0:
                    raise ValueError(
                        f"artefact_channel_indices must be >= 0; got {self.artefact_channel_indices}"
                    )
        if self.snr_channel_index < 0:
            raise ValueError("snr_channel_index must be >= 0")

        allowed_agg = ("single", "max", "mean", "median")
        if self.snr_aggregate not in allowed_agg:
            raise ValueError(
                f"snr_aggregate must be one of {allowed_agg}; got {self.snr_aggregate!r}"
            )
        if self.snr_aggregate != "single" and self.snr_channel_indices is not None:
            if not self.snr_channel_indices:
                raise ValueError(
                    "snr_channel_indices must be non-empty when set for aggregate SNR"
                )
            for i in self.snr_channel_indices:
                if i < 0:
                    raise ValueError(
                        f"snr_channel_indices must be >= 0; got {self.snr_channel_indices}"
                    )

        if self.artefact_policy not in ("drop", "penalize", "ignore"):
            raise ValueError(
                f"artefact_policy must be 'drop', 'penalize', or 'ignore'; "
                f"got {self.artefact_policy!r}"
            )
        if not (0.0 < self.artefact_penalty <= 1.0):
            raise ValueError("artefact_penalty must be in (0, 1]")

        if self.prediction_smoothing_window < 0:
            raise ValueError("prediction_smoothing_window must be >= 0")


def load_config(
    path: str | os.PathLike,
    overrides: Optional[dict] = None,
) -> ProcessingConfig:
    """Load a Layer 2 YAML config and apply optional overrides."""
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    aci_raw = raw.get("artefact_channel_indices", None)
    if aci_raw is None:
        artefact_channel_indices = None
    else:
        artefact_channel_indices = [int(i) for i in aci_raw]

    add_notch = raw.get("additional_notch_freqs_hz") or []
    additional_notch_freqs_hz = [float(x) for x in add_notch]

    sci_raw = raw.get("snr_channel_indices", None)
    if sci_raw is None:
        snr_channel_indices = None
    else:
        snr_channel_indices = [int(i) for i in sci_raw]

    cfg = ProcessingConfig(
        lsl_stream_name=str(raw["lsl_stream_name"]),
        sample_rate_hz=int(raw["sample_rate_hz"]),
        notch_freq_hz=float(raw["notch_freq_hz"]),
        notch_q=float(raw["notch_q"]),
        bandpass_low_hz=float(raw["bandpass_low_hz"]),
        bandpass_high_hz=float(raw["bandpass_high_hz"]),
        bandpass_order=int(raw["bandpass_order"]),
        artefact_threshold_uv=float(raw["artefact_threshold_uv"]),
        epoch_length_s=float(raw["epoch_length_s"]),
        epoch_step_s=float(raw["epoch_step_s"]),
        classifier=str(raw["classifier"]),
        sub_bands_hz=[[float(lo), float(hi)] for lo, hi in raw["sub_bands_hz"]],
        sub_band_filter_order=int(raw["sub_band_filter_order"]),
        weight_a=float(raw["weight_a"]),
        weight_b=float(raw["weight_b"]),
        n_harmonics=int(raw["n_harmonics"]),
        stimulus_frequencies_hz=[float(f) for f in raw["stimulus_frequencies_hz"]],
        snr_min_db=float(raw["snr_min_db"]),
        snr_noise_band_hz=float(raw["snr_noise_band_hz"]),
        snr_channel_index=int(raw["snr_channel_index"]),
        websocket_host=str(raw["websocket_host"]),
        websocket_port=int(raw["websocket_port"]),
        osc_host=str(raw["osc_host"]),
        osc_port=int(raw["osc_port"]),
        osc_address=str(raw["osc_address"]),
        inlet_resolve_timeout_s=float(raw.get("inlet_resolve_timeout_s", 5.0)),
        inlet_pull_timeout_s=float(raw.get("inlet_pull_timeout_s", 0.05)),
        log_interval_s=int(raw.get("log_interval_s", 5)),
        snr_aggregate=str(raw.get("snr_aggregate", "single")).lower(),  # type: ignore[arg-type]
        snr_channel_indices=snr_channel_indices,
        snr_adaptive_relax=bool(raw.get("snr_adaptive_relax", False)),
        snr_adaptive_floor_db=float(raw.get("snr_adaptive_floor_db", 2.5)),
        snr_gate_enabled=bool(raw.get("snr_gate_enabled", True)),
        artefact_channel_indices=artefact_channel_indices,
        artefact_policy=str(raw.get("artefact_policy", "penalize")).lower(),  # type: ignore[arg-type]
        artefact_penalty=float(raw.get("artefact_penalty", 0.65)),
        prediction_smoothing_window=int(raw.get("prediction_smoothing_window", 4)),
        additional_notch_freqs_hz=additional_notch_freqs_hz,
        use_car=bool(raw.get("use_car", False)),
    )
    cfg.validate()
    return cfg
