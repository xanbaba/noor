"""Tests for layer2_processing.config — ProcessingConfig loading & validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from layer2_processing.config import ProcessingConfig, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    lsl_stream_name: BCI_RawEEG
    sample_rate_hz: 500
    notch_freq_hz: 60
    notch_q: 35
    bandpass_low_hz: 5
    bandpass_high_hz: 45
    bandpass_order: 4
    artefact_threshold_uv: 100.0
    epoch_length_s: 2.0
    epoch_step_s: 0.5
    classifier: fbcca
    sub_bands_hz:
      - [6, 90]
      - [14, 90]
    sub_band_filter_order: 5
    weight_a: 1.25
    weight_b: 0.25
    n_harmonics: 3
    stimulus_frequencies_hz: [9.0, 12.0, 15.0]
    snr_min_db: 3.5
    snr_noise_band_hz: 1.0
    snr_channel_index: 0
    websocket_host: localhost
    websocket_port: 9001
    osc_host: 127.0.0.1
    osc_port: 9000
    osc_address: /bci/command
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

def test_load_valid_yaml(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
    assert isinstance(cfg, ProcessingConfig)
    assert cfg.lsl_stream_name == "BCI_RawEEG"
    assert cfg.sample_rate_hz == 500
    assert cfg.stimulus_frequencies_hz == [9.0, 12.0, 15.0]
    assert cfg.classifier == "fbcca"
    assert cfg.artefact_channel_indices is None
    assert cfg.snr_gate_enabled is True


def test_artefact_channel_indices_roundtrip(tmp_path):
    y = _VALID_YAML + "\nartefact_channel_indices: [0, 1, 2]\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.artefact_channel_indices == [0, 1, 2]


def test_artefact_channel_indices_null(tmp_path):
    y = _VALID_YAML + "\nartefact_channel_indices: null\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.artefact_channel_indices is None


def test_artefact_channel_indices_empty_rejected(tmp_path):
    y = _VALID_YAML + "\nartefact_channel_indices: []\n"
    with pytest.raises(ValueError, match="artefact_channel_indices"):
        load_config(_write_yaml(tmp_path, y))


def test_artefact_channel_indices_negative_rejected(tmp_path):
    y = _VALID_YAML + "\nartefact_channel_indices: [0, -1]\n"
    with pytest.raises(ValueError, match="artefact_channel_indices"):
        load_config(_write_yaml(tmp_path, y))


def test_additional_notch_out_of_range_rejected(tmp_path):
    y = _VALID_YAML + "\nadditional_notch_freqs_hz: [300]\n"
    with pytest.raises(ValueError, match="additional_notch_freqs_hz"):
        load_config(_write_yaml(tmp_path, y))


def test_derived_epoch_samples(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
    assert cfg.epoch_length_samples == 1000   # 2.0 s × 500 Hz
    assert cfg.epoch_step_samples == 250      # 0.5 s × 500 Hz


def test_nyquist_property(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
    assert cfg.nyquist_hz == 250.0


def test_override_applies(tmp_path):
    p = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(p, overrides={"classifier": "fbcca", "snr_min_db": 5.0})
    assert cfg.snr_min_db == 5.0


def test_default_config_file_loads():
    """The shipped layer2_default.yaml must be valid."""
    default = Path("configs/layer2_default.yaml")
    if not default.exists():
        pytest.skip("configs/layer2_default.yaml not found — run from project root.")
    cfg = load_config(default)
    assert cfg.stimulus_frequencies_hz, "stimulus_frequencies_hz must be non-empty"
    assert cfg.snr_aggregate == "max"
    assert cfg.use_car is False
    assert cfg.snr_gate_enabled is True


def test_minimal_config_loads():
    p = Path("configs/layer2_minimal.yaml")
    if not p.exists():
        pytest.skip("configs/layer2_minimal.yaml not found — run from project root.")
    cfg = load_config(p)
    assert cfg.snr_gate_enabled is False
    assert cfg.artefact_policy == "ignore"
    assert cfg.prediction_smoothing_window == 0


def test_fast_demo_config_loads():
    p = Path("configs/layer2_fast_demo.yaml")
    if not p.exists():
        pytest.skip("configs/layer2_fast_demo.yaml not found — run from project root.")
    cfg = load_config(p)
    assert cfg.epoch_length_s == 1.5
    assert cfg.epoch_step_s == 0.4
    assert cfg.epoch_length_samples == 750


def test_use_car_roundtrip(tmp_path):
    y = _VALID_YAML + "\nuse_car: true\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.use_car is True


def test_snr_aggregate_roundtrip(tmp_path):
    y = _VALID_YAML + "\nsnr_aggregate: median\nsnr_channel_indices: [0, 1]\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.snr_aggregate == "median"
    assert cfg.snr_channel_indices == [0, 1]


def test_invalid_snr_aggregate_rejected(tmp_path):
    y = _VALID_YAML + "\nsnr_aggregate: bogus\n"
    with pytest.raises(ValueError, match="snr_aggregate"):
        load_config(_write_yaml(tmp_path, y))


def test_empty_snr_channel_indices_when_aggregate_rejected(tmp_path):
    y = _VALID_YAML + "\nsnr_aggregate: max\nsnr_channel_indices: []\n"
    with pytest.raises(ValueError, match="snr_channel_indices"):
        load_config(_write_yaml(tmp_path, y))


def test_invalid_artefact_policy_rejected(tmp_path):
    y = _VALID_YAML + "\nartefact_policy: soft\n"
    with pytest.raises(ValueError, match="artefact_policy"):
        load_config(_write_yaml(tmp_path, y))


def test_artefact_policy_ignore_roundtrip(tmp_path):
    y = _VALID_YAML + "\nartefact_policy: ignore\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.artefact_policy == "ignore"


def test_snr_gate_enabled_roundtrip(tmp_path):
    y = _VALID_YAML + "\nsnr_gate_enabled: false\n"
    cfg = load_config(_write_yaml(tmp_path, y))
    assert cfg.snr_gate_enabled is False


def test_artefact_penalty_out_of_range_rejected(tmp_path):
    y = _VALID_YAML + "\nartefact_penalty: 0\n"
    with pytest.raises(ValueError, match="artefact_penalty"):
        load_config(_write_yaml(tmp_path, y))


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

def _bad_yaml(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(_VALID_YAML)
    raw.update(overrides)
    text = yaml.dump(raw)
    return _write_yaml(tmp_path, text)


def test_empty_stimulus_frequencies_rejected(tmp_path):
    with pytest.raises(ValueError, match="stimulus_frequencies_hz"):
        load_config(_bad_yaml(tmp_path, stimulus_frequencies_hz=[]))


def test_frequency_above_nyquist_rejected(tmp_path):
    # 9 Hz at 500 Hz SR → nyquist 250; 300 Hz is over limit
    with pytest.raises(ValueError, match="stimulus_frequencies_hz"):
        load_config(_bad_yaml(tmp_path, stimulus_frequencies_hz=[300.0]))


def test_bandpass_out_of_order_rejected(tmp_path):
    with pytest.raises(ValueError, match="bandpass"):
        load_config(_bad_yaml(tmp_path, bandpass_low_hz=50.0, bandpass_high_hz=10.0))


def test_empty_sub_bands_rejected(tmp_path):
    with pytest.raises(ValueError, match="sub_bands_hz"):
        load_config(_bad_yaml(tmp_path, sub_bands_hz=[]))


def test_sub_band_above_nyquist_rejected(tmp_path):
    with pytest.raises(ValueError, match="sub_band"):
        load_config(_bad_yaml(tmp_path, sub_bands_hz=[[6, 300]]))


def test_zero_harmonics_rejected(tmp_path):
    with pytest.raises(ValueError, match="n_harmonics"):
        load_config(_bad_yaml(tmp_path, n_harmonics=0))


def test_negative_artefact_threshold_rejected(tmp_path):
    with pytest.raises(ValueError, match="artefact_threshold_uv"):
        load_config(_bad_yaml(tmp_path, artefact_threshold_uv=-1.0))


def test_epoch_step_larger_than_epoch_rejected(tmp_path):
    with pytest.raises(ValueError, match="epoch_step_s"):
        load_config(_bad_yaml(tmp_path, epoch_length_s=1.0, epoch_step_s=2.0))


def test_negative_prediction_smoothing_window_rejected(tmp_path):
    with pytest.raises(ValueError, match="prediction_smoothing_window"):
        load_config(_bad_yaml(tmp_path, prediction_smoothing_window=-1))
