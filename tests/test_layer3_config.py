"""Tests for layer3_backend.config — BackendConfig loading & validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from layer3_backend.config import BackendConfig, load_config


_VALID_YAML = textwrap.dedent("""\
    layer2_ws_url: ws://localhost:9001
    host: localhost
    port: 8000
    stimulus_frequencies_hz: [6.0, 20.0]
    static_dir: static
    phrases:
      - id: water
        label: Water, please
        frequency_hz: 6.0
        color: '#00e5ff'
        utterance: Water.
      - id: pain
        label: Pain
        frequency_hz: 20.0
        color: '#ff3d71'
        utterance: I am in pain.
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return p


# ── Happy path ────────────────────────────────────────────────────────


def test_load_valid_yaml(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
    assert isinstance(cfg, BackendConfig)
    assert cfg.host == "localhost"
    assert cfg.port == 8000
    assert cfg.layer2_ws_url == "ws://localhost:9001"
    assert cfg.stimulus_frequencies_hz == [6.0, 20.0]
    assert len(cfg.phrases) == 2
    assert cfg.phrases[0].id == "water"


def test_override_applies(tmp_path):
    cfg = load_config(
        _write_yaml(tmp_path, _VALID_YAML),
        overrides={"port": 9999},
    )
    assert cfg.port == 9999


def test_default_config_file_loads():
    default = Path("configs/layer3_default.yaml")
    if not default.exists():
        pytest.skip("configs/layer3_default.yaml not present — run from project root")
    cfg = load_config(default)
    assert cfg.stimulus_frequencies_hz == [6.0, 20.0]
    assert len(cfg.phrases) == 2
    ids = {p.id for p in cfg.phrases}
    assert ids == {"action_select", "action_next"}


# ── Validation ────────────────────────────────────────────────────────


def _bad_yaml(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(_VALID_YAML)
    raw.update(overrides)
    return _write_yaml(tmp_path, yaml.dump(raw))


def test_bad_ws_url_rejected(tmp_path):
    with pytest.raises(ValueError, match="ws://"):
        load_config(_bad_yaml(tmp_path, layer2_ws_url="http://localhost:9001"))


def test_invalid_port_rejected(tmp_path):
    with pytest.raises(ValueError, match="port"):
        load_config(_bad_yaml(tmp_path, port=99999))


def test_empty_frequencies_rejected(tmp_path):
    with pytest.raises(ValueError, match="stimulus_frequencies_hz"):
        load_config(_bad_yaml(tmp_path, stimulus_frequencies_hz=[]))


def test_phrase_frequency_not_in_stimulus_rejected(tmp_path):
    raw = yaml.safe_load(_VALID_YAML)
    raw["phrases"] = [
        {
            "id": "x",
            "label": "X",
            "frequency_hz": 99.0,
            "color": "#fff",
            "utterance": "no",
        }
    ]
    p = _write_yaml(tmp_path, yaml.dump(raw))
    with pytest.raises(ValueError, match="frequency_hz"):
        load_config(p)


def test_missing_phrases_key_errors(tmp_path):
    raw = yaml.safe_load(_VALID_YAML)
    del raw["phrases"]
    p = _write_yaml(tmp_path, yaml.dump(raw))
    with pytest.raises(ValueError, match="phrases"):
        load_config(p)
