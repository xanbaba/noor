"""Configuration loading and validation for Layer 3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class BackendConfig:
    layer2_ws_url: str
    host: str
    port: int
    stimulus_frequencies_hz: list[float]
    static_dir: str

    def validate(self) -> None:
        if not self.layer2_ws_url.startswith(("ws://", "wss://")):
            raise ValueError(
                f"layer2_ws_url must start with ws:// or wss://; "
                f"got '{self.layer2_ws_url}'"
            )
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be 1–65535; got {self.port}")
        if not self.stimulus_frequencies_hz:
            raise ValueError("stimulus_frequencies_hz must be non-empty")


def load_config(
    path: str | os.PathLike,
    overrides: Optional[dict] = None,
) -> BackendConfig:
    """Load a Layer 3 YAML config and apply optional overrides."""
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    cfg = BackendConfig(
        layer2_ws_url=str(raw["layer2_ws_url"]),
        host=str(raw["host"]),
        port=int(raw["port"]),
        stimulus_frequencies_hz=[float(f) for f in raw["stimulus_frequencies_hz"]],
        static_dir=str(raw.get("static_dir", "static")),
    )
    cfg.validate()
    return cfg
