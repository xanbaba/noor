"""Configuration loading and validation for Layer 3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import yaml


def _norm_hz(hz: float) -> float:
    return round(float(hz), 1)


@dataclass(frozen=True)
class PhraseCard:
    """One communicative phrase bound to a single SSVEP flicker frequency."""

    id: str
    label: str
    frequency_hz: float
    color: str
    utterance: str

    @staticmethod
    def from_mapping(m: dict[str, Any]) -> "PhraseCard":
        return PhraseCard(
            id=str(m["id"]),
            label=str(m["label"]),
            frequency_hz=float(m["frequency_hz"]),
            color=str(m["color"]),
            utterance=str(m["utterance"]),
        )


@dataclass
class BackendConfig:
    layer2_ws_url: str
    host: str
    port: int
    stimulus_frequencies_hz: list[float]
    static_dir: str
    phrases: list[PhraseCard]
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

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

        allowed = {_norm_hz(f) for f in self.stimulus_frequencies_hz}
        if not self.phrases:
            raise ValueError("phrases must be a non-empty list")
        seen_ids: set[str] = set()
        seen_freqs: set[float] = set()
        for p in self.phrases:
            if not p.id or not p.label:
                raise ValueError(f"phrase id/label must be non-empty: {p!r}")
            if p.id in seen_ids:
                raise ValueError(f"duplicate phrase id: {p.id!r}")
            seen_ids.add(p.id)
            fk = _norm_hz(p.frequency_hz)
            if fk not in allowed:
                raise ValueError(
                    f"phrase {p.id!r} frequency_hz={p.frequency_hz} not in "
                    f"stimulus_frequencies_hz {self.stimulus_frequencies_hz}"
                )
            if fk in seen_freqs:
                raise ValueError(
                    f"duplicate phrase frequency_hz={p.frequency_hz} "
                    f"(normalised {fk})"
                )
            seen_freqs.add(fk)


def load_config(
    path: str | os.PathLike,
    overrides: Optional[dict] = None,
) -> BackendConfig:
    """Load a Layer 3 YAML config and apply optional overrides."""
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    phrases_raw = raw.get("phrases")
    if not phrases_raw:
        raise ValueError("phrases must be set in YAML (non-empty list)")
    phrases = [PhraseCard.from_mapping(dict(x)) for x in phrases_raw]

    cfg = BackendConfig(
        layer2_ws_url=str(raw["layer2_ws_url"]),
        host=str(raw["host"]),
        port=int(raw["port"]),
        stimulus_frequencies_hz=[float(f) for f in raw["stimulus_frequencies_hz"]],
        static_dir=str(raw.get("static_dir", "static")),
        phrases=phrases,
        elevenlabs_voice_id=str(
            raw.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")
        ),
    )
    cfg.validate()
    return cfg
