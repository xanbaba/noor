"""Board factory — maps board name strings to concrete AbstractBoard subclasses."""

from __future__ import annotations

from ..config import AcquisitionConfig
from .base import AbstractBoard


def create_board(cfg: AcquisitionConfig) -> AbstractBoard:
    """Instantiate the correct board implementation from the config board name.

    Supported names (case-insensitive):
      - ``cyton``     → CytonBoard (OpenBCI Cyton via USB dongle)
      - ``synthetic`` → SyntheticBoard (BrainFlow built-in, no hardware needed)

    Raises:
        ValueError: If ``cfg.board`` is not a recognised board name.
    """
    name = cfg.board.lower()

    if name == "cyton":
        from .cyton import CytonBoard
        return CytonBoard(cfg)

    if name == "synthetic":
        from .synthetic import SyntheticBoard
        return SyntheticBoard(cfg)

    raise ValueError(
        f"Unknown board '{cfg.board}'. "
        "Valid values: 'cyton', 'synthetic'."
    )
