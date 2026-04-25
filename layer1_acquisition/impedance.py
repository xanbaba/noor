"""Impedance gating — blocks session start until electrode contact quality is met.

Architecture requirement:
  Contact impedance must be below 10 kΩ on all active channels before
  acquisition begins.  This is checked once at session start.
"""

from __future__ import annotations

import logging

from .boards.base import AbstractBoard
from .config import AcquisitionConfig

logger = logging.getLogger(__name__)


class ImpedanceGateError(RuntimeError):
    """Raised when one or more channels fail the impedance threshold check."""


def check_impedance(
    board: AbstractBoard,
    cfg: AcquisitionConfig,
    skip: bool = False,
) -> dict[str, float]:
    """Measure electrode impedance and gate session start.

    Args:
        board: Prepared (but not yet streaming) board instance.
        cfg:   Active acquisition config (provides the threshold and labels).
        skip:  If True, bypass the gate entirely (useful for bench testing
               without gel/paste).  A WARNING is emitted.

    Returns:
        Mapping of channel label → measured impedance in kΩ.

    Raises:
        ImpedanceGateError: If any active channel exceeds ``cfg.impedance_max_kohm``
                            and ``skip`` is False.
        NotImplementedError: If the board does not support impedance measurement
                             and ``skip`` is False.
    """
    if skip:
        logger.warning(
            "Impedance gate SKIPPED (--skip-impedance flag set). "
            "Electrode contact quality is unverified — do not use in clinical sessions."
        )
        return {}

    logger.info(
        "Running impedance check (threshold: %.1f kΩ)…", cfg.impedance_max_kohm
    )

    try:
        readings = board.impedance_kohm()
    except NotImplementedError:
        raise NotImplementedError(
            f"Board '{cfg.board}' does not support impedance measurement. "
            "Use --skip-impedance to bypass (bench/dev only)."
        )

    failed: list[str] = []
    for label, kohm in readings.items():
        status = "OK" if kohm <= cfg.impedance_max_kohm else "FAIL"
        logger.info("  %-6s  %6.1f kΩ  [%s]", label, kohm, status)
        if kohm > cfg.impedance_max_kohm:
            failed.append(f"{label} ({kohm:.1f} kΩ)")

    if failed:
        raise ImpedanceGateError(
            f"Impedance threshold exceeded on {len(failed)} channel(s): "
            + ", ".join(failed)
            + f".  Maximum allowed: {cfg.impedance_max_kohm} kΩ.  "
            "Re-apply conductive paste, recheck electrode seating, and retry."
        )

    logger.info("Impedance gate PASSED — all active channels within threshold.")
    return readings
