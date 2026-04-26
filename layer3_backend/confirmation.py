"""Consecutive-frequency confirmation for Layer 3 (on top of Layer 2 SELECT)."""

from __future__ import annotations


def normalise_frequency_hz(hz: float) -> float:
    """Round to 0.1 Hz so Layer 2 payload decimals match streak keys."""
    return round(float(hz), 1)


class StreakTracker:
    """Emit a normalised frequency after ``required`` identical consecutive hits."""

    def __init__(self, required: int = 5) -> None:
        if required < 1:
            raise ValueError("required must be >= 1")
        self._required = int(required)
        self._streak = 0
        self._current: float | None = None

    @property
    def required(self) -> int:
        return self._required

    def feed(self, frequency_hz: float) -> float | None:
        """Return normalised Hz when the streak completes; otherwise ``None``.

        After a successful confirmation the internal streak resets so the user
        must accumulate ``required`` matches again.
        """
        f = normalise_frequency_hz(frequency_hz)
        if self._current is None or f != self._current:
            self._current = f
            self._streak = 1
            return None
        self._streak += 1
        if self._streak < self._required:
            return None
        self._current = None
        self._streak = 0
        return f
