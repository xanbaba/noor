"""Temporal smoothing for SSVEP command frequencies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def smoothed_frequency_hz(history: Sequence[float]) -> float:
    """Return the modal frequency; ties break toward the most recent occurrence.

    Args:
        history: Non-empty sequence of candidate frequencies (Hz), typically
            the classifier output from recent epochs that passed the SNR gate.

    Raises:
        ValueError: if ``history`` is empty.
    """
    if not history:
        raise ValueError("history must be non-empty")
    counts = Counter(history)
    best = max(counts.values())
    candidates = [f for f, c in counts.items() if c == best]
    if len(candidates) == 1:
        return float(candidates[0])
    last_i: dict[float, int] = {}
    for i, f in enumerate(history):
        if f in candidates:
            last_i[f] = i
    return float(max(candidates, key=lambda f: last_i[f]))
