"""Pseudorandom trial order with bounded runs of the same class."""

from __future__ import annotations

import random


def make_trial_schedule(
    n_per_class: int,
    *,
    max_same_run: int = 4,
    rng: random.Random | None = None,
) -> list[str]:
    """Return a list of length ``2 * n_per_class`` with values ``6hz`` and ``15hz``.

    Each class appears exactly ``n_per_class`` times. No more than
    ``max_same_run`` consecutive trials share the same label.
    """
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    r = rng or random.Random()
    labels = ["6hz"] * n_per_class + ["15hz"] * n_per_class
    for _ in range(50_000):
        r.shuffle(labels)
        run = 1
        ok = True
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run += 1
                if run > max_same_run:
                    ok = False
                    break
            else:
                run = 1
        if ok:
            return labels
    raise RuntimeError(
        "Could not produce a trial schedule; try lowering n_per_class or "
        "increasing max_same_run."
    )


def label_to_frequency_hz(label: str) -> float:
    if label == "6hz":
        return 6.0
    if label == "15hz":
        return 15.0
    raise ValueError(f"Unknown label: {label!r}")
