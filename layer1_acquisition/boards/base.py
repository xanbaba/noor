"""AbstractBoard — the extensibility contract for Layer 1 hardware drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractBoard(ABC):
    """All board implementations must satisfy this interface.

    Downstream layers (Layer 2 LSL inlet) never touch board objects directly,
    but this contract ensures that swapping Cyton → Cyton+Daisy / Unicorn /
    g.USBamp requires only a new subclass here — nothing else changes.
    """

    #: Nominal samples per second (must match the LSL stream nominal rate).
    sample_rate_hz: int

    #: Number of EEG channels this board exposes.
    channel_count: int

    #: Human-readable electrode labels, length == channel_count.
    channel_labels: list[str]

    @abstractmethod
    def prepare(self) -> None:
        """Initialise the hardware session.  Must be called before start_stream."""

    @abstractmethod
    def start_stream(self) -> None:
        """Begin sample streaming from the board."""

    @abstractmethod
    def get_chunk(self) -> np.ndarray:
        """Return all samples buffered since the last call.

        Returns:
            Array of shape (channel_count, n_samples), dtype float32.
            Returns an empty array with shape (channel_count, 0) when no new
            samples are available yet.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all board resources."""

    @abstractmethod
    def impedance_kohm(self) -> dict[str, float]:
        """Return a mapping of channel label → impedance in kΩ.

        Only active (non "--") channels need to be included.
        Raises NotImplementedError for boards that do not support impedance
        measurement — callers must handle this and either skip the gate or
        abort with an informative message.
        """
