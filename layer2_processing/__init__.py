"""Layer 2 — Signal Processing and Classification.

Consumes a continuous LSL `BCI_RawEEG` stream and emits discrete `SELECT`
commands at the user's gazed stimulus frequency over WebSocket and OSC.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
