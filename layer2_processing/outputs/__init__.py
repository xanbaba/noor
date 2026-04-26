"""Output emitters for Layer 2 SELECT commands.

Both transports broadcast the **same** JSON payload (the contract is frozen
in :mod:`layer2_processing.pipeline`)::

    {"command":"SELECT","frequency":6.0,"snr_db":4.1,"confidence":0.87,"epoch_ms":2000}
"""

from layer2_processing.outputs.osc_emitter import OscEmitter
from layer2_processing.outputs.websocket_emitter import WebSocketEmitter

__all__ = ["OscEmitter", "WebSocketEmitter"]
