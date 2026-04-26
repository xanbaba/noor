"""Layer 2 output verifier — listens to both transports and prints commands.

Spawns:

  * an OSC server on UDP ``--osc-host:--osc-port`` listening on ``/bci/command``
  * a WebSocket client connected to ``ws://--ws-host:--ws-port``

Every received SELECT payload is printed with a transport tag so you can
confirm both emitters fire identical content::

    [OSC] {"command":"SELECT","frequency":6.0,"snr_db":4.1,...}
    [WS ] {"command":"SELECT","frequency":6.0,"snr_db":4.1,...}

Usage::

    python scripts/verify_layer2_output.py
    python scripts/verify_layer2_output.py --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from threading import Thread

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer
except ImportError:
    print("ERROR: python-osc not installed. Run: pip install python-osc", file=sys.stderr)
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify Layer 2 SELECT command output (OSC + WebSocket)."
    )
    p.add_argument("--osc-host", default="127.0.0.1", help="UDP bind host (default: 127.0.0.1).")
    p.add_argument("--osc-port", type=int, default=9000, help="UDP port (default: 9000).")
    p.add_argument("--osc-address", default="/bci/command", help="OSC address (default: /bci/command).")
    p.add_argument("--ws-host", default="localhost", help="WebSocket host (default: localhost).")
    p.add_argument("--ws-port", type=int, default=9001, help="WebSocket port (default: 9001).")
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to listen (default: 0 = forever, until Ctrl+C).",
    )
    return p


def _run_osc_server(host: str, port: int, address: str, stop_event) -> None:
    def handler(addr: str, *args) -> None:
        msg = args[0] if args else ""
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [OSC] {msg}")

    disp = Dispatcher()
    disp.map(address, handler)
    server = BlockingOSCUDPServer((host, port), disp)
    print(f"OSC listening on udp://{host}:{port} (address {address})")

    def watcher():
        stop_event.wait()
        server.shutdown()

    Thread(target=watcher, daemon=True).start()
    server.serve_forever()


async def _run_ws_client(host: str, port: int, stop_event) -> None:
    uri = f"ws://{host}:{port}"
    print(f"WS  connecting to {uri}")
    backoff = 0.5
    while not stop_event.is_set():
        try:
            async with websockets.connect(uri, open_timeout=2.0) as ws:
                print(f"WS  connected to {uri}")
                backoff = 0.5
                async for message in ws:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] [WS ] {message}")
                    if stop_event.is_set():
                        break
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
            if stop_event.is_set():
                return
            print(f"WS  reconnect in {backoff:.1f}s ({exc.__class__.__name__})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import threading
    stop_event = threading.Event()

    osc_thread = Thread(
        target=_run_osc_server,
        args=(args.osc_host, args.osc_port, args.osc_address, stop_event),
        daemon=True,
    )
    osc_thread.start()

    async def _ws_with_timeout():
        ws_task = asyncio.create_task(
            _run_ws_client(args.ws_host, args.ws_port, stop_event)
        )
        if args.duration > 0:
            try:
                await asyncio.wait_for(ws_task, timeout=args.duration)
            except asyncio.TimeoutError:
                pass
        else:
            await ws_task

    try:
        asyncio.run(_ws_with_timeout())
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.2)

    print("\nVerifier exit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
