"""CLI entry point: python -m layer1_acquisition [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .logging_config import configure_logging, get_logger
from .boards.factory import create_board
from .impedance import check_impedance, ImpedanceGateError
from .lsl_outlet import RawEegOutlet
from .acquisition import AcquisitionLoop


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m layer1_acquisition",
        description="SSVEP-BCI Layer 1 — streams raw EEG from OpenBCI Cyton to LSL.",
    )
    p.add_argument(
        "--config",
        default="configs/cyton_default.yaml",
        metavar="PATH",
        help="Path to YAML configuration file (default: configs/cyton_default.yaml).",
    )
    p.add_argument(
        "--board",
        choices=["cyton", "synthetic"],
        default=None,
        help="Override the board type from the config file.",
    )
    p.add_argument(
        "--serial-port",
        default=None,
        metavar="PORT",
        help="Override the serial port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux).",
    )
    p.add_argument(
        "--raw-eeg-log",
        default=None,
        metavar="PATH",
        help="Append raw µV samples (all channels, CSV) to this file. Overrides config.",
    )
    p.add_argument(
        "--skip-impedance",
        action="store_true",
        default=False,
        help="Bypass impedance gate (bench/dev only — not for clinical use).",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    logger = get_logger("layer1_acquisition.main")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    overrides = {
        "board": args.board,
        "serial_port": args.serial_port,
        "raw_eeg_log_path": args.raw_eeg_log,
    }

    try:
        cfg = load_config(config_path, overrides)
    except (ValueError, KeyError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    logger.info(
        "Starting Layer 1 | board=%s | rate=%d Hz | stream=%s",
        cfg.board,
        cfg.sample_rate_hz,
        cfg.lsl_stream_name,
    )

    board = create_board(cfg)

    try:
        board.prepare()
    except Exception as exc:
        logger.error("Board preparation failed: %s", exc)
        return 1

    try:
        check_impedance(board, cfg, skip=args.skip_impedance)
    except ImpedanceGateError as exc:
        logger.error("Impedance gate FAILED: %s", exc)
        board.stop()
        return 1
    except NotImplementedError as exc:
        logger.error("%s", exc)
        board.stop()
        return 1

    board.start_stream()

    outlet = RawEegOutlet(cfg, board_name=cfg.board)
    loop = AcquisitionLoop(board, outlet, cfg)

    logger.info(
        "LSL stream '%s' is live. Consumers can now connect. Press Ctrl+C to stop.",
        cfg.lsl_stream_name,
    )
    loop.run()

    logger.info("Layer 1 shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
