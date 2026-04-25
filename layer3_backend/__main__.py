"""CLI entry point: ``python -m layer3_backend [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from layer1_acquisition.logging_config import configure_logging, get_logger
from layer3_backend.config import load_config
from layer3_backend.server import create_app


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m layer3_backend",
        description=(
            "SSVEP-BCI Layer 3 MVP — bridges Layer 2 SELECT commands to a "
            "browser-based SSVEP frontend."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/layer3_default.yaml",
        metavar="PATH",
        help="Path to YAML configuration file (default: configs/layer3_default.yaml).",
    )
    p.add_argument(
        "--host",
        default=None,
        help="Override bind host (default: from config).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override bind port (default: from config).",
    )
    p.add_argument(
        "--layer2-ws",
        default=None,
        metavar="URL",
        help="Override Layer 2 WebSocket URL (e.g. ws://localhost:9001).",
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
    logger = get_logger("layer3_backend.main")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.layer2_ws is not None:
        overrides["layer2_ws_url"] = args.layer2_ws

    try:
        cfg = load_config(config_path, overrides)
    except (ValueError, KeyError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    logger.info(
        "Starting Layer 3 MVP | host=%s | port=%d | upstream=%s | freqs=%s",
        cfg.host,
        cfg.port,
        cfg.layer2_ws_url,
        cfg.stimulus_frequencies_hz,
    )

    app = create_app(cfg)

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
