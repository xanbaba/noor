"""CLI entry point: ``python -m layer2_processing [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layer2_processing.classifiers.factory import registered_classifiers
from layer2_processing.config import load_config
from layer2_processing.logging_config import configure_logging, get_logger
from layer2_processing.lsl_inlet import StreamNotFoundError
from layer2_processing.pipeline import build_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m layer2_processing",
        description=(
            "SSVEP-BCI Layer 2 — consumes a BCI_RawEEG LSL stream and emits "
            "SELECT commands over WebSocket and OSC."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/layer2_default.yaml",
        metavar="PATH",
        help="Path to YAML configuration file (default: configs/layer2_default.yaml).",
    )
    p.add_argument(
        "--classifier",
        choices=registered_classifiers(),
        default=None,
        help="Override the classifier name from the config.",
    )
    p.add_argument(
        "--stream-name",
        default=None,
        help="Override LSL stream name from the config (e.g. BCI_RawEEG_Test).",
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
    logger = get_logger("layer2_processing.main")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    overrides: dict[str, object] = {}
    if args.classifier is not None:
        overrides["classifier"] = args.classifier
    if args.stream_name is not None:
        overrides["lsl_stream_name"] = args.stream_name

    try:
        cfg = load_config(config_path, overrides)
    except (ValueError, KeyError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    logger.info(
        "Starting Layer 2 | classifier=%s | stream=%s | rate=%d Hz | freqs=%s",
        cfg.classifier,
        cfg.lsl_stream_name,
        cfg.sample_rate_hz,
        cfg.stimulus_frequencies_hz,
    )

    pipeline = inlet = ws = osc = None
    try:
        pipeline, inlet, ws, osc = build_pipeline(cfg)
    except StreamNotFoundError as exc:
        logger.error("LSL stream not available: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Failed to bind output socket: %s", exc)
        if ws is not None:
            ws.stop()
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline build failed: %s", exc)
        return 1

    try:
        pipeline.run()
    finally:
        if ws is not None:
            ws.stop()
        if osc is not None:
            osc.close()
        if inlet is not None:
            inlet.close()
        logger.info("Layer 2 shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
