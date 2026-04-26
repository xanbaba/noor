"""Run Layer 1 acquisition and fullscreen SSVEP flicker in one process (sample-aligned).

Writes ``raw_eeg.csv`` (same schema as :class:`~layer1_acquisition.raw_eeg_file_logger.RawEegFileLogger`)
and ``events.jsonl`` with ``sample_index`` at each ``stim_on`` / ``stim_off``.

Requires optional dependency: ``pip install 'neuron[experiments]'`` (pygame).

Example::

    python -m experiments.ssvep_ml.joint_session --out-dir data/ssvep_pilot \\
        --config configs/cyton_default.yaml --board synthetic --skip-impedance \\
        --trials-per-class 8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from experiments.ssvep_ml.schedule import label_to_frequency_hz, make_trial_schedule

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised when pygame missing
    raise SystemExit(
        "pygame is required for joint_session. Install with: pip install 'neuron[experiments]'"
    ) from exc

from layer1_acquisition.acquisition import AcquisitionLoop
from layer1_acquisition.boards.factory import create_board
from layer1_acquisition.config import load_config
from layer1_acquisition.impedance import ImpedanceGateError, check_impedance
from layer1_acquisition.logging_config import configure_logging, get_logger
from layer1_acquisition.lsl_outlet import RawEegOutlet


def _append_jsonl(path: Path, obj: object) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


def _flicker_loop_fixed(
    *,
    frequency_hz: float,
    duration_s: float,
    instruction_s: float,
    label: str,
    trial_id: int,
    screen: pygame.Surface,
    font: pygame.font.Font,
    events_path: Path,
    total_samples_cb: Callable[[], int],
) -> None:
    w, h = screen.get_size()
    half_period = 1.0 / (2.0 * frequency_hz)

    t_instr_end = time.perf_counter() + instruction_s
    while time.perf_counter() < t_instr_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt
        screen.fill((40, 40, 40))
        txt = font.render("Look at the flicker", True, (220, 220, 220))
        screen.blit(txt, txt.get_rect(center=(w // 2, h // 2)))
        pygame.display.flip()

    _append_jsonl(
        events_path,
        {
            "event": "stim_on",
            "trial_id": trial_id,
            "label": label,
            "frequency_hz": frequency_hz,
            "sample_index": int(total_samples_cb()),
        },
    )

    phase = 0
    t_deadline = time.perf_counter() + duration_s
    next_flip = time.perf_counter()
    while time.perf_counter() < t_deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt
        color = (255, 255, 255) if phase == 0 else (0, 0, 0)
        screen.fill(color)
        pygame.display.flip()
        phase = 1 - phase
        next_flip += half_period
        sleep_s = next_flip - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_flip = time.perf_counter()

    _append_jsonl(
        events_path,
        {
            "event": "stim_off",
            "trial_id": trial_id,
            "sample_index": int(total_samples_cb()),
        },
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint SSVEP recording: Layer1 CSV + aligned events.jsonl.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for raw_eeg.csv, events.jsonl, session_meta.yaml",
    )
    p.add_argument("--config", type=Path, default=Path("configs/cyton_default.yaml"))
    p.add_argument("--board", choices=["cyton", "synthetic"], default=None)
    p.add_argument("--serial-port", default=None)
    p.add_argument("--skip-impedance", action="store_true")
    p.add_argument("--trials-per-class", type=int, default=10)
    p.add_argument("--warmup-s", type=float, default=3.0)
    p.add_argument("--iti-s", type=float, default=2.5)
    p.add_argument("--instruction-s", type=float, default=1.0)
    p.add_argument("--stim-s", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--copy-meta-template",
        action="store_true",
        help="Copy session_template/session_meta.yaml.example to out-dir if missing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    log = get_logger("experiments.ssvep_ml.joint_session")

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_eeg.csv"
    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    if args.copy_meta_template:
        template = (
            Path(__file__).resolve().parent / "session_template" / "session_meta.yaml.example"
        )
        dest = out_dir / "session_meta.yaml"
        if not dest.exists() and template.exists():
            shutil.copy(template, dest)

    overrides = {
        "board": args.board,
        "serial_port": args.serial_port,
        "raw_eeg_log_path": str(csv_path),
    }
    cfg = load_config(args.config, overrides)

    board = create_board(cfg)
    try:
        board.prepare()
    except Exception as exc:
        log.error("Board preparation failed: %s", exc)
        return 1

    try:
        check_impedance(board, cfg, skip=args.skip_impedance)
    except ImpedanceGateError as exc:
        log.error("Impedance gate FAILED: %s", exc)
        board.stop()
        return 1
    except NotImplementedError as exc:
        log.error("%s", exc)
        board.stop()
        return 1

    board.start_stream()
    outlet = RawEegOutlet(cfg, board_name=cfg.board)
    loop = AcquisitionLoop(board, outlet, cfg)

    acq_thread = threading.Thread(target=loop.run, name="acquisition", daemon=True)
    acq_thread.start()

    time.sleep(max(0.0, float(args.warmup_s)))

    rng = random.Random(int(args.seed))
    schedule = make_trial_schedule(int(args.trials_per_class), rng=rng)

    (out_dir / "protocol_version").write_text("ssvep_ml_protocol_v1\n", encoding="utf-8")

    pygame.init()
    pygame.display.set_caption("SSVEP recording — ESC to abort")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("arial", 48)

    try:
        for trial_id, label in enumerate(schedule):
            # ITI
            t_iti = time.perf_counter() + float(args.iti_s)
            while time.perf_counter() < t_iti:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        raise KeyboardInterrupt
                screen.fill((28, 28, 28))
                pygame.display.flip()

            hz = label_to_frequency_hz(label)
            _flicker_loop_fixed(
                frequency_hz=hz,
                duration_s=float(args.stim_s),
                instruction_s=float(args.instruction_s),
                label=label,
                trial_id=trial_id,
                screen=screen,
                font=font,
                events_path=events_path,
                total_samples_cb=lambda: loop.total_samples,
            )
    except KeyboardInterrupt:
        log.info("Aborted by user.")
    finally:
        loop.stop()
        pygame.quit()
        acq_thread.join(timeout=15.0)
        log.info("Session data written under %s", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
