"""Fullscreen SSVEP flicker with LSL string markers (LabRecorder / XDF workflows).

Emits trial markers on a separate LSL stream so you can record EEG + markers
with LabRecorder while running Layer 1 acquisition in another process.

Requires: ``pip install 'neuron[experiments]'`` (pygame). Uses existing ``pylsl``.

Example::

    python -m experiments.ssvep_ml.stimulus_lsl_markers \\
        --stream-name SSVEP_Markers --trials-per-class 10

Markers are UTF-8 strings like ``stim_on,trial=0,label=6hz`` and ``stim_off,trial=0``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from experiments.ssvep_ml.schedule import label_to_frequency_hz, make_trial_schedule

try:
    import pygame
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pygame is required. Install with: pip install 'neuron[experiments]'"
    ) from exc

from pylsl import IRREGULAR_RATE, StreamInfo, StreamOutlet, cf_string, local_clock

from layer1_acquisition.logging_config import configure_logging, get_logger


def _push_marker(outlet: StreamOutlet, text: str) -> None:
    outlet.push_sample([text], local_clock())


def _flicker_block(
    *,
    frequency_hz: float,
    duration_s: float,
    instruction_s: float,
    trial_id: int,
    label: str,
    screen: pygame.Surface,
    font: pygame.font.Font,
    marker_outlet: StreamOutlet,
) -> None:
    w, h = screen.get_size()
    half_period = 1.0 / (2.0 * frequency_hz)

    t0 = time.perf_counter() + instruction_s
    while time.perf_counter() < t0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt
        screen.fill((40, 40, 40))
        txt = font.render("Look at the flicker", True, (220, 220, 220))
        screen.blit(txt, txt.get_rect(center=(w // 2, h // 2)))
        pygame.display.flip()

    _push_marker(
        marker_outlet,
        f"stim_on,trial={trial_id},label={label},hz={frequency_hz}",
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

    _push_marker(marker_outlet, f"stim_off,trial={trial_id}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SSVEP flicker + LSL string markers.")
    p.add_argument("--stream-name", default="SSVEP_Markers")
    p.add_argument("--trials-per-class", type=int, default=10)
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--iti-s", type=float, default=2.5)
    p.add_argument("--instruction-s", type=float, default=1.0)
    p.add_argument("--stim-s", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_logging(args.log_level)
    log = get_logger("experiments.ssvep_ml.stimulus_lsl_markers")

    info = StreamInfo(
        name=args.stream_name,
        type="Markers",
        channel_format=cf_string,
        channel_count=1,
        nominal_srate=IRREGULAR_RATE,
        source_id="neuron_ssvep_ml",
    )
    outlet = StreamOutlet(info)
    log.info("LSL marker stream '%s' is live.", args.stream_name)

    rng = random.Random(int(args.seed))
    schedule = make_trial_schedule(int(args.trials_per_class), rng=rng)

    time.sleep(max(0.0, float(args.warmup_s)))

    pygame.init()
    pygame.display.set_caption("SSVEP markers — ESC to quit")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("arial", 48)

    try:
        for trial_id, label in enumerate(schedule):
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
            _flicker_block(
                frequency_hz=hz,
                duration_s=float(args.stim_s),
                instruction_s=float(args.instruction_s),
                trial_id=trial_id,
                label=label,
                screen=screen,
                font=font,
                marker_outlet=outlet,
            )
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
