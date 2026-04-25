"""Synthetic SSVEP LSL source — lets the whole Layer 2 stack run with no Cyton.

Emits an 8-channel ``BCI_RawEEG`` stream at 500 Hz containing:

  * A pure sinusoid at the chosen stimulus frequency on the Oz row (channel 0)
    with optional harmonics, simulating a steady gaze response.
  * Pink (1/f) noise on every channel at a calibrated amplitude so the FBCCA
    classifier sees a realistic signal-to-noise ratio (~10 dB by default).

Usage::

    python scripts/synthetic_ssvep_source.py --frequency 12.0 --duration 60

    # Randomly switch among several frequencies every few seconds (same Oz SSVEP)
    python scripts/synthetic_ssvep_source.py --frequencies 12 15 --frequency-switch-s 3

Run this in one terminal, then in another start::

    python -m layer2_processing --stream-name BCI_RawEEG_Test
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Sequence

import numpy as np

try:
    from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock
except ImportError:
    print("ERROR: pylsl is not installed.  Run: pip install pylsl", file=sys.stderr)
    sys.exit(1)


_DEFAULT_LABELS = ("Oz", "O1", "O2", "Pz", "--", "--", "--", "--")


def pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """Generate (channels, n) pink (1/f) noise via FFT shaping.

    Result is unit-variance per channel.
    """
    white = rng.standard_normal((n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1.0  # avoid div-by-zero at DC
    spectrum /= np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples, axis=1)
    pink -= pink.mean(axis=1, keepdims=True)
    std = pink.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    pink /= std
    return pink.astype(np.float32)


def _build_outlet(
    stream_name: str,
    sample_rate_hz: int,
    channel_labels: Sequence[str],
) -> StreamOutlet:
    info = StreamInfo(
        name=stream_name,
        type="EEG",
        channel_count=len(channel_labels),
        nominal_srate=float(sample_rate_hz),
        channel_format=cf_float32,
        source_id=f"synthetic_ssvep_{uuid.uuid4().hex[:8]}",
    )
    channels_node = info.desc().append_child("channels")
    for label in channel_labels:
        ch = channels_node.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info, max_buffered=360)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthetic SSVEP source for Layer 2 testing."
    )
    p.add_argument(
        "--stream-name",
        default="BCI_RawEEG_Test",
        help="LSL stream name (default: BCI_RawEEG_Test).",
    )
    p.add_argument(
        "--frequency",
        type=float,
        default=12.0,
        help="Single stimulus frequency in Hz on Oz (default: 12.0). "
        "Ignored if --frequencies is set.",
    )
    p.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=None,
        metavar="HZ",
        help="One or more stimulus frequencies (Hz). Each --frequency-switch-s "
        "wall-clock interval, a new frequency is chosen uniformly at random from "
        "this list (for testing dynamic gaze / classifier tracking). "
        "Overrides --frequency when given.",
    )
    p.add_argument(
        "--frequency-switch-s",
        type=float,
        default=2.0,
        help="Seconds between random picks from --frequencies (default: 2.0). "
        "Ignored when only a single frequency is active.",
    )
    p.add_argument(
        "--amplitude-uv",
        type=float,
        default=15.0,
        help="Stimulus amplitude in µV (default: 15).",
    )
    p.add_argument(
        "--noise-uv",
        type=float,
        default=8.0,
        help="Pink-noise standard deviation in µV (default: 8).",
    )
    p.add_argument(
        "--harmonics",
        type=int,
        default=2,
        help="Number of harmonics in addition to the fundamental (default: 2).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=500,
        help="Nominal sample rate in Hz (default: 500).",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=8,
        help="Number of channels to emit (default: 8).",
    )
    p.add_argument(
        "--ssvep-channel",
        type=int,
        default=0,
        help="0-based channel index that carries the SSVEP signal (default: 0 = Oz).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to stream (default: 0 = forever, until Ctrl+C).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Samples per push (default: 10 = 20 ms at 500 Hz).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible noise.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.frequencies is not None:
        freq_list = [float(f) for f in args.frequencies]
    else:
        freq_list = [float(args.frequency)]

    if not freq_list:
        print("ERROR: frequency list is empty.", file=sys.stderr)
        return 2
    if any(f <= 0 for f in freq_list):
        print("ERROR: all frequencies must be > 0 Hz.", file=sys.stderr)
        return 2
    if len(freq_list) > 1 and args.frequency_switch_s <= 0:
        print(
            "ERROR: --frequency-switch-s must be > 0 when using multiple frequencies.",
            file=sys.stderr,
        )
        return 2

    if args.channels < args.ssvep_channel + 1:
        print(
            f"ERROR: --ssvep-channel {args.ssvep_channel} out of range for "
            f"{args.channels}-channel stream.",
            file=sys.stderr,
        )
        return 2

    labels = list(_DEFAULT_LABELS[: args.channels])
    if len(labels) < args.channels:
        labels += [f"CH{i+1}" for i in range(len(labels), args.channels)]

    outlet = _build_outlet(args.stream_name, args.sample_rate, labels)

    fs = args.sample_rate
    chunk = args.chunk_size
    chunk_dt = chunk / fs
    rotate_freqs = len(freq_list) > 1
    current_f = float(rng.choice(freq_list)) if rotate_freqs else freq_list[0]

    freq_desc = (
        f"random among {freq_list} every {args.frequency_switch_s:g}s"
        if rotate_freqs
        else f"{current_f:g} Hz"
    )
    print(
        f"Streaming '{args.stream_name}' "
        f"({args.channels} ch @ {fs} Hz, {freq_desc} on ch {args.ssvep_channel}, "
        f"sig={args.amplitude_uv} µV / noise={args.noise_uv} µV)"
    )
    print(
        f"  duration={'∞' if args.duration <= 0 else f'{args.duration:.1f}s'} | "
        f"chunk={chunk} samples ({chunk_dt*1e3:.1f} ms)"
    )
    print("Press Ctrl+C to stop.\n")

    n_emitted = 0
    t0 = time.monotonic()
    next_emit = t0
    next_freq_switch = t0 + args.frequency_switch_s if rotate_freqs else float("inf")

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if args.duration > 0 and elapsed >= args.duration:
                break

            if rotate_freqs and now >= next_freq_switch:
                prev = current_f
                current_f = float(rng.choice(freq_list))
                next_freq_switch = now + args.frequency_switch_s
                print(
                    f"[{time.strftime('%H:%M:%S')}] switched stimulus "
                    f"{prev:g} Hz → {current_f:g} Hz",
                    flush=True,
                )

            # Time vector for this chunk (continuous phase)
            t_idx = np.arange(n_emitted, n_emitted + chunk) / fs
            ssvep = np.zeros(chunk, dtype=np.float64)
            for h in range(1, args.harmonics + 2):  # fundamental + harmonics
                amp = args.amplitude_uv / h  # 1/h amplitude rolloff
                ssvep += amp * np.sin(2 * np.pi * h * current_f * t_idx)

            noise = pink_noise(chunk, args.channels, rng) * args.noise_uv
            data = noise.astype(np.float32)
            data[args.ssvep_channel] += ssvep.astype(np.float32)

            outlet.push_chunk(data.T.tolist(), local_clock())
            n_emitted += chunk

            # Pace to wall-clock so the consumer sees ~ nominal rate.
            next_emit += chunk_dt
            sleep_for = next_emit - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t0
    print(
        f"\nDone. Emitted {n_emitted} samples in {elapsed:.2f}s "
        f"(~{n_emitted/max(elapsed, 1e-6):.0f} Hz)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
