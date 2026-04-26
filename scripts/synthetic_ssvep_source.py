"""Synthetic SSVEP LSL source — lets the whole Layer 2 stack run with no Cyton.

Emits an 8-channel ``BCI_RawEEG`` stream at 500 Hz (same contract as Layer 1:
float32, microvolts, channel labels).

**Presets**

* **minimal** — SSVEP on one channel + pink noise (legacy, fast CI).
* **realistic** — spatial SSVEP (Oz/O1/O2/Pz), pink, mains, common-mode, alpha,
  drift, EMG spikes, plus *mild* optional: ADC clip headroom, electrode pops,
  simplified blinks, EOG-like slow ramps.  Not anatomically exact.
* **stress** — same machinery with aggressive rates/amplitudes + packet gaps
  + tighter ADC clip for worst-case Layer 2 testing.

**Artifact flags** (see README; ``-1`` = use preset default where applicable)

===================  =========
Artifact            CLI / behaviour
===================  =========
ADC clip            ``--adc-clip-uv``
Electrode pop       ``--pop-rate-per-min``, ``--pop-step-uv``
Simplified blink    ``--blink-rate-per-min``, ``--blink-duration-ms``, ``--blink-amplitude-uv``
EOG-like ramp       ``--eog-ramp-peak-uv``, ``--eog-ramp-hz``
Packet gap          ``--packet-loss-prob-chunk``, ``--packet-loss-mean-gap-ms``
===================  =========

Usage::

    python scripts/synthetic_ssvep_source.py --frequency 6.0 --duration 60

    python scripts/synthetic_ssvep_source.py --preset realistic --frequency 6.0

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

# Local helpers (same directory as this script; not an installed package)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import synthetic_signal_model as ssm  # noqa: E402

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


def _bandpass_alpha(
    n_samples: int, fs: float, rng: np.random.Generator, low: float = 9.0, high: float = 11.0
) -> np.ndarray:
    """Single-channel ~10 Hz band-limited noise (one draw, shared across occipital).

    ``sosfiltfilt`` needs edge padding — short LSL chunks (e.g. 10 samples) are
    extended, filtered, then cropped back to ``n_samples``.
    """
    pad = max(256, n_samples * 4)
    white = rng.standard_normal(n_samples + 2 * pad).astype(np.float64)
    sos = signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, white)
    x = x[pad : pad + n_samples]
    x -= x.mean()
    s = x.std()
    if s > 1e-9:
        x /= s
    return x.astype(np.float64)


@dataclass
class StreamState:
    """Carries slow-varying and Poisson-timed state across chunks."""

    line_phase: np.ndarray  # (n_ch,) radians, fixed per channel at init
    drift: np.ndarray  # (n_ch,) uV offset, random-walk updated per chunk
    emg_timer: float
    pop_timer: float
    blink_arm_timer: float  # time until next blink *starts* (when idle)
    eog: ssm.EogRampState
    # Streaming blink (template may span many chunks)
    blink_profile: np.ndarray
    blink_cursor: int
    blink_samples_left: int
    blink_amplitude_uv: float


def _init_stream_state(
    n_ch: int,
    rng: np.random.Generator,
    emg_rate_per_s: float,
    pop_rate_per_s: float,
    blink_rate_per_s: float,
) -> StreamState:
    return StreamState(
        line_phase=rng.uniform(0, 2 * np.pi, size=n_ch),
        drift=np.zeros(n_ch, dtype=np.float64),
        emg_timer=ssm.poisson_next_interval(emg_rate_per_s, rng),
        pop_timer=ssm.poisson_next_interval(pop_rate_per_s, rng),
        blink_arm_timer=ssm.poisson_next_interval(blink_rate_per_s, rng),
        eog=ssm.EogRampState(),
        blink_profile=np.zeros(0, dtype=np.float64),
        blink_cursor=0,
        blink_samples_left=0,
        blink_amplitude_uv=0.0,
    )


def _spatial_gains(n_ch: int, ssvep_ch: int, spatial: bool) -> np.ndarray:
    """Relative coupling of the SSVEP source onto each channel (Cyton layout)."""
    g = np.zeros(n_ch, dtype=np.float64)
    if not spatial:
        g[ssvep_ch] = 1.0
        return g
    # Occipital cluster: strongest on Oz, weaker on O1/O2, small on Pz.
    layout = {0: 1.0, 1: 0.52, 2: 0.52, 3: 0.18}
    for idx, val in layout.items():
        if idx < n_ch:
            g[idx] = val
    if g.sum() < 1e-9:
        g[ssvep_ch] = 1.0
    return g


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
        description="Synthetic SSVEP source for Layer 2 testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=["minimal", "realistic", "stress"],
        default="minimal",
        help="minimal | realistic | stress — see module docstring.",
    )
    p.add_argument(
        "--stream-name",
        default="BCI_RawEEG_Test",
        help="LSL stream name (default: BCI_RawEEG_Test).",
    )
    p.add_argument(
        "--frequency",
        type=float,
        default=6.0,
        help="Single stimulus frequency in Hz on Oz (default: 6.0). "
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
        help="Seconds between random picks from --frequencies. "
        "Ignored when only a single frequency is active.",
    )
    p.add_argument(
        "--amplitude-uv",
        type=float,
        default=-1.0,
        help="Fundamental SSVEP peak amplitude in µV on Oz (before spatial spread). "
        "-1 = preset default (15 minimal / 5.5 realistic).",
    )
    p.add_argument(
        "--noise-uv",
        type=float,
        default=-1.0,
        help="Per-channel pink-noise RMS scale in µV. "
        "-1 = preset default (8 minimal / 12 realistic).",
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
        help="Primary SSVEP channel index (default: 0 = Oz). With --preset realistic "
        "and spatial spread, neighbours O1/O2/Pz also receive scaled SSVEP.",
    )
    p.add_argument(
        "--no-spatial-ssvep",
        action="store_true",
        help="Force SSVEP onto --ssvep-channel only even when preset=realistic.",
    )
    p.add_argument(
        "--line-noise-uv",
        type=float,
        default=-1.0,
        help="Peak µV of 50/60 Hz sinusoid on every channel (-1 = use preset default).",
    )
    p.add_argument(
        "--mains-hz",
        type=float,
        default=60.0,
        help="Mains frequency for line noise (60 US, 50 EU).",
    )
    p.add_argument(
        "--common-mode-uv",
        type=float,
        default=-1.0,
        help="RMS µV of one pink-noise waveform added identically to all channels "
        "(-1 = use preset default).",
    )
    p.add_argument(
        "--alpha-occipital-uv",
        type=float,
        default=-1.0,
        help="RMS µV of shared 9–11 Hz band noise on Oz/O1/O2 (0 = off, -1 = preset).",
    )
    p.add_argument(
        "--drift-uv-per-s",
        type=float,
        default=-1.0,
        help="Random-walk drift: Gaussian step std-dev in µV per second of wall time "
        "(-1 = preset, 0 = off).",
    )
    p.add_argument(
        "--emg-spikes-per-min",
        type=float,
        default=-1.0,
        help="Poisson rate of brief high-amplitude muscle-like spikes on a random "
        "channel (-1 = preset realistic uses 1.5, minimal uses 0).",
    )
    p.add_argument(
        "--emg-spike-uv",
        type=float,
        default=120.0,
        help="Half-amplitude (µV) of rectangular EMG spike (default: 120).",
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
    # --- ADC / electrode / blink / EOG / packet (artifact simulation) ----------
    p.add_argument(
        "--adc-clip-uv",
        type=float,
        default=-1.0,
        help="Hard clip to +/- this value in uV (0 = off, -1 = preset default).",
    )
    p.add_argument(
        "--pop-rate-per-min",
        type=float,
        default=-1.0,
        help="Poisson rate of electrode pops per minute (-1 = preset, 0 = off).",
    )
    p.add_argument(
        "--pop-step-uv",
        type=float,
        default=220.0,
        help="Initial step amplitude (uV) for a pop, before decay tail.",
    )
    p.add_argument(
        "--blink-rate-per-min",
        type=float,
        default=-1.0,
        help="Simplified blink events per minute on occipital channels (-1 = preset).",
    )
    p.add_argument(
        "--blink-duration-ms",
        type=float,
        default=210.0,
        help="Full width of one blink template in ms.",
    )
    p.add_argument(
        "--blink-amplitude-uv",
        type=float,
        default=-1.0,
        help="Peak uV scale on Oz for blink template (-1 = preset).",
    )
    p.add_argument(
        "--eog-ramp-peak-uv",
        type=float,
        default=-1.0,
        help="Peak uV of slow lateral EOG-like sinusoid (-1 = preset, 0 = off).",
    )
    p.add_argument(
        "--eog-ramp-hz",
        type=float,
        default=0.12,
        help="Frequency of EOG-like ramp (Hz), default 0.12.",
    )
    p.add_argument(
        "--packet-loss-prob-chunk",
        type=float,
        default=-1.0,
        help="Per-chunk probability of a gap (no push, phase still advances). "
        "-1 = preset (0 for minimal/realistic), stress sets non-zero.",
    )
    p.add_argument(
        "--packet-loss-mean-gap-ms",
        type=float,
        default=48.0,
        help="Mean extra wall-clock gap in ms (exponential) when a packet-loss event fires.",
    )
    return p


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    """Fill -1 sentinels from preset tables."""
    if args.preset == "realistic":
        auto = {
            "line_noise_uv": 2.8,
            "common_mode_uv": 3.5,
            "alpha_occipital_uv": 5.5,
            "drift_uv_per_s": 0.9,
            "emg_spikes_per_min": 1.5,
            "noise_uv": 12.0,
            "amplitude_uv": 5.5,
            "adc_clip_uv": 220.0,
            "pop_rate_per_min": 1.0,
            "blink_rate_per_min": 0.55,
            "blink_amplitude_uv": 62.0,
            "eog_ramp_peak_uv": 11.0,
            "packet_loss_prob_chunk": 0.0,
        }
    elif args.preset == "stress":
        auto = {
            "line_noise_uv": 3.2,
            "common_mode_uv": 5.0,
            "alpha_occipital_uv": 7.0,
            "drift_uv_per_s": 1.4,
            "emg_spikes_per_min": 4.0,
            "noise_uv": 18.0,
            "amplitude_uv": 4.2,
            "adc_clip_uv": 92.0,
            "pop_rate_per_min": 4.5,
            "blink_rate_per_min": 2.8,
            "blink_amplitude_uv": 95.0,
            "eog_ramp_peak_uv": 24.0,
            "packet_loss_prob_chunk": 0.11,
        }
    else:
        auto = {
            "line_noise_uv": 0.0,
            "common_mode_uv": 0.0,
            "alpha_occipital_uv": 0.0,
            "drift_uv_per_s": 0.0,
            "emg_spikes_per_min": 0.0,
            "noise_uv": 8.0,
            "amplitude_uv": 15.0,
            "adc_clip_uv": 0.0,
            "pop_rate_per_min": 0.0,
            "blink_rate_per_min": 0.0,
            "blink_amplitude_uv": 0.0,
            "eog_ramp_peak_uv": 0.0,
            "packet_loss_prob_chunk": 0.0,
        }

    for key, val in auto.items():
        if getattr(args, key) == -1.0:
            setattr(args, key, val)

    if args.packet_loss_prob_chunk < 0:
        args.packet_loss_prob_chunk = 0.0
    if args.packet_loss_mean_gap_ms <= 0:
        args.packet_loss_mean_gap_ms = 48.0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _apply_preset_defaults(args)

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

    fs = float(args.sample_rate)
    chunk = args.chunk_size
    chunk_dt = chunk / fs
    rotate_freqs = len(freq_list) > 1
    current_f = float(rng.choice(freq_list)) if rotate_freqs else freq_list[0]

    use_spatial = (
        args.preset in ("realistic", "stress")
        and not args.no_spatial_ssvep
    )
    gains = _spatial_gains(args.channels, args.ssvep_channel, use_spatial)

    emg_rate_per_s = max(args.emg_spikes_per_min / 60.0, 0.0)
    pop_rate_per_s = max(args.pop_rate_per_min / 60.0, 0.0)
    blink_rate_per_s = max(args.blink_rate_per_min / 60.0, 0.0)
    blink_spatial = ssm.occipital_spatial_blink_weights(args.channels)
    eog_spatial = ssm.occipital_spatial_eog_weights(args.channels)

    stream_state: StreamState | None = None
    if (
        args.line_noise_uv > 0
        or args.common_mode_uv > 0
        or args.alpha_occipital_uv > 0
        or args.drift_uv_per_s > 0
        or emg_rate_per_s > 0
        or pop_rate_per_s > 0
        or blink_rate_per_s > 0
        or args.eog_ramp_peak_uv > 0
    ):
        stream_state = _init_stream_state(
            args.channels,
            rng,
            emg_rate_per_s,
            pop_rate_per_s,
            blink_rate_per_s,
        )

    freq_desc = (
        f"random among {freq_list} every {args.frequency_switch_s:g}s"
        if rotate_freqs
        else f"{current_f:g} Hz"
    )
    print(
        f"Streaming '{args.stream_name}' "
        f"({args.channels} ch @ {int(fs)} Hz, preset={args.preset}, {freq_desc})"
    )
    print(
        f"  SSVEP amp={args.amplitude_uv} uV (fundamental on Oz) | pink={args.noise_uv} uV RMS | "
        f"spatial={'on' if use_spatial else 'off'}"
    )
    if args.preset in ("realistic", "stress") or any(
        getattr(args, k) > 0
        for k in (
            "line_noise_uv",
            "common_mode_uv",
            "alpha_occipital_uv",
            "drift_uv_per_s",
            "emg_spikes_per_min",
            "adc_clip_uv",
            "pop_rate_per_min",
            "blink_rate_per_min",
            "eog_ramp_peak_uv",
            "packet_loss_prob_chunk",
        )
    ):
        print(
            f"  line={args.line_noise_uv:g} uV @ {args.mains_hz:g} Hz | "
            f"common_mode={args.common_mode_uv:g} uV | alpha_occ={args.alpha_occipital_uv:g} uV | "
            f"drift={args.drift_uv_per_s:g} uV/s | EMG~{args.emg_spikes_per_min:g}/min"
        )
        print(
            f"  artefacts: adc_clip={'off' if args.adc_clip_uv <= 0 else f'±{args.adc_clip_uv:g} µV'} | "
            f"pops~{args.pop_rate_per_min:g}/min @ {args.pop_step_uv:g} µV step | "
            f"blinks~{args.blink_rate_per_min:g}/min ({args.blink_duration_ms:g} ms, "
            f"{args.blink_amplitude_uv:g} µV pk) | "
            f"eog_ramp={'off' if args.eog_ramp_peak_uv <= 0 else f'{args.eog_ramp_peak_uv:g} µV @ {args.eog_ramp_hz:g} Hz'} | "
            f"packet_gap P={args.packet_loss_prob_chunk:g} mean={args.packet_loss_mean_gap_ms:g} ms"
        )
    print(
        f"  duration={'inf' if args.duration <= 0 else f'{args.duration:.2f}s'} | "
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
                    f"{prev:g} Hz -> {current_f:g} Hz",
                    flush=True,
                )

            # Application-level packet gap: no push, wall-clock jump, phase still advances.
            if args.packet_loss_prob_chunk > 0 and rng.random() < args.packet_loss_prob_chunk:
                gap_s = rng.exponential(max(args.packet_loss_mean_gap_ms, 1e-6) / 1000.0)
                time.sleep(gap_s)
                n_emitted += chunk
                next_emit = time.monotonic()
                continue

            t_idx = np.arange(n_emitted, n_emitted + chunk) / fs
            ssvep = np.zeros(chunk, dtype=np.float64)
            for h in range(1, args.harmonics + 2):
                amp = args.amplitude_uv / h
                ssvep += amp * np.sin(2 * np.pi * h * current_f * t_idx)

            data = (pink_noise(chunk, args.channels, rng) * args.noise_uv).astype(
                np.float64
            )

            # Spatially weighted SSVEP on all channels
            for ch in range(args.channels):
                data[ch] += gains[ch] * ssvep

            if stream_state is not None:
                # Mains pickup (per-channel phase)
                if args.line_noise_uv > 0:
                    for ch in range(args.channels):
                        ph = stream_state.line_phase[ch]
                        data[ch] += args.line_noise_uv * np.sin(
                            2 * np.pi * args.mains_hz * t_idx + ph
                        )

                # Common-mode reference noise
                if args.common_mode_uv > 0:
                    cm = pink_noise(chunk, 1, rng).ravel().astype(np.float64)
                    cm -= cm.mean()
                    s = cm.std()
                    if s > 1e-9:
                        cm /= s
                    data += args.common_mode_uv * cm

                # Shared alpha band on occipital cluster (channels 0–2)
                if args.alpha_occipital_uv > 0 and args.channels >= 3:
                    alpha = _bandpass_alpha(chunk, fs, rng)
                    occ_gains = np.array([1.0, 0.88, 0.88] + [0.0] * max(0, args.channels - 3))
                    for ch in range(min(3, args.channels)):
                        data[ch] += args.alpha_occipital_uv * occ_gains[ch] * alpha

                # Slow random-walk baseline per channel
                if args.drift_uv_per_s > 0:
                    step_std = args.drift_uv_per_s * np.sqrt(chunk / fs)
                    stream_state.drift += rng.normal(0.0, step_std, size=args.channels)
                    data += stream_state.drift[:, np.newaxis]

                # Rare EMG-like spikes (rectangular, few ms)
                if emg_rate_per_s > 0:
                    stream_state.emg_timer -= chunk_dt
                    if stream_state.emg_timer <= 0:
                        stream_state.emg_timer = ssm.poisson_next_interval(emg_rate_per_s, rng)
                        ch = int(rng.integers(0, args.channels))
                        width = max(2, int(0.004 * fs))  # ~4 ms
                        start = int(rng.integers(0, max(1, chunk - width)))
                        sign = 1.0 if rng.random() > 0.5 else -1.0
                        data[ch, start : start + width] += sign * args.emg_spike_uv

                if pop_rate_per_s > 0:
                    stream_state.pop_timer -= chunk_dt
                    while stream_state.pop_timer <= 0:
                        stream_state.pop_timer += ssm.poisson_next_interval(pop_rate_per_s, rng)
                        ch = int(rng.integers(0, args.channels))
                        start = int(rng.integers(0, max(1, chunk)))
                        sign = 1.0 if rng.random() > 0.5 else -1.0
                        ssm.inject_pop_step_decay(
                            data, ch, start, sign, args.pop_step_uv, fs
                        )

                if blink_rate_per_s > 0:
                    if stream_state.blink_samples_left == 0:
                        stream_state.blink_arm_timer -= chunk_dt
                        if stream_state.blink_arm_timer <= 0:
                            stream_state.blink_profile = ssm.blink_waveform(
                                fs, args.blink_duration_ms / 1000.0
                            )
                            stream_state.blink_samples_left = int(stream_state.blink_profile.size)
                            stream_state.blink_cursor = 0
                            stream_state.blink_amplitude_uv = args.blink_amplitude_uv
                            stream_state.blink_arm_timer = ssm.poisson_next_interval(
                                blink_rate_per_s, rng
                            )
                    if stream_state.blink_samples_left > 0:
                        take = min(chunk, stream_state.blink_samples_left)
                        prof_chunk = np.zeros(chunk, dtype=np.float64)
                        c0 = stream_state.blink_cursor
                        prof_chunk[:take] = stream_state.blink_profile[c0 : c0 + take]
                        ssm.add_blink_occipital(
                            data,
                            prof_chunk,
                            stream_state.blink_amplitude_uv,
                            blink_spatial,
                        )
                        stream_state.blink_cursor += take
                        stream_state.blink_samples_left -= take

                if args.eog_ramp_peak_uv > 0:
                    ssm.eog_ramp_add(
                        data,
                        n_emitted,
                        fs,
                        args.eog_ramp_hz,
                        args.eog_ramp_peak_uv,
                        eog_spatial,
                        stream_state.eog,
                    )

            if args.adc_clip_uv > 0:
                data = ssm.apply_adc_clip(data, args.adc_clip_uv)

            outlet.push_chunk(data.astype(np.float32).T.tolist(), local_clock())
            n_emitted += chunk

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
