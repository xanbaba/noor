"""Standalone LSL stream verifier for BCI_RawEEG.

Usage:
    python scripts/verify_lsl_stream.py [--stream-name BCI_RawEEG] [--duration 10]

Resolves the named LSL outlet, pulls samples for the given duration, and prints:
  - Channel count and metadata (labels, units)
  - Measured sample rate vs. nominal rate
  - Per-window timestamp jitter (µs)
  - Estimated dropped sample count

Run this after starting layer1_acquisition to confirm the stream is healthy
before connecting Layer 2.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import numpy as np

try:
    from pylsl import StreamInlet, resolve_byprop, local_clock
except ImportError:
    print("ERROR: pylsl is not installed.  Run: pip install pylsl", file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify the BCI_RawEEG LSL stream health."
    )
    p.add_argument(
        "--stream-name",
        default="BCI_RawEEG",
        help="LSL stream name to resolve (default: BCI_RawEEG).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How many seconds to monitor the stream (default: 10).",
    )
    p.add_argument(
        "--resolve-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the stream to appear (default: 5).",
    )
    return p


def _channel_labels(inlet: StreamInlet) -> list[str]:
    info = inlet.info()
    channels = info.desc().child("channels")
    labels: list[str] = []
    ch = channels.child("channel")
    while ch.name() == "channel":
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


def main() -> int:
    args = _build_parser().parse_args()

    print(f"Resolving LSL stream '{args.stream_name}' (timeout {args.resolve_timeout}s)…")
    streams = resolve_byprop("name", args.stream_name, timeout=args.resolve_timeout)
    if not streams:
        print(f"ERROR: No stream named '{args.stream_name}' found.", file=sys.stderr)
        return 1

    info = streams[0]
    print(
        f"\nStream found!\n"
        f"  name         : {info.name()}\n"
        f"  type         : {info.type()}\n"
        f"  channels     : {info.channel_count()}\n"
        f"  nominal rate : {info.nominal_srate()} Hz\n"
        f"  source_id    : {info.source_id()}\n"
    )

    inlet = StreamInlet(info, max_buflen=30)
    labels = _channel_labels(inlet)
    if labels:
        print(f"  channel labels: {labels}\n")
    else:
        print("  (no channel label metadata found)\n")

    nominal_rate = info.nominal_srate()
    total_samples = 0
    timestamps: deque[float] = deque(maxlen=500)
    jitter_us_list: list[float] = []

    print(f"Monitoring for {args.duration:.0f} seconds… (Ctrl+C to stop early)\n")
    t_start = time.monotonic()
    t_last_report = t_start
    last_ts: float | None = None

    try:
        while (elapsed := time.monotonic() - t_start) < args.duration:
            chunk, ts_list = inlet.pull_chunk(timeout=0.05)
            if not chunk:
                continue

            n = len(chunk)
            total_samples += n

            for ts in ts_list:
                timestamps.append(ts)
                if last_ts is not None:
                    gap_us = (ts - last_ts) * 1e6
                    expected_gap_us = 1e6 / nominal_rate
                    jitter_us = abs(gap_us - expected_gap_us)
                    jitter_us_list.append(jitter_us)
                last_ts = ts

            # Print a rolling report every 2 seconds.
            if time.monotonic() - t_last_report >= 2.0:
                measured_rate = total_samples / elapsed if elapsed > 0 else 0.0
                mean_jitter = float(np.mean(jitter_us_list[-200:])) if jitter_us_list else 0.0
                p95_jitter = float(np.percentile(jitter_us_list[-200:], 95)) if jitter_us_list else 0.0
                print(
                    f"  t={elapsed:5.1f}s | "
                    f"total={total_samples:6d} samples | "
                    f"rate={measured_rate:6.1f} Hz | "
                    f"jitter mean={mean_jitter:.1f} µs  p95={p95_jitter:.1f} µs"
                )
                t_last_report = time.monotonic()

    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t_start
    measured_rate = total_samples / elapsed if elapsed > 0 else 0.0
    rate_error_pct = abs(measured_rate - nominal_rate) / nominal_rate * 100

    mean_jitter = float(np.mean(jitter_us_list)) if jitter_us_list else 0.0
    p95_jitter = float(np.percentile(jitter_us_list, 95)) if jitter_us_list else 0.0

    expected_total = int(elapsed * nominal_rate)
    estimated_drops = max(0, expected_total - total_samples)

    print(f"\n{'─'*60}")
    print("Summary")
    print(f"{'─'*60}")
    print(f"  Duration          : {elapsed:.2f} s")
    print(f"  Samples received  : {total_samples}")
    print(f"  Measured rate     : {measured_rate:.2f} Hz  (nominal {nominal_rate:.0f} Hz, error {rate_error_pct:.2f}%)")
    print(f"  Timestamp jitter  : mean={mean_jitter:.1f} µs, p95={p95_jitter:.1f} µs")
    print(f"  Estimated drops   : {estimated_drops} samples")

    ok = rate_error_pct < 5.0 and p95_jitter < 2000.0 and estimated_drops == 0
    print(f"\n  Result: {'PASS' if ok else 'WARN — check hardware or driver'}")
    print(f"{'─'*60}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
