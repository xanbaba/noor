# SSVEP-BCI — Layers 1 & 2: Data Acquisition + Signal Processing

Layer 1 streams raw EEG from an OpenBCI Cyton (8-channel, 500 Hz) onto a Lab
Streaming Layer (LSL) outlet named `BCI_RawEEG`.  Layer 2 consumes that stream
in real time, classifies the user's gazed stimulus frequency with FBCCA, and
emits `SELECT` commands over WebSocket and OSC to any connected consumer (Layer 4
Quest 3 frontend, game engine, etc.).

---

## Hardware requirements

| Item | Specification |
|---|---|
| EEG amplifier | OpenBCI Cyton (8-channel) + USB dongle |
| Electrodes | Ag/AgCl cup electrodes or active dry electrodes |
| Conductive medium | Ten20 paste or equivalent (impedance < 10 kΩ required) |
| Ground | Right mastoid (A2) |

### Electrode placement (10-20 system)

```
       Fpz
      /   \
   Fp1     Fp2
    |         |
   F3   Fz   F4
    \    |   /
     F7      F8
      \      /
  T3 — C3 — Cz — C4 — T4
      /      \
     T5        T6
      \        /
       P3—Pz—P4
        \    /
    O1 — Oz — O2   ← Channel cluster for SSVEP
```

**Active channels (ch 1–4):**

| Channel | Label | Location |
|---|---|---|
| 1 | Oz | Primary occipital (midline) |
| 2 | O1 | Left occipital |
| 3 | O2 | Right occipital |
| 4 | Pz | Parietal reference |

Channels 5–8 are unused (`--`) but still streamed to maintain the fixed
8-channel LSL contract.

---

## Electrode preparation procedure

1. **Locate landmarks.** Mark Oz (inion + 10% of nasion–inion distance),
   O1/O2 (±10% lateral from Oz), and Pz (50% nasion–inion midline).
2. **Skin preparation.** Lightly abrade each site with NuPrep gel and a cotton
   swab. Clean residue with 70% isopropyl alcohol.
3. **Apply paste.** Fill the cup electrode with Ten20 paste (or equivalent).
   Press firmly for 5–10 seconds and secure with surgical tape or an EEG cap.
4. **Ground.** Place the Cyton's SRB2 ground lead on the right mastoid (A2)
   using a flat Ag/AgCl electrode and paste.
5. **Impedance check.** The software will automatically measure impedance at
   session start and block acquisition if any active channel exceeds 10 kΩ.
   Re-apply paste and recheck if any channel fails.

---

## Installation

Requires Python 3.11+ (tested with Python 3.12 on Windows 10/11).

```bash
pip install -e ".[dev]"
```

This installs:
- `brainflow~=5.14` — BrainFlow EEG driver
- `pylsl~=1.16` — Lab Streaming Layer
- `pyyaml~=6.0` — configuration file parsing
- `numpy~=1.26` — numerical arrays
- `pytest~=8.0`, `pytest-asyncio~=0.23` — test dependencies

---

## Running

### With the Cyton (real hardware)

1. Plug in the OpenBCI USB dongle.
2. Note the assigned COM port (Device Manager → Ports on Windows).
3. Prepare electrodes and verify impedance as described above.
4. Start the acquisition:

```bash
python -m layer1_acquisition --config configs/cyton_default.yaml --serial-port COM3
```

Replace `COM3` with your actual port. On Linux/macOS use `/dev/ttyUSB0` or similar.

### Bench testing (no Cyton)

Use the built-in BrainFlow synthetic board to run without hardware:

```bash
python -m layer1_acquisition --config configs/cyton_default.yaml --board synthetic --skip-impedance
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `configs/cyton_default.yaml` | YAML configuration file |
| `--board` | _(from config)_ | Override board: `cyton` or `synthetic` |
| `--serial-port PORT` | _(from config)_ | Override serial port (e.g. `COM3`) |
| `--skip-impedance` | off | Bypass impedance gate (bench/dev only) |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Verifying the stream

While acquisition is running, open a second terminal:

```bash
python scripts/verify_lsl_stream.py --duration 15
```

This resolves the `BCI_RawEEG` LSL stream, monitors it for 15 seconds, and prints:

- Measured sample rate vs. nominal 500 Hz
- Timestamp jitter (mean and 95th percentile)
- Estimated dropped sample count

A healthy stream will show:
- Rate within ±5% of 500 Hz
- Jitter p95 < 2 ms
- Zero dropped samples

---

## Configuration

Edit `configs/cyton_default.yaml` to adjust:

| Key | Default | Description |
|---|---|---|
| `board` | `cyton` | Board driver: `cyton` or `synthetic` |
| `serial_port` | `auto` | COM port, or `auto` to detect |
| `sample_rate_hz` | `500` | Must be ≥ 250 Hz (500 recommended) |
| `channel_labels` | `[Oz, O1, O2, Pz, --, ...]` | Per-channel electrode label |
| `impedance_max_kohm` | `10.0` | Gate threshold in kΩ |
| `lsl_stream_name` | `BCI_RawEEG` | **Do not change** — downstream contract |
| `pull_interval_ms` | `10` | BrainFlow ring-buffer poll interval |
| `log_interval_s` | `5` | Health log cadence |

---

## Running tests

```bash
python -m pytest tests/ -v
```

The test suite (23 tests) covers:

| Module | Tests |
|---|---|
| `test_config.py` | YAML loading, validation, overrides |
| `test_factory.py` | Board factory dispatch |
| `test_lsl_outlet.py` | LSL stream metadata, push behaviour |
| `test_impedance.py` | Gate pass/fail/skip logic |
| `test_acquisition_synthetic.py` | End-to-end: synthetic board → LSL inlet |

All tests run without any hardware attached.

---

## LSL stream contract (frozen)

Downstream layers depend on these fields being stable:

| Field | Value |
|---|---|
| `name` | `BCI_RawEEG` |
| `type` | `EEG` |
| `channel_count` | `8` |
| `nominal_srate` | `500` Hz |
| `channel_format` | `float32` |
| `channel labels` | per YAML (Oz, O1, O2, Pz, --, --, --, --) |
| `channel unit` | `microvolts` |

To extend to Cyton+Daisy (16 channels) or a different amplifier: implement a new `AbstractBoard` subclass in `layer1_acquisition/boards/`, register it in `factory.py`, and update the config. The stream name, type, and format remain unchanged.

---

## Project structure

```
neuron/
├── configs/
│   ├── cyton_default.yaml          # Layer 1 — Cyton board configuration
│   └── layer2_default.yaml         # Layer 2 — signal processing configuration
├── layer1_acquisition/
│   ├── __init__.py
│   ├── __main__.py                 # CLI entry point
│   ├── acquisition.py              # Main acquisition loop
│   ├── config.py                   # YAML loading + AcquisitionConfig
│   ├── impedance.py                # Impedance gate
│   ├── logging_config.py           # Structured logging
│   ├── lsl_outlet.py               # pylsl StreamOutlet wrapper
│   └── boards/
│       ├── base.py                 # AbstractBoard interface
│       ├── factory.py              # Board name → class
│       ├── cyton.py                # OpenBCI Cyton driver
│       └── synthetic.py            # BrainFlow synthetic (CI/no-hardware)
├── layer2_processing/
│   ├── __init__.py
│   ├── __main__.py                 # CLI entry point
│   ├── config.py                   # YAML loading + ProcessingConfig
│   ├── logging_config.py           # Re-exports Layer 1 logging helpers
│   ├── lsl_inlet.py                # pylsl StreamInlet wrapper
│   ├── preprocessing.py            # Preprocessor + EpochBuffer
│   ├── pipeline.py                 # Main pipeline loop
│   ├── snr.py                      # Sum-of-harmonics SNR estimator
│   ├── classifiers/
│   │   ├── base.py                 # AbstractClassifier interface
│   │   ├── fbcca.py                # FBCCAClassifier (Chebyshev sub-bands + CCA)
│   │   └── factory.py              # Classifier name → class dispatch
│   └── outputs/
│       ├── websocket_emitter.py    # asyncio WebSocket broadcast (:9001)
│       └── osc_emitter.py          # python-osc UDP emitter (/bci/command)
├── scripts/
│   ├── verify_lsl_stream.py        # Layer 1 stream health checker
│   ├── synthetic_ssvep_source.py   # Synthetic SSVEP LSL outlet (no Cyton needed)
│   └── verify_layer2_output.py     # WS + OSC consumer — prints every SELECT
├── tests/
│   ├── test_config.py
│   ├── test_factory.py
│   ├── test_impedance.py
│   ├── test_lsl_outlet.py
│   ├── test_acquisition_synthetic.py
│   ├── test_layer2_config.py
│   ├── test_preprocessing.py
│   ├── test_fbcca.py
│   ├── test_snr.py
│   ├── test_outputs.py
│   └── test_layer2_e2e.py
├── pyproject.toml
└── ARCHITECTURE.md
```

---

## Layer 2 — Signal Processing & Classification

### What it does

Layer 2 is a standalone Python process that connects to the `BCI_RawEEG` LSL
stream produced by Layer 1, processes each sliding 2-second epoch through:

1. **Notch filter** (60 Hz mains removal, Q = 35)
2. **Bandpass filter** (5–45 Hz, 4th-order Butterworth)
3. **Common Average Reference (CAR)** — reduces common-mode noise
4. **Artefact gate** — drops epochs where any channel exceeds ±100 µV peak-to-peak
5. **FBCCA classifier** — calibration-free frequency detection using 4 Chebyshev
   sub-bands and sklearn CCA
6. **SNR gate** — only emits if signal-to-noise ratio ≥ 3.5 dB on Oz

…and broadcasts a `SELECT` command over **WebSocket** and **OSC**.

### Additional dependencies (installed automatically via `pyproject.toml`)

| Package | Purpose |
|---|---|
| `scipy~=1.13` | IIR filters (`sosfiltfilt`, `butter`, `cheby1`) |
| `scikit-learn~=1.4` | `CCA` for FBCCA |
| `mne~=1.7` | (available for future spatial filtering) |
| `python-osc~=1.8` | UDP OSC emitter |
| `websockets~=12.0` | asyncio WebSocket server |

### Running Layer 2

**With a live Cyton (Layer 1 must be running first):**

```powershell
# Terminal 1 — start Layer 1
python -m layer1_acquisition --serial-port COM5

# Terminal 2 — start Layer 2
python -m layer2_processing
```

**With the synthetic SSVEP source (no Cyton required):**

```powershell
# Terminal 1 — emit a 12 Hz SSVEP sine on BCI_RawEEG_Test
python scripts/synthetic_ssvep_source.py --frequency 12.0

# Terminal 2 — run Layer 2 pointing at the test stream
python -m layer2_processing --stream-name BCI_RawEEG_Test

# Terminal 3 (optional) — watch both transports
python scripts/verify_layer2_output.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `configs/layer2_default.yaml` | YAML configuration file |
| `--classifier NAME` | from config | Override classifier (`fbcca`) |
| `--stream-name NAME` | from config | Override LSL stream name |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

### Output contract (frozen)

Both WebSocket (`ws://localhost:9001`, broadcast) and OSC (`/bci/command` UDP
to `127.0.0.1:9000`) emit identical JSON payloads:

```json
{"command":"SELECT","frequency":12.0,"snr_db":4.1,"confidence":0.87,"epoch_ms":2000}
```

| Field | Type | Description |
|---|---|---|
| `command` | string | Always `"SELECT"` |
| `frequency` | float | Detected stimulus frequency in Hz |
| `snr_db` | float | Signal-to-noise ratio in dB on Oz channel |
| `confidence` | float | Softmax-normalised classifier score in [0, 1] |
| `epoch_ms` | int | Epoch length in milliseconds (e.g. 2000) |

Consumers connect to `ws://localhost:9001` (WebSocket) or bind a UDP listener
on port 9000 (OSC) — no authentication or TLS required for local deployments.

### Configuration reference (`configs/layer2_default.yaml`)

All stimulus frequencies are fully configuration-driven — there are no
hard-coded values in the classifier code:

```yaml
stimulus_frequencies_hz: [9.0, 10.0, 12.0, 15.0, 18.0, 30.0]
```

Layer 3 (stimulus rendering) must publish the same list.  All other parameters
(sub-band cutoffs, SNR threshold, epoch length, output ports) can be changed
without touching any Python source.

### Running the tests

```powershell
# Unit + integration + E2E (all headless — no Cyton or Quest required)
python -m pytest

# Layer 2 tests only
python -m pytest tests/test_layer2_config.py tests/test_preprocessing.py \
                 tests/test_fbcca.py tests/test_snr.py tests/test_outputs.py \
                 tests/test_layer2_e2e.py -v
```

| Test module | What it covers |
|---|---|
| `test_layer2_config.py` | YAML loading, validation, overrides |
| `test_preprocessing.py` | Notch/bandpass attenuation, CAR, artefact gate, epoch buffer |
| `test_fbcca.py` | Frequency detection, confidence, factory, custom frequency list |
| `test_snr.py` | Pure-sine → high SNR; noise → low SNR; channel selection |
| `test_outputs.py` | WebSocket broadcast; OSC UDP delivery; JSON payload |
| `test_layer2_e2e.py` | Synthetic SSVEP source → pipeline → correct SELECT emitted |
