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

**Default Cyton channel order** (`configs/cyton_default.yaml` → `channel_labels`):

| Cyton CH | Label | Typical location |
|---|---|---|
| 1 | Oz | Primary occipital (midline) |
| 2 | O1 | Left occipital |
| 3 | O2 | Right occipital |
| 4 | POz | Midline parieto-occipital |
| 5 | PO3 | Left parieto-occipital |
| 6 | PO4 | Right parieto-occipital |
| 7 | Pz | Parietal midline |
| 8 | Cz | Central midline |

Edit `channel_labels` if your cap wiring order differs. Layer 2’s SNR gate uses
`snr_channel_index` (default `0` = first row, expected Oz).

### 19-site cap — which eight sites on Cyton N1P–N8P?

Cyton has **eight EEG inputs** (N1P…N8P) plus separate **GND** and **reference/bias**
pins. Your silkscreen lists many sites (e.g. O1, O2, P3, Pz, P4, …, FP1, FP2,
**GND**, **REF**) but there is **no Oz** — for SSVEP you still want a **posterior-heavy**
set on the eight signal pins.

**Recommended eight (signal → N1P…N8P in order):** **O1, O2, Pz, P3, P4, C3, C4,
T5** (or swap **T5** for **T6** if routing is easier). Put **GND** and **REF** on the
Cyton **ground / bias** inputs per the OpenBCI Cyton + cap guide (they are **not**
wired into N1P–N8P).

**Why:** O1/O2 carry most steady-state visual response when Oz is absent; Pz/P3/P4
give midline and lateral parietal context; C3/C4 add spatial spread for CAR/FBCCA;
T5 (or T6) keeps one more **posterior** row than frontal sites (F7–Fp2), which are
usually worse for flicker SNR.

**Paired configs in this repo** (labels + Layer 2 SNR/artefact rows already aligned):

```powershell
python -m layer1_acquisition --config configs/cyton_headset_19el_8ch_ssvep.yaml --serial-port COMx
python -m layer2_processing --config configs/layer2_headset_19el_8ch_ssvep.yaml
```

If you change the **physical** order on the Cyton header, edit **`channel_labels`**
in the Layer 1 YAML to match **CH1 = row 0**, then set **`snr_channel_index`** in
Layer 2 to the **0-based row** of your preferred SSVEP channel, and
**`artefact_channel_indices`** to the subset used for the peak-to-peak gate.

---

## Electrode preparation procedure

1. **Locate landmarks.** Mark Oz (inion + 10% of nasion–inion distance),
   O1/O2 (±10% lateral from Oz), and Pz (50% nasion–inion midline).
2. **Skin preparation.** Lightly abrade each site with NuPrep gel and a cotton
   swab. Clean residue with 70% isopropyl alcohol.
3. **Apply paste.** Fill the cup electrode with Ten20 paste (or equivalent).
   Press firmly for 5–10 seconds and secure with surgical tape or an EEG cap.
4. **Reference / ground.** Follow your OpenBCI wiring guide (e.g. SRB on the
   right earlobe for the default cap mapping in `cyton_default.yaml`, or mastoid
   if you prefer that montage).
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

To **record raw µV** for all channels to a CSV (same samples as LSL), either set
`raw_eeg_log_path` in YAML or use `--raw-eeg-log` (500 Hz × 8 ch grows quickly on
disk):

```bash
python -m layer1_acquisition --config configs/cyton_default.yaml --board synthetic --skip-impedance --raw-eeg-log recordings/run.csv
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `configs/cyton_default.yaml` | YAML configuration file |
| `--board` | _(from config)_ | Override board: `cyton` or `synthetic` |
| `--serial-port PORT` | _(from config)_ | Override serial port (e.g. `COM3`) |
| `--raw-eeg-log PATH` | off | Append raw multi-channel µV CSV (overrides YAML) |
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
| `channel_labels` | `[Oz, O1, O2, POz, PO3, PO4, Pz, Cz]` | Per Cyton CH1–8, LSL row order |
| `impedance_max_kohm` | `10.0` | Gate threshold in kΩ |
| `lsl_stream_name` | `BCI_RawEEG` | **Do not change** — downstream contract |
| `pull_interval_ms` | `10` | BrainFlow ring-buffer poll interval |
| `log_interval_s` | `5` | Health log cadence |
| `raw_eeg_log_path` | _(unset)_ | If set, append CSV: `sample_index`, `monotonic_s`, then one column per `channel_labels` row (float32 µV, same as LSL) |
| `raw_eeg_log_format` | `csv` | Only `csv` supported today |

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

1. **Notch filter(s)** — primary ``notch_freq_hz`` (default 60 Hz) plus optional
   ``additional_notch_freqs_hz`` (default includes 50 Hz so EU mains is not left
   inside the 5–45 Hz bandpass), Q = 35
2. **Bandpass filter** (5–45 Hz, 4th-order Butterworth)
3. **Common Average Reference (CAR)** — reduces common-mode noise
4. **Artefact gate** — drops epochs when peak-to-peak (after filters) on the
   evaluated channels exceeds ``artefact_threshold_uv``. Default YAML evaluates
   occipital/PO rows only (``artefact_channel_indices``) so Pz/Cz do not veto SSVEP.
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

# …or a richer synthetic (spatial SSVEP, mains, alpha, drift, EMG-like spikes)
python scripts/synthetic_ssvep_source.py --preset realistic --frequency 12.0

# Aggressive artefacts + packet gaps for stress-testing Layer 2
python scripts/synthetic_ssvep_source.py --preset stress --frequency 12.0

# Terminal 2 — run Layer 2 pointing at the test stream
python -m layer2_processing --stream-name BCI_RawEEG_Test

# Terminal 3 (optional) — watch both transports
python scripts/verify_layer2_output.py
```

#### Artifact simulation (synthetic SSVEP)

`scripts/synthetic_ssvep_source.py` can add **phenomenological** (not anatomically exact)
artefacts before LSL: ADC clipping, electrode pops, simplified occipital blinks,
EOG-like slow ramps, and **application-level** packet gaps (extra sleep with no
`push_chunk`, while sample indices still advance so the consumer sees missing wall-clock
data and a lower **measured** Hz). Use `--preset realistic` for mild defaults or
`--preset stress` for aggressive rates plus packet gaps. Override any default with flags
such as `--adc-clip-uv`, `--pop-rate-per-min`, `--blink-rate-per-min`,
`--eog-ramp-peak-uv`, `--packet-loss-prob-chunk`, and `--packet-loss-mean-gap-ms`
(`0` turns off clip / pops / blinks / EOG; packet loss uses probability `0`).

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

**Artefact gate:** if every epoch is rejected (`artefactual` equals `epochs` in
health logs), peak-to-peak on at least one **evaluated** channel is above
``artefact_threshold_uv``. Default YAML uses **1600 µV** (not clinical 100 µV)
because real Cyton streams can show **~1–3 mV** peak-to-peak on Oz after
filters when contacts or reference are noisy. Indices ``[0, 1, 2, 3, 4, 5]``
(Oz through PO4) exclude Pz/Cz from
the veto. Tighten ``artefact_threshold_uv`` once traces are calmer. Health lines
include ``last_max_ptp`` and ``eval_max``.

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

---

## Layer 3 + Layer 4 MVP — Backend Bridge & Browser Frontend

### What it does

A thin FastAPI server that:

1. Subscribes to Layer 2's WebSocket on `:9001` (the bridge).
2. Re-broadcasts every `SELECT` payload to all connected browser clients.
3. Serves a single-page SSVEP frontend with two flickering tiles (12 Hz,
   15 Hz) and a live status panel showing detected frequency, SNR, and
   confidence.

This is the **MVP** — no spelling, no navigation, no AI predictor.  The
`AbstractClassifier` interface and the `stimulus_frequencies_hz` config field
exist so full speller pages can be added later without restructuring.

### Additional dependencies

| Package | Purpose |
|---|---|
| `fastapi~=0.115` | HTTP + WebSocket server |
| `uvicorn~=0.30` | ASGI server to host FastAPI |

### Running the full stack (no Cyton)

```powershell
# Terminal 1 — synthetic 12 Hz SSVEP on BCI_RawEEG_Test
python scripts/synthetic_ssvep_source.py --frequency 12.0

# Terminal 2 — Layer 2 pipeline
python -m layer2_processing --stream-name BCI_RawEEG_Test

# Terminal 3 — Layer 3 backend + frontend
python -m layer3_backend

# Browser — open http://localhost:8000
```

You should see two large flickering tiles (12 Hz and 15 Hz).  The status
panel updates ~2x/sec with the detected frequency, SNR, and confidence.
The tile matching the detected frequency briefly highlights on each emission.

### Layer 3 CLI options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `configs/layer3_default.yaml` | YAML configuration file |
| `--host HOST` | `localhost` | Bind host |
| `--port PORT` | `8000` | Bind port |
| `--layer2-ws URL` | `ws://localhost:9001` | Layer 2 WebSocket URL |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the SSVEP flicker page (`index.html`) |
| `/health` | GET | Returns `{"ok": true}` |
| `/config` | GET | Returns `{"stimulus_frequencies_hz": [12.0, 15.0]}` |
| `/ws` | WS | Browser clients connect here; receives re-broadcast SELECT payloads |

### Frequency note (60 Hz laptop)

The MVP uses `[12.0, 15.0]` Hz — both are exact divisors of 60 Hz (5-frame
and 4-frame periods respectively).  When moving to the Meta Quest 3 (90 Hz),
widen the list to `[9, 10, 12, 15, 18, 30]` in both `configs/layer2_default.yaml`
and `configs/layer3_default.yaml`.

### Layer 3 tests

```powershell
python -m pytest tests/test_layer3_config.py tests/test_layer3_bridge.py \
                 tests/test_layer3_server.py -v
```

| Test module | What it covers |
|---|---|
| `test_layer3_config.py` | YAML loading, validation, overrides |
| `test_layer3_bridge.py` | Broadcaster, bridge receive + reconnect, clean shutdown |
| `test_layer3_server.py` | `/`, `/health`, `/config` endpoints via httpx |
