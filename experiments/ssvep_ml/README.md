# Supervised SSVEP ML (6 Hz vs 15 Hz)

Offline pipeline from the architecture plan: **joint recording** → **epochs.npz** → **train / eval** → **inference wrapper**.

## Install

```bash
pip install -e ".[dev,experiments]"
```

`experiments` adds **pygame** (stimulus), **torch** (optional CNN), **matplotlib** (optional plots).

## Protocol and montage

See [PROTOCOL.md](PROTOCOL.md) and `session_template/session_meta.yaml.example`.

## 1) Record (aligned sample indices)

Single-process acquisition + fullscreen flicker (recommended):

```bash
python -m experiments.ssvep_ml.joint_session --out-dir data/ssvep/s01 ^
  --config configs/cyton_default.yaml --board synthetic --skip-impedance ^
  --trials-per-class 10 --copy-meta-template
```

For Cyton hardware, drop `--board synthetic` and set `--serial-port COMx`. Edit `session_meta.yaml` in the output folder.

**Alternative — LSL markers** (Layer 1 in one process, markers + flicker here; record with LabRecorder to XDF):

```bash
python -m experiments.ssvep_ml.stimulus_lsl_markers --stream-name SSVEP_Markers
```

## 2) Build epochs

```bash
python -m experiments.ssvep_ml.build_dataset ^
  --eeg-csv data/ssvep/s01/raw_eeg.csv ^
  --events-jsonl data/ssvep/s01/events.jsonl ^
  --output data/ssvep/s01/epochs.npz ^
  --session-id s01
```

Flags: `--onset-delay-s` (default 0.75), `--window-s` (default 1.5), `--no-preprocess` for raw µV only.

## 3) Pilot QC (PSD peak check)

```bash
python -m experiments.ssvep_ml.pilot_qc --epochs-npz data/ssvep/s01/epochs.npz
```

## 4) Train

**Sklearn band-power baseline** (no GPU):

```bash
python -m experiments.ssvep_ml.train --npz data/ssvep/s01/epochs.npz --output-dir models/run1 --backend sklearn
```

**Session-wise split** (after multiple sessions):

```bash
python -m experiments.ssvep_ml.train --npz data/ssvep/s01/epochs.npz data/ssvep/s02/epochs.npz ^
  --output-dir models/run2 --val-session s01 --test-session s02 --backend sklearn
```

**CNN** (requires torch):

```bash
python -m experiments.ssvep_ml.train --npz data/ssvep/s01/epochs.npz --output-dir models/cnn1 --backend cnn
```

Artifacts:

- `manifest.json` — metrics, model type, class names
- `sklearn_model.joblib` — baseline pipeline
- `cnn_weights.pt` + `cnn_config.json` — CNN bundle

## 5) Evaluate

```bash
python -m experiments.ssvep_ml.eval --model-dir models/run1 --epochs-npz data/ssvep/s02/epochs.npz
```

## 6) Inference wrapper (export notes)

From Python:

```python
from experiments.ssvep_ml.inference_wrapper import load_bundle, predict_proba
import numpy as np

bundle = load_bundle("models/run1")
epoch = np.random.randn(8, 750).astype(np.float32)  # example shape
probs = predict_proba(bundle, epoch)  # (2,) for 6 Hz vs 15 Hz
```

For production BCI, re-use the same preprocessing as training (`Preprocessor` + same window length), then map argmax to 6 / 15 Hz commands. TorchScript export is not generated automatically; add `torch.jit.trace` around `TinyEegCNN` if you need a standalone binary.
