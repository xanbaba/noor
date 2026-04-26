# SSVEP recording protocol — 6 Hz vs 15 Hz (single participant v1)

## Purpose

Collect raw EEG aligned to trial boundaries so a supervised classifier can learn **6 Hz** vs **15 Hz** steady-state visually evoked responses from one team member.

## Equipment and montage

- OpenBCI Cyton (8-channel), 500 Hz, same YAML as production (`configs/cyton_default.yaml` or your cap-specific file).
- Impedance under 10 kΩ on every active channel; repeat check if a channel looks noisy.
- Same chair distance to monitor (record viewing distance in `session_meta.yaml`).
- **Monitor:** document refresh rate (e.g. 60 Hz) and flicker implementation (square-wave inversion timing). Disable variable refresh / gaming overlays for the session.
- Same lighting and approximate time of day across sessions when possible.

## Session schedule

- Aim for **3–5 short sessions** (15–20 min) on **different days** for session-wise train/val/test splits.
- Avoid caffeine withdrawal extremes; note sleep hours in `session_meta.yaml`.

## Trial timing (defaults in `joint_session.py`)

| Phase | Duration | Notes |
|--------|----------|--------|
| Warm-up (no flicker) | 3 s | Acquisition already running; discard in analysis |
| Inter-trial interval (ITI) | 2.5 s | Mid-grey screen, fixation cross optional |
| Instruction | 1.0 s | Text: “Look at the flicker” |
| Stimulus | 6.0 s | Full-screen flicker at **one** frequency only |
| **Total per trial** | ~9.5 s + ITI | |

## Counterbalancing

- Trial order is **pseudorandom** with equal counts per class (`6hz`, `15hz`).
- No more than **4 consecutive** trials of the same class (enforced in scheduler).

## Data files per session (output directory)

| File | Description |
|------|-------------|
| `raw_eeg.csv` | Layer 1 format: `sample_index`, `monotonic_s`, channel columns |
| `events.jsonl` | One JSON object per line; `stim_on` / `stim_off` with `sample_index` |
| `session_meta.yaml` | Participant id, notes, montage, monitor Hz |
| `protocol_version` | Text marker for this document |

## Synchronisation

Use **`python -m experiments.ssvep_ml.joint_session`** so acquisition and stimulus share one process: `sample_index` at `stim_on` / `stim_off` matches rows in `raw_eeg.csv`.

## Ethics

Obtain informed consent from the participant; do not commit identifying information in the repository.
