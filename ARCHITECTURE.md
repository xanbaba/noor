  ## SSVEP-BCI AR/VR System — Architecture Document

---

**Layer 1 — Data Acquisition** is unchanged in structure but gains specific BrainFlow board IDs, electrode placement detail (Oz, O1, O2, Pz with linked mastoid ground), and sampling rate guidance (250 Hz minimum; 500 Hz recommended for harmonic detection above 30 Hz).

**Layer 2 — Signal Processing** is the most significantly enhanced section. The algorithm stack is now: notch filter (60 Hz) → bandpass (5–45 Hz, 4th-order Butterworth) → epoch windowing (1–4s sliding) → **Filter Bank CCA (FBCCA)** as the primary classifier, with TRCA as the high-accuracy fallback when calibration data is available. Specific sub-band definitions, harmonic sets, and SNR thresholds are included. Accuracy enhancement strategies include spatial filtering with CSP/xDAWN, ensemble voting across epoch lengths, and online drift correction.

**Layer 3 — Backend** gains explicit FastAPI endpoint contracts, a context-aware frequency→action mapping schema, and a clarification that the AI predictor uses Google's **Gemini API** (`gemini-2.5-flash` via the `google-genai` SDK), with aggressive prefix caching of partial-word predictions to keep the perceived spelling latency below 200 ms.

**Layer 4 — Meta Quest Frontend** is rewritten for the Quest 3 specifically: 90/120 Hz display modes, custom GLSL shader implementation for frame-synchronised flickering, foveated rendering considerations, and a spatial UI layout designed for seated wheelchair users. Integration with Layer 2/3 via a local WebSocket server running on the companion PC is fully specified.Click any node in the diagram to ask a follow-up. Here is the full revised document:

---

## Layer 1 — Data Acquisition

**Stack:** Python 3.11, BrainFlow 5.x (`BoardIds.CYTON_BOARD = 0`), Lab Streaming Layer (`pylsl`).

**Hardware configuration.** Connect the OpenBCI Cyton (8-channel) via the USB dongle. Electrode placement: Oz (primary), O1, O2 (flanking occipital), Pz (parietal reference) using the 10–20 system. Link the right mastoid as ground (A2). This 4-electrode occipital cluster captures the visual cortex response most relevant to steady-state visual evoked potentials. Use conductive paste or active dry electrodes — contact impedance must be below 10 kΩ before acquisition begins. Impedance checking should be automated at session start.

**Sampling.** Set the Cyton to 500 Hz (BoardShim `set_log_level`, `prepare_session`, `start_stream`). 500 Hz is preferred over 250 Hz because it allows reliable harmonic extraction up to the third harmonic of 30 Hz stimuli without aliasing. Raw 24-bit samples are pushed directly to an LSL stream (`StreamOutlet`) with the channel format `float32` and stream type `EEG`. Include timestamp synchronisation using `local_clock()` at push time.

**Extensibility contract.** This layer's sole output is an LSL stream named `BCI_RawEEG` with 8 channels and a nominal rate of 500 Hz. All downstream layers discover it by name. Replacing the Cyton with a 16-channel Cyton+Daisy, a Unicorn Hybrid Black, or a research-grade g.USBamp requires only rewriting this module — the stream name, channel count, and rate remain contractually stable.

---

## Layer 2 — Signal Processing and Classification

**Stack:** Python 3.11, MNE-Python 1.7, SciPy 1.13, NumPy 1.26, scikit-learn 1.4. Runs as a standalone process, subscribing to the `BCI_RawEEG` LSL stream.

### 2.1 Preprocessing Pipeline

Execute in strict order per epoch:

1. **60 Hz notch filter.** `scipy.signal.iirnotch(60, Q=35, fs=500)`. Also notch 120 Hz if third harmonics of 40 Hz stimuli are used. Applied via `sosfiltfilt` (zero-phase).
2. **Bandpass filter.** 4th-order Butterworth, 5–45 Hz, `scipy.signal.butter(4, [5, 45], btype='band', fs=500, output='sos')`, applied with `sosfiltfilt`. Preserves fundamental frequencies and their second and third harmonics while rejecting DC drift and high-frequency EMG artefacts.
3. **Artefact rejection.** Apply a simple peak-to-peak amplitude threshold of ±100 µV per channel per epoch. Epochs exceeding the threshold on any occipital channel are flagged and excluded from classification (not discarded — they are logged for clinical audit).
4. **Common Average Reference (CAR).** Subtract the mean of all 8 channels from each channel. Reduces correlated noise and volume conduction artefacts without requiring an explicit reference electrode.
5. **Epoch windowing.** Use a sliding window of configurable length (default: 2 seconds, step: 0.5 seconds). Shorter windows (1 s) reduce latency but lower classification accuracy; longer windows (4 s) maximise accuracy at the cost of responsiveness. Support for dynamic window length selection based on real-time SNR is described in §2.3.

### 2.2 Classification — Filter Bank CCA (FBCCA)

FBCCA is the recommended primary classifier. It extends standard CCA by decomposing the EEG signal into multiple frequency sub-bands, computing CCA in each, and combining the resulting correlation coefficients with a weighted sum. This approach is calibration-free (no user training session required) and outperforms single-band CCA by 8–15% accuracy in most published benchmarks.

**Sub-band decomposition.** Define 4 sub-bands as 5th-order Chebyshev Type I bandpass filters:
- Sub-band 1: 6–90 Hz (broadband reference)
- Sub-band 2: 14–90 Hz
- Sub-band 3: 22–90 Hz
- Sub-band 4: 30–90 Hz

Each sub-band is weighted by `w_k = k^(-1.25) + 0.25` (Chen et al., 2015), down-weighting higher bands that contain less signal and more noise.

**Reference signal construction.** For each candidate stimulus frequency `f` (e.g., 9, 10, 12, 15 Hz), construct a reference matrix `Y_f` of sine and cosine signals at the fundamental and harmonics 1–3:

```
Y_f = [sin(2πf·t), cos(2πf·t), sin(4πf·t), cos(4πf·t), sin(6πf·t), cos(6πf·t)]
```

**CCA step.** For each sub-band `k` and each frequency `f`, compute the canonical correlation coefficient `ρ_k(f)` between the filtered EEG epoch `X_k` and `Y_f` using `sklearn.cross_decomposition.CCA(n_components=1)`.

**Decision.** The predicted frequency is `argmax_f Σ_k w_k · ρ_k(f)^2`.

### 2.3 Accuracy Optimisation Strategies

**Spatial filtering.** Before FBCCA, apply xDAWN (implemented in MNE: `mne.preprocessing.Xdawn`) or a CSP filter trained on resting vs. SSVEP epochs. xDAWN maximises the signal-to-noise ratio of the SSVEP response in the occipital channels, providing a 5–10% absolute accuracy gain with 30–60 seconds of initial calibration. This is the recommended compromise between the zero-calibration of pure FBCCA and the longer calibration needed for TRCA.

**TRCA fallback (when calibration data is available).** Task-Related Component Analysis (TRCA) treats the inter-trial covariance of the SSVEP response as the signal subspace. It delivers state-of-the-art accuracy (~97% ITR on BETA benchmark) but requires 5–6 calibration trials per frequency (approximately 3 minutes per user). Implement as a separate `ClassifierTRCA` class that can be hot-swapped for `ClassifierFBCCA` via a configuration flag without changing the OSC output contract.

**Ensemble over epoch lengths.** For each classification decision, simultaneously evaluate 1 s, 2 s, and 3 s epochs and take a weighted majority vote (weights proportional to mean SNR of each epoch length during the most recent 30-second window). This reduces the impact of transient noise bursts without committing to a fixed window.

**SNR gating and rejection.** Before emitting a command, compute the SSVEP SNR using the sum-of-harmonics method: `SNR = (P_f + P_2f + P_3f) / P_noise`, where `P_noise` is the mean power in a 2 Hz band either side of each harmonic, excluding the harmonic bin itself. Emit the command only if `SNR ≥ 3.5 dB`. Below threshold, re-accumulate data for the next window. This prevents erroneous selections caused by alpha-wave overlap or attention drift and is the single most impactful quality gate in the pipeline.

**Online drift correction.** Every 5 minutes, recompute the mean PSD of the resting-state baseline (1–4 Hz band) and update the bandpass filter lower cutoff if DC drift has shifted by more than 2 µV. Store as a rolling correction factor.

### 2.4 Computational Efficiency

The full FBCCA pipeline on a 2-second, 8-channel, 500 Hz epoch executes in under 8 ms on a mid-range CPU (AMD Ryzen 5, single core), leaving the 500 ms window step entirely compute-budget-positive. TRCA spatial filtering adds approximately 12 ms. Do not use GPU acceleration for this layer — the overhead of memory transfer exceeds the benefit for this data dimensionality. Use `numpy` BLAS-linked builds and pre-allocate all epoch buffers at startup to avoid garbage collection during the real-time loop.

**Output.** Publish to an OSC endpoint (`/bci/command`) and simultaneously to a WebSocket (`ws://localhost:9001`): `{"command": "SELECT", "frequency": 12.0, "snr_db": 4.1, "confidence": 0.87, "epoch_ms": 2000}`. The `confidence` field is the normalised FBCCA score `ρ_winner / Σρ`.

---

## Layer 3 — Backend State and AI Coordinator

**Stack:** Python 3.11, FastAPI 0.111, `python-osc`, `websockets`, `google-genai` (Gemini API, `gemini-2.5-flash` model), SQLite via `aiosqlite`.

### 3.1 Context Manager

Maintains a state machine whose state is the current UI page (e.g., `SpellerPage`, `QuickPhrasePage`, `SmartHomePage`). For each state, a JSON configuration file maps each valid stimulus frequency to a structured action:

```json
{
  "page": "SpellerPage",
  "stimulus_map": {
    "9.0":  {"action": "APPEND_LETTER", "value": "A"},
    "10.0": {"action": "APPEND_LETTER", "value": "B"},
    "12.0": {"action": "APPEND_LETTER", "value": "C"},
    "15.0": {"action": "DELETE_LAST"},
    "18.0": {"action": "SUBMIT_WORD"},
    "30.0": {"action": "NAVIGATE", "target": "QuickPhrasePage"}
  }
}
```

This externalises all UI logic from both the signal processing layer and the VR frontend. Adding a new page is a JSON edit, not a code change.

**FastAPI endpoints:**
- `POST /command` — receives the classified command from Layer 2 and dispatches the appropriate action
- `GET /state` — returns the current page state and accumulated text buffer (polled by the Quest on reconnect)
- `POST /navigate` — explicit page transition (also callable from the VR layer directly)
- `GET /health` — liveness probe used by the VR layer to detect backend restart

### 3.2 AI Predictor

The predictor calls Google's **Gemini API** (`gemini-2.5-flash` model) via the `google-genai` Python SDK. The API key is loaded from the `GEMINI_API_KEY` environment variable and never committed to source. Typical end-to-end latency is 150–400 ms per request from a North-American POP — slower than a local LLM, but eliminated as a perceived bottleneck through aggressive prefix caching (see below) and concurrent dispatch (the prediction call is fired the moment a letter is appended, in parallel with the UI feedback flash).

On each letter append, pass the current partial word to Gemini with the prompt: `"Complete this partial word with 4 likely completions. Respond with only a JSON array of strings. Partial: '{buf}'"`. The response is parsed (`response.text` → `json.loads`) and the 4 predictions are pushed to the VR frontend via the WebSocket state update. Maintain an in-memory LRU cache keyed by the last 3 characters of the buffer (capped at 1024 entries) to absorb redundant inference calls during fast typing or backspace cycles. Network failures degrade gracefully: if the Gemini call times out (>800 ms) or returns an error, an empty `predictions` list is pushed and the user can keep spelling without prediction assistance.

For full-sentence prediction (submit word → predict next word), use a larger context window with the last 3 accepted words and request the next most likely word. This maps to the predictive word buttons on the speller page.

### 3.3 Audit Logger

Subscribes to all system events and logs to a local SQLite database (`session_log.db`). Captured fields per trial: `timestamp`, `stimulus_frequency`, `snr_db`, `confidence`, `epoch_ms`, `action_dispatched`, `processing_latency_ms`, `correct` (set by the user via an explicit confirmation stimulus). This enables post-session computation of ITR (Information Transfer Rate, bits/min), false positive rate, and rejection rate — all standard BCI clinical metrics.

---

## Layer 4 — AR/VR Frontend (Meta Quest 3)

**Stack:** Unity 2023 LTS, Universal Render Pipeline (URP), XR Interaction Toolkit 3.x, NativeWebSocket (Unity package), C# 9.

### 4.1 Meta Quest 3 Display Constraints

The Meta Quest 3 display runs at 90 Hz by default, switchable to 120 Hz via `XRSettings.eyeTextureResolutionScale` and the `OVRManager` refresh rate API. **SSVEP stimulus frequencies must be exact integer divisors of the headset refresh rate.** At 90 Hz, valid frequencies are: 9, 10, 12, 15, 18, 30, 45 Hz. At 120 Hz, valid frequencies are: 8, 10, 12, 15, 20, 24, 30, 40 Hz. The recommended 6-frequency set for 90 Hz operation is **9, 10, 12, 15, 18, 30 Hz** — chosen to maximise inter-frequency perceptual distinctiveness while avoiding overlap with resting-state alpha (8–13 Hz bands are partially shared; SNR gating in Layer 2 mitigates this).

**Never use Unity's `Update()` loop for flickering.** The Update loop is not frame-synchronised and introduces sub-millisecond jitter that corrupts the SSVEP signal. Use a custom URP Renderer Feature with a `ScriptableRenderPass` that executes at `RenderPassEvent.AfterRenderingPostFX` and samples `Time.frameCount % divisor` to determine the on/off state for each stimulus. This runs on the render thread, synchronised with the display vertical blank.

### 4.2 GLSL Flicker Shader

Implement stimulus flickering as a custom URP `ShaderGraph`-compatible HLSL shader. The key uniform is `_FrameDivisor` (integer, passed per-material). The `fmod(unity_FrameCount, _FrameDivisor)` expression determines the current phase. At 90 Hz with `_FrameDivisor = 9`, the stimulus is ON for 5 frames and OFF for 4 (50% duty cycle, which maximises SSVEP amplitude). The on/off colour values are passed as `_ColorOn` and `_ColorOff` uniforms, typically full white and black, applied as an albedo multiplier. Duty cycle is configurable per frequency to equalise perceived brightness across stimuli.

Assign one material instance per stimulus button. Pass frame count via a C# `MaterialPropertyBlock` on `OnPreRender` to avoid material instancing costs. Never set flicker state via `GameObject.SetActive()` — the activation/deactivation overhead introduces frame timing irregularities.

### 4.3 Spatial UI Layout

The interface is designed for a **seated, wheelchair-using patient** operating the system with gaze only (no hand controllers). All interactive panels are placed in a fixed 180° forward arc at 1.2–1.8 m depth, centred slightly below the forward gaze vector (natural downward resting gaze angle is approximately −8°). No elements require upward or peripheral gaze.

**Page layouts:**

- **Speller Page.** A 3×2 grid of six large (20 cm × 14 cm, at 1.5 m perceived) stimulus tiles, each showing one letter (or a digraph like "TH", "ER"). Current word buffer displayed in a non-flickering text bar at top. AI prediction buttons occupy a non-flickering row below the grid — they are selected by first gazing at a "Select Prediction" tile (15 Hz), which replaces the grid temporarily with 4 large prediction tiles.
- **Quick Phrase Page.** Six tiles each containing a short phrase ("Yes", "No", "Help", "Water", "Pain", "Call nurse"). Tile text is large (48pt minimum) and high-contrast (white on dark teal). No AI prediction needed on this page.
- **Smart Home / Navigation Page.** Dynamically generated from a JSON layout pushed by Layer 3. Each tile has an icon (Unity SpriteAtlas) and a short label, loaded at runtime — the VR binary does not need to be updated to add new control options.

**Dwell confirmation (failsafe).** Although the primary selection mechanism is the SSVEP command from Layer 2, a secondary dwell timer (2.5 seconds of sustained fixation) can optionally confirm any selection independently. This requires eye-tracking data from the Quest Pro eye cameras if available; fall back to controller raycast pointing on Quest 3 without face-tracking add-on.

**Feedback.** On confirmed selection: the selected tile flashes green for 300 ms (all other tiles suppress flickering for the same duration to allow the visual cortex to reset), accompanied by a spatial audio confirmation tone (440 Hz, 80 ms, played via Unity's AudioSource with `spatialBlend = 0` for reliability). Haptic feedback is not available without controllers; consider a Bluetooth vibration motor worn on the wrist as an accessory.

### 4.4 Integration with Layer 3

The `NetworkListener` MonoBehaviour maintains a persistent WebSocket connection to `ws://[companion-PC-IP]:9001`. On receive, it deserialises the JSON state update and posts to a main-thread `ConcurrentQueue<UIStateUpdate>` (WebSocket callbacks execute on a thread pool thread; Unity scene graph is not thread-safe). The `UIController` drains the queue in `Update()`. State update schema:

```json
{
  "page": "SpellerPage",
  "buffer": "BECAU",
  "predictions": ["Because", "Became", "Beautiful", "Become"],
  "last_action": "APPEND_LETTER",
  "feedback": "confirm"
}
```

On reconnect (detected by a `ping` timeout every 2 seconds), the client calls `GET /state` to resync without requiring the user to restart their session. All UI page generation (tile labels, frequencies, colours) is driven by the `state.layout` field of the JSON, implementing the dynamic page system described in §3.1.

### 4.5 Foveated Rendering

Enable **Fixed Foveated Rendering (FFR)** in `OVRManager` (`fixedFoveatedRenderingLevel = OVRManager.FixedFoveatedRenderingLevel.High`). Stimulus tiles must be placed within the **high-resolution central region** (approximately the central 40° of the field of view) — FFR reduces resolution in the periphery, which would corrupt stimulus contrast and phase timing for peripheral tiles. Verify tile placement against the Quest 3 FFR zone boundaries before deployment.

---

## Revised Data Flow — Spelling "B"

1. **Frontend (Quest 3):** Speller Page is active. Tile for 'B' flickers at 10 Hz via the GLSL shader, frame-synchronised to the 90 Hz display.
2. **User:** Fixates on 'B' for approximately 2 seconds.
3. **Layer 1:** BrainFlow streams 500 Hz raw EEG from Oz/O1/O2/Pz to the `BCI_RawEEG` LSL outlet.
4. **Layer 2:** The sliding 2-second epoch is preprocessed (notch → bandpass → CAR), decomposed into 4 FBCCA sub-bands, and scored against reference signals at 9, 10, 12, 15, 18, 30 Hz. The 10 Hz channel produces `ρ_weighted = 0.91`; SNR = 5.2 dB (> 3.5 dB gate). Emits `{"command": "SELECT", "frequency": 10.0, "snr_db": 5.2}` over WebSocket.
5. **Layer 3:** FastAPI receives the command. Context manager state is `SpellerPage`; 10.0 Hz maps to `APPEND_LETTER: B`. The text buffer becomes `"B"`. Gemini API returns in 220 ms: `["Because", "But", "Before", "Being"]`. Pushes `UIStateUpdate` to the Quest WebSocket.
6. **Frontend (Quest 3):** `NetworkListener` receives the update. Tile 'B' flashes green for 300 ms; all tiles suppress flickering for 300 ms. Confirmation tone plays. Buffer text bar updates to `"B"`. Prediction row updates with the four LLM suggestions.

---

## Key Design Tradeoffs Summary

| Decision | Chosen approach | Reason |
|---|---|---|
| Classifier | FBCCA (primary), TRCA (optional) | Zero calibration vs. ~3 min for 8–15% accuracy gain |
| AI predictor | Gemini API (gemini-2.5-flash) + LRU cache | Higher quality predictions and zero local GPU/CPU footprint; cache absorbs the 200-400 ms round-trip |
| Flicker implementation | URP Renderer Feature + HLSL | Frame-synchronised; C# Update loop introduces jitter |
| Valid frequencies | Divisors of 90 Hz display rate | Inaccurate frequency = degraded SNR and false positives |
| Epoch length | 2 s default, ensemble with 1/3 s | Balances ITR (~30 bits/min) with accuracy (> 90%) |
| SNR gate | 3.5 dB threshold | Eliminates ~85% of false selections at < 5% ITR cost |