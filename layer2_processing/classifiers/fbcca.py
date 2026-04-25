"""Filter Bank Canonical Correlation Analysis (FBCCA) classifier.

Calibration-free SSVEP classifier following Chen et al. 2015
(*Filter bank canonical correlation analysis for implementing a high-speed
SSVEP-based brain–computer interface*, J. Neural Eng.).

Pipeline per epoch:

    for each sub-band b:
        X_b = bandpass(epoch, sub_band_b)          # via Chebyshev I (sosfiltfilt)
        for each candidate frequency f:
            Y_f = reference signals at f, 2f, …, n_harmonics·f
            ρ_b,f = max canonical correlation (X_b, Y_f)

    score[f] = Σ_b w_b · ρ_b,f²       with   w_b = b^(−a) + b_offset

The frequency with the maximum score wins; confidence is the softmax-style
ratio score_winner / Σ score_i.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning

from layer2_processing.classifiers.base import (
    AbstractClassifier,
    ClassifierResult,
)
from layer2_processing.config import ProcessingConfig


class FBCCAClassifier(AbstractClassifier):
    """Calibration-free filter-bank CCA classifier."""

    def __init__(self, cfg: ProcessingConfig) -> None:
        self._fs = float(cfg.sample_rate_hz)
        self._freqs = list(cfg.stimulus_frequencies_hz)
        self._n_harm = int(cfg.n_harmonics)

        # Pre-build a Chebyshev-Type-I band-pass SOS per sub-band.
        # rp = 1 dB pass-band ripple (Chen et al. 2015 default).
        self._sos = [
            signal.cheby1(
                N=cfg.sub_band_filter_order,
                rp=1.0,
                Wn=[lo, hi],
                btype="band",
                fs=self._fs,
                output="sos",
            )
            for lo, hi in cfg.sub_bands_hz
        ]

        # Sub-band weights w_k = k^(-a) + b   (Chen et al. 2015, eq. 7)
        self._weights = np.array(
            [
                k ** (-cfg.weight_a) + cfg.weight_b
                for k in range(1, len(self._sos) + 1)
            ],
            dtype=np.float64,
        )

        # Reference signals are deterministic in n; cache lazily once epoch
        # length is known (CCA is invariant to n only after recompute).
        self._ref_cache: dict[int, list[np.ndarray]] = {}

    @property
    def stimulus_frequencies_hz(self) -> list[float]:
        return list(self._freqs)

    def _references(self, n: int) -> list[np.ndarray]:
        """Return a list of (n, 2·n_harm) reference matrices, one per freq."""
        cached = self._ref_cache.get(n)
        if cached is not None:
            return cached
        t = np.arange(n) / self._fs
        refs: list[np.ndarray] = []
        for f in self._freqs:
            cols: list[np.ndarray] = []
            for h in range(1, self._n_harm + 1):
                cols.append(np.sin(2 * np.pi * h * f * t))
                cols.append(np.cos(2 * np.pi * h * f * t))
            refs.append(np.stack(cols, axis=1).astype(np.float64))
        self._ref_cache[n] = refs
        return refs

    @staticmethod
    def _max_corr(X: np.ndarray, Y: np.ndarray) -> float:
        """Fit a 1-component CCA and return the canonical correlation in [0, 1]."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                cca = CCA(n_components=1, max_iter=500)
                cca.fit(X, Y)
                Xc, Yc = cca.transform(X, Y)
            except Exception:  # noqa: BLE001
                return 0.0
        Xc = Xc.ravel()
        Yc = Yc.ravel()
        if np.std(Xc) < 1e-12 or np.std(Yc) < 1e-12:
            return 0.0
        rho = float(np.corrcoef(Xc, Yc)[0, 1])
        if np.isnan(rho):
            return 0.0
        return abs(rho)

    def predict(self, epoch: np.ndarray) -> ClassifierResult:
        if epoch.ndim != 2:
            raise ValueError(
                f"epoch must be 2-D (channels, n_samples); got shape {epoch.shape}"
            )
        n = epoch.shape[1]
        epoch_64 = np.asarray(epoch, dtype=np.float64)
        refs = self._references(n)

        # Pre-filter the epoch through every sub-band once (most expensive step).
        Xs = [
            signal.sosfiltfilt(sos, epoch_64, axis=1).T  # (n, channels)
            for sos in self._sos
        ]

        scores = np.zeros(len(self._freqs), dtype=np.float64)
        for i, Y in enumerate(refs):
            s = 0.0
            for k, X in enumerate(Xs):
                rho = self._max_corr(X, Y)
                s += self._weights[k] * (rho ** 2)
            scores[i] = s

        win = int(np.argmax(scores))
        total = float(scores.sum())
        confidence = float(scores[win] / total) if total > 1e-12 else 0.0
        return ClassifierResult(
            frequency_hz=float(self._freqs[win]),
            confidence=confidence,
            raw_scores=scores.astype(np.float32),
        )
