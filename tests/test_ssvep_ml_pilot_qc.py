"""pilot_qc CLI smoke."""

from __future__ import annotations

import numpy as np

from experiments.ssvep_ml.pilot_qc import main as pilot_main


def test_pilot_qc_runs(tmp_path: Path) -> None:
    n, c, t = 6, 8, 375
    X = np.random.default_rng(0).standard_normal((n, c, t)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    p = tmp_path / "e.npz"
    np.savez_compressed(
        p,
        X=X,
        y=y,
        trial_id=np.arange(n, dtype=np.int64),
        session=np.array(["s"] * n, dtype=object),
        channel_names=np.array([f"ch{i}" for i in range(c)], dtype=object),
        sample_rate_hz=np.int32(500),
    )
    rc = pilot_main(["--epochs-npz", str(p), "--max-trials", "3"])
    assert rc == 0
