import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from d10_core import compute_horizon


def test_missing_forward_return_does_not_change_d10_membership():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    codes = [f"{i:06d}.SZ" for i in range(1, 21)]
    signal = pd.DataFrame(
        np.tile(np.arange(1, 21, dtype=float), (3, 1)), index=dates, columns=codes
    )
    mem = pd.DataFrame(True, index=dates, columns=codes)
    fwd = pd.DataFrame(0.01, index=dates, columns=codes)
    # Top-signal stock has no forward return on day 2; it must still be in D10.
    fwd.loc[dates[1], codes[-1]] = np.nan
    stats = compute_horizon(signal, fwd, mem, h=1)
    assert stats.missing_d10 >= 1
    assert abs(stats.d10_series.loc[dates[1]]) < 1e-12
