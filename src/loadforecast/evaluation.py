"""Error metrics and the Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(f - y) / y) * 100)


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(f, float) - np.asarray(y, float)) ** 2)))


def interval_scores(y, lo, hi, coverage: float = 0.8) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    a = 1 - coverage
    winkler = (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo)),
            "winkler": float(np.mean(winkler))}


def diebold_mariano(y, f1, f2, dates, loss: str = "abs") -> dict:
    """Is f1 more accurate than f2? Negative statistic means yes.

    Hours within a day are highly correlated, so the loss difference is averaged
    per day first. Newey-West variance (lag 7) with the HLN small-sample factor.
    """
    y, f1, f2 = (np.asarray(a, float) for a in (y, f1, f2))
    err = (lambda e: np.abs(e)) if loss == "abs" else np.square
    d = pd.Series(err(y - f1) - err(y - f2)).groupby(np.asarray(dates)).mean().values
    n, h = len(d), 7
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h + 1)]
    var = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm_stat": float(stat), "p_value": float(p), "n_days": int(n), "mean_loss_diff": float(d.mean())}
