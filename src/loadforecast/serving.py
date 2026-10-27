"""Turn recent history into a day-ahead forecast with the saved model."""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import MARKETS
from .features import ISSUE_HOUR, build_dataset

ARTIFACTS = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).resolve().parents[2] / "artifacts"))
MIN_HISTORY_DAYS = 16   # the two-week lag plus the weekly rolling mean


@lru_cache(maxsize=4)
def load_artifact(market: str) -> dict:
    with open(ARTIFACTS / f"{market}_model.pkl", "rb") as f:
        return pickle.load(f)


def forecast_day(market: str, history: pd.Series, target_date: str) -> pd.DataFrame:
    """24 rows of hour, point, lo80, hi80. Anything in `history` after the issue time is ignored."""
    target = pd.Timestamp(target_date).normalize()
    cutoff = target - pd.Timedelta("1D") + pd.Timedelta(hours=ISSUE_HOUR)
    hist = history[history.index < cutoff].sort_index()
    if hist.index.min() > target - pd.Timedelta(days=MIN_HISTORY_DAYS):
        raise ValueError(f"need at least {MIN_HISTORY_DAYS} days of history before the target date")
    grid = pd.date_range(hist.index.min().normalize(), target + pd.Timedelta("23h"), freq="1h")
    df = pd.DataFrame({"load": hist.reindex(grid), "local_time": grid}, index=grid)
    long, _ = build_dataset(df, MARKETS[market].country, keep_unlabelled=True)
    rows = long[long.date == target].sort_values("hour")
    if len(rows) != 24:
        raise ValueError("could not build 24 hourly rows; check the history for gaps")
    art = load_artifact(market)
    lo, hi = art["interval"].predict_interval(rows)
    return pd.DataFrame({"hour": rows.hour.values, "point": art["point"].predict(rows), "lo80": lo, "hi80": hi})
