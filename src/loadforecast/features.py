"""Features for day-ahead forecasting.

The forecast for day T goes out at 10:00 on T-1, like European grid operators do it.
So the latest data any feature can use is 09:00 on T-1.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

ISSUE_HOUR = 10

FEATURES = ["hour", "lag48", "lag72", "lag168", "lag336", "morning_tm1", "last_obs_tm1", "mean_tm2",
            "mean_week", "dow", "month", "sin_y1", "cos_y1", "sin_y2", "cos_y2", "sin_y3", "cos_y3",
            "holiday", "holiday_tm1", "holiday_tp1", "weekend"]


def to_local_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Local wall-clock time with exactly 24 hours per day, so DST days line up."""
    cols = [c for c in ("load", "tso_forecast") if c in df]
    loc = df.set_index("local_time")[cols].groupby(level=0).mean()
    end = (loc.index.max() + pd.Timedelta("1h")).floor("D") - pd.Timedelta("1h")
    return loc.reindex(pd.date_range(loc.index.min().ceil("D"), end, freq="1h")).interpolate(limit=3)


def daily_matrix(s: pd.Series) -> pd.DataFrame:
    """One row per day, one column per hour."""
    return pd.DataFrame({"date": s.index.normalize(), "hour": s.index.hour, "v": s.values}) \
        .pivot(index="date", columns="hour", values="v").sort_index()


def build_dataset(df: pd.DataFrame, country: str, keep_unlabelled: bool = False):
    """Returns (long, wide).

    long: one row per (date, hour) with features, target `y` and the operator's `tso` if there is one.
    wide: the daily load matrix and holiday flags, for the MLP.
    """
    grid = to_local_grid(df)
    L = daily_matrix(grid["load"])
    dates = L.index
    hol = holidays.country_holidays(country, years=range(dates.min().year, dates.max().year + 2))
    is_hol = pd.Series([d in hol for d in dates.date], index=dates, dtype=float)

    morning = L.loc[:, : ISSUE_HOUR - 1].mean(axis=1)
    daily_mean = L.mean(axis=1)
    parts = []
    for h in range(24):
        parts.append(pd.DataFrame({
            "date": dates, "hour": h,
            "lag48": L[h].shift(2).values,
            "lag72": L[h].shift(3).values,
            "lag168": L[h].shift(7).values,
            "lag336": L[h].shift(14).values,
            "morning_tm1": morning.shift(1).values,          # 00:00-09:00 yesterday
            "last_obs_tm1": L[ISSUE_HOUR - 1].shift(1).values,
            "mean_tm2": daily_mean.shift(2).values,
            "mean_week": daily_mean.shift(2).rolling(7).mean().values,
            "y": L[h].values,
        }))
    long = pd.concat(parts, ignore_index=True)
    long["dow"] = long.date.dt.dayofweek
    long["month"] = long.date.dt.month
    doy = long.date.dt.dayofyear
    for k in (1, 2, 3):
        long[f"sin_y{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        long[f"cos_y{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    long["holiday"] = long.date.map(is_hol).fillna(0.0)
    long["holiday_tm1"] = long.date.map(is_hol.shift(1)).fillna(0.0)
    long["holiday_tp1"] = long.date.map(is_hol.shift(-1)).fillna(0.0)
    long["weekend"] = (long.dow >= 5).astype(float)
    if "tso_forecast" in grid:
        tso = daily_matrix(grid["tso_forecast"]).stack().rename("tso").reset_index()
        tso.columns = ["date", "hour", "tso"]
        long = long.merge(tso, on=["date", "hour"], how="left")
    optional = ("tso", "y") if keep_unlabelled else ("tso",)
    long = long.dropna(subset=[c for c in long.columns if c not in optional])
    long = long.sort_values(["date", "hour"]).reset_index(drop=True)
    return long, {"L": L, "holiday": is_hol}
