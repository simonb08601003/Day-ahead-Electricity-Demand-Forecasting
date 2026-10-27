"""Loading and sanity-checking hourly load data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


@dataclass(frozen=True)
class Market:
    file: str
    timezone: str
    country: str      # holiday calendar
    val_start: str
    test_start: str


MARKETS = {
    "spain": Market("energy_dataset.csv", "Europe/Madrid", "ES", "2017-07-01", "2018-01-01"),
    "pjme": Market("PJME_hourly.csv", "America/New_York", "US", "2016-08-01", "2017-08-01"),
}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, max_missing_frac: float = 0.01) -> None:
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise DataValidationError("timestamps must be sorted and unique")
    if (pd.Series(df.index).diff().dropna() != pd.Timedelta("1h")).any():
        raise DataValidationError("gaps in hourly index")
    ratio = midnight_jump_ratio(df["load"])
    if ratio > 1.15:
        raise DataValidationError(f"midnight jumps on days 1-12 are {ratio:.2f}x days 13-31; "
                                  "day and month may be swapped")
    miss = df["load"].isna().mean()
    if miss > max_missing_frac:
        raise DataValidationError(f"{miss:.2%} of load values missing")
    if (df["load"].dropna() <= 0).any():
        raise DataValidationError("non-positive load values")


def load_market(market: str) -> pd.DataFrame:
    """Hourly frame with `load` (MW) and, for Spain, the operator's `tso_forecast`."""
    m = MARKETS[market]
    path = DATA_DIR / m.file
    if market == "spain":
        raw = pd.read_csv(path, usecols=["time", "total load forecast", "total load actual"])
        ts = pd.to_datetime(raw["time"], utc=True).dt.tz_convert(m.timezone).dt.tz_localize(None)
        # This copy of the data has day and month swapped whenever the day is <= 12.
        # midnight_jump_ratio() is how we caught it.
        ts = ts.map(unswap_day_month)
        df = pd.DataFrame({"load": raw["total load actual"].values,
                           "tso_forecast": raw["total load forecast"].values}, index=ts)
    else:
        raw = pd.read_csv(path)
        df = pd.DataFrame({"load": raw["PJME_MW"].values}, index=pd.to_datetime(raw["Datetime"]))
    # DST leaves duplicate and missing hours; average the duplicates and fill short gaps
    df = df.groupby(level=0).mean().asfreq("1h")
    df.index.name = "ts"
    df["load"] = df["load"].interpolate(limit=6)
    df = df.dropna(subset=["load"]).asfreq("1h").ffill()
    df["local_time"] = df.index
    return df


def unswap_day_month(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.replace(month=ts.day, day=ts.month) if ts.day <= 12 else ts


def midnight_jump_ratio(load: pd.Series) -> float:
    """Average jump across midnight on days 1-12 vs days 13-31.

    Should be about 1. If day and month are swapped, days 1-12 get shuffled
    and their midnights stop lining up (the raw Spain file gives 1.35).
    """
    s = load.asfreq("1h")
    jump = s.diff().abs()[s.index.hour == 0]
    return float(jump[jump.index.day <= 12].mean() / jump[jump.index.day > 13].mean())
