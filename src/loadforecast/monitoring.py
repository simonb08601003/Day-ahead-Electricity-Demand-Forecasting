"""Replay the test year one day at a time and compare three retraining policies:
never, monthly, and when the rolling error gets too high.

python -m loadforecast.monitoring --market pjme
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MARKETS, load_market
from .evaluation import mape
from .features import build_dataset
from .models import LightGBMForecaster

REPORTS = Path(__file__).resolve().parents[2] / "reports"
POLICIES = ["static", "monthly", "on_alert"]


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def replay(market: str, alert_window: int = 14, alert_factor: float = 1.3, cooldown: int = 28) -> dict:
    m = MARKETS[market]
    long, _ = build_dataset(load_market(market), m.country)
    history = long[long.date < m.test_start]
    test_days = sorted(long[long.date >= m.test_start].date.unique())

    # what "normal" error looks like: validation MAPE of a model trained before it
    val = history[history.date >= m.val_start]
    ref_mape = mape(val.y, LightGBMForecaster(seed=0).fit(long[long.date < m.val_start]).predict(val))
    ref_load = history[history.date >= pd.Timestamp(m.test_start) - pd.Timedelta(days=365)].y.values

    models = {p: LightGBMForecaster(seed=0).fit(history) for p in POLICIES}
    retrains = {p: [] for p in POLICIES}
    last_alert_retrain = test_days[0]
    log = []
    for i, day in enumerate(test_days):
        today, seen = long[long.date == day], long[long.date < day]
        row = {"date": str(pd.Timestamp(day).date())}
        for p in POLICIES:
            row[f"mape_{p}"] = mape(today.y, models[p].predict(today))
        log.append(row)
        recent = pd.DataFrame(log[-alert_window:])
        roll = recent.mape_on_alert.mean()
        row["rolling_mape_on_alert"] = float(roll)
        row["psi_load_30d"] = psi(ref_load, seen[seen.date >= day - pd.Timedelta(days=30)].y.values)
        if i + 1 == len(test_days):
            break
        if pd.Timestamp(test_days[i + 1]).day == 1:
            models["monthly"] = LightGBMForecaster(seed=0).fit(seen)
            retrains["monthly"].append(row["date"])
        days_since = (pd.Timestamp(day) - pd.Timestamp(last_alert_retrain)).days
        if len(recent) >= alert_window and roll > alert_factor * ref_mape and days_since >= cooldown:
            models["on_alert"] = LightGBMForecaster(seed=0).fit(seen)
            last_alert_retrain = day
            retrains["on_alert"].append(row["date"])

    df = pd.DataFrame(log)
    summary = {
        "market": market,
        "reference_val_mape": ref_mape,
        "alert_rule": f"rolling {alert_window}-day MAPE > {alert_factor} x reference, cooldown {cooldown} days",
        "test_mape": {p: float(df[f"mape_{p}"].mean()) for p in POLICIES},
        "n_retrains": {p: len(v) for p, v in retrains.items()},
        "alert_retrain_dates": retrains["on_alert"],
        "max_psi_load_30d": float(df.psi_load_30d.max()),
    }
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / f"{market}_monitoring_log.csv", index=False)
    (REPORTS / f"{market}_monitoring.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=list(MARKETS), default="pjme")
    print(json.dumps(replay(ap.parse_args().market), indent=2))
