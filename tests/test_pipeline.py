import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from loadforecast.data import DataValidationError, midnight_jump_ratio, unswap_day_month, validate  # noqa: E402
from loadforecast.evaluation import diebold_mariano, interval_scores, mape  # noqa: E402
from loadforecast.features import ISSUE_HOUR, build_dataset  # noqa: E402
from loadforecast.models import ConformalQuantileLGBM, LightGBMForecaster  # noqa: E402


def synthetic(days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=days * 24, freq="h")
    daily = 1 + 0.25 * np.sin(2 * np.pi * (idx.hour - 6) / 24)
    weekly = np.where(idx.dayofweek >= 5, 0.85, 1.0)
    load = 30000 * daily * weekly * (1 + 0.02 * rng.standard_normal(len(idx)))
    df = pd.DataFrame({"load": load}, index=idx)
    df["local_time"] = df.index
    return df


def test_validation_accepts_clean_and_rejects_gaps():
    df = synthetic()
    validate(df)
    with pytest.raises(DataValidationError):
        validate(df.drop(df.index[100]))


def test_swap_detector_flags_day_month_swap():
    df = synthetic(days=400)
    assert midnight_jump_ratio(df.load) < 1.15
    swapped = df.load.copy()
    swapped.index = swapped.index.map(unswap_day_month)   # swapping twice undoes it
    swapped = swapped.sort_index()
    assert midnight_jump_ratio(swapped) > midnight_jump_ratio(df.load)


def test_features_do_not_use_data_after_issue_time():
    df = synthetic()
    long, _ = build_dataset(df, "US")
    # triple every value from 10:00 on T-1 onward; day T's features should not move
    target = long.date.iloc[-1]
    cut = target - pd.Timedelta("1D") + pd.Timedelta(hours=ISSUE_HOUR)
    df2 = df.copy()
    df2.loc[df2.index >= cut, "load"] *= 3
    long2, _ = build_dataset(df2, "US")
    feats = [c for c in long.columns if c not in ("y", "date")]
    a = long[long.date == target][feats].reset_index(drop=True)
    b = long2[long2.date == target][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_lightgbm_beats_naive_on_synthetic():
    long, _ = build_dataset(synthetic(days=200), "US")
    tr, te = long[long.date < "2017-06-01"], long[long.date >= "2017-06-01"]
    p = LightGBMForecaster().fit(tr).predict(te)
    assert mape(te.y, p) < mape(te.y, te.lag168)


def test_conformal_interval_coverage():
    long, _ = build_dataset(synthetic(days=260, seed=1), "US")
    tr = long[long.date < "2017-06-01"]
    ca = long[(long.date >= "2017-06-01") & (long.date < "2017-08-01")]
    te = long[long.date >= "2017-08-01"]
    lo, hi = ConformalQuantileLGBM(coverage=0.8).fit(tr, ca).predict_interval(te)
    assert 0.72 < interval_scores(te.y, lo, hi)["coverage"] < 0.9


def test_diebold_mariano_detects_better_forecast():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2018-01-01", periods=200), 24)
    y = rng.normal(100, 10, len(dates))
    good, bad = y + rng.normal(0, 1, len(y)), y + rng.normal(0, 3, len(y))
    r = diebold_mariano(y, good, bad, dates)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.01
