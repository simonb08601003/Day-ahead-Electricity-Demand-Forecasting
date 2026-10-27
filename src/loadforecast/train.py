"""Train and score every model for one market, save the serving model, log to MLflow.

python -m loadforecast.train --market spain
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MARKETS, load_market, validate
from .evaluation import diebold_mariano, interval_scores, mape, rmse
from .features import build_dataset
from .models import ConformalQuantileLGBM, HarmonicRidge, LightGBMForecaster, SeasonalNaive, TorchMLP, make_windows

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
REPORTS = ROOT / "reports"


def mlp_forecast(wide, tr, va, te, seeds=3) -> pd.DataFrame:
    """Average of a few MLPs trained with different seeds, as a (date, hour, torch_mlp) frame."""
    win = lambda part: make_windows(wide["L"], wide["holiday"], pd.DatetimeIndex(part.date.unique()))  # noqa: E731
    Xtr, Ytr, _, _ = win(tr)
    Xva, Yva, _, _ = win(va)
    Xte, _, Ste, dte = win(te)
    pred = np.mean([TorchMLP(seed=s).fit(Xtr, Ytr, Xva, Yva).predict(Xte, Ste) for s in range(seeds)], axis=0)
    out = pd.DataFrame(pred, index=dte).stack().reset_index()
    out.columns = ["date", "hour", "torch_mlp"]
    return out


def run(market: str, use_mlflow: bool = True) -> dict:
    t0 = time.time()
    m = MARKETS[market]
    df = load_market(market)
    validate(df)
    long, wide = build_dataset(df, m.country)
    tr = long[long.date < m.val_start]
    va = long[(long.date >= m.val_start) & (long.date < m.test_start)]
    te = long[long.date >= m.test_start]
    trva = pd.concat([tr, va])

    # tree and linear models: refit on train+val, score the test year once
    preds = {model.name: model.fit(trva).predict(te)
             for model in (SeasonalNaive(), HarmonicRidge(), LightGBMForecaster(seed=0))}
    te = te.merge(mlp_forecast(wide, tr, va, te), on=["date", "hour"], how="left")
    for k, p in preds.items():
        te[k] = p
    te = te.dropna(subset=["torch_mlp"])
    te["ensemble_lgbm_mlp"] = (te.lightgbm + te.torch_mlp) / 2

    cqr = ConformalQuantileLGBM(coverage=0.8, seed=0).fit(tr, va)
    te["lo80"], te["hi80"] = cqr.predict_interval(te)

    cols = ["seasonal_naive", "harmonic_ridge", "lightgbm", "torch_mlp", "ensemble_lgbm_mlp"]
    tso = ["tso"] if "tso" in te and te.tso.notna().all() else []
    results = {c: {"mape": mape(te.y, te[c]), "rmse": rmse(te.y, te[c])} for c in cols + tso}
    best = min(cols, key=lambda c: results[c]["mape"])
    refs = ["seasonal_naive", "harmonic_ridge"] + tso
    dm = {f"{best}_vs_{r}": diebold_mariano(te.y, te[best], te[r], te.date) for r in refs if r != best}

    span = lambda part: [str(part.date.min().date()), str(part.date.max().date())]  # noqa: E731
    summary = {
        "market": market,
        "train": span(tr), "validation": span(va), "test": span(te),
        "n_test_hours": len(te),
        "point": results,
        "best_model": best,
        "diebold_mariano": dm,
        "interval_80": interval_scores(te.y, te.lo80, te.hi80, 0.8),
        "conformal_correction_mw": cqr.q,
        "runtime_sec": round(time.time() - t0, 1),
    }

    # the API serves LightGBM: cheap to run, no torch needed in the image
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / f"{market}_model.pkl", "wb") as f:
        pickle.dump({"point": LightGBMForecaster(seed=0).fit(trva), "interval": cqr, "market": market,
                     "trained_through": str(trva.date.max().date())}, f)
    REPORTS.mkdir(exist_ok=True)
    te.to_parquet(REPORTS / f"{market}_test_predictions.parquet", index=False)
    (REPORTS / f"{market}_metrics.json").write_text(json.dumps(summary, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
        mlflow.set_experiment(f"load-forecast-{market}")
        with mlflow.start_run(run_name=f"{market}-{time.strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_params({"market": market, "test_start": m.test_start, "val_start": m.val_start})
            for c, r in results.items():
                mlflow.log_metrics({f"{c}_mape": r["mape"], f"{c}_rmse": r["rmse"]})
            mlflow.log_metrics({f"pi80_{k}": v for k, v in summary["interval_80"].items()})
            mlflow.log_artifact(str(REPORTS / f"{market}_metrics.json"))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=list(MARKETS), default="spain")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.market, use_mlflow=not a.no_mlflow), indent=2))
