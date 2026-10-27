"""README figures."""
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "reports"
F = R / "figures"
F.mkdir(exist_ok=True)

# one week in Spain
t = pd.read_parquet(R / "spain_test_predictions.parquet")
t["ts"] = t.date + pd.to_timedelta(t.hour, unit="h")
w = t[(t.ts >= "2018-11-12") & (t.ts < "2018-11-19")].sort_values("ts")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(w.ts, w.lo80 / 1000, w.hi80 / 1000, alpha=0.25, label="80% conformal interval")
ax.plot(w.ts, w.y / 1000, "k", lw=1.6, label="Actual")
ax.plot(w.ts, w.ensemble_lgbm_mlp / 1000, label="LightGBM + MLP ensemble")
ax.plot(w.ts, w.tso / 1000, ":", label="TSO day-ahead forecast")
ax.set_ylabel("Load (GW)")
ax.set_title("Spain, 12-18 Nov 2018: day-ahead forecasts issued 10:00 the day before")
ax.legend(loc="upper left", ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(F / "spain_week.png", dpi=150)

# test MAPE by model
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
names = {"seasonal_naive": "Seasonal\nnaive", "harmonic_ridge": "Harmonic\nridge", "lightgbm": "LightGBM",
         "torch_mlp": "PyTorch\nMLP", "ensemble_lgbm_mlp": "Ensemble", "tso": "TSO\n(uses weather)"}
for ax, mk, title in zip(axes, ["spain", "pjme"], ["Spain (test 2018)", "PJM East (test Aug 2017 - Aug 2018)"]):
    r = json.loads((R / f"{mk}_metrics.json").read_text())["point"]
    keys = [k for k in names if k in r]
    vals = [r[k]["mape"] for k in keys]
    bars = ax.bar([names[k] for k in keys], vals, color=["#999"] * 2 + ["#4C72B0"] * 3 + ["#DD8452"] * (len(keys) - 5))
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("MAPE (%)")
    ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(F / "mape_by_model.png", dpi=150)

# PJM monitoring replay
mon = pd.read_csv(R / "pjme_monitoring_log.csv", parse_dates=["date"])
summ = json.loads((R / "pjme_monitoring.json").read_text())
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mon.date, mon.mape_static.rolling(14).mean(), label="Static model")
ax.plot(mon.date, mon.mape_on_alert.rolling(14).mean(), label="Retrain on alert")
ax.axhline(1.3 * summ["reference_val_mape"], color="r", ls="--", lw=1, label="Alert threshold")
for d in summ["alert_retrain_dates"]:
    ax.axvline(pd.Timestamp(d), color="gray", lw=0.8, alpha=0.6)
ax.set_ylabel("Rolling 14-day MAPE (%)")
ax.set_title("PJM East replay: alerts cluster in heat waves and cold snaps (no weather inputs)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(F / "pjme_monitoring.png", dpi=150)
print("figures written")
