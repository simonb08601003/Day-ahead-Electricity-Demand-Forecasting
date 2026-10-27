"""The forecasters: seasonal naive, harmonic ridge, LightGBM (point and conformal interval), PyTorch MLP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURES, ISSUE_HOUR


class SeasonalNaive:
    """Same hour, one week ago."""
    name = "seasonal_naive"

    def fit(self, df):
        return self

    def predict(self, df):
        return df["lag168"].values


class HarmonicRidge:
    """Ridge regression on an hour-by-weekday profile, yearly Fourier terms and lags."""
    name = "harmonic_ridge"
    CAT = ["hour", "dow", "month"]

    def __init__(self, alpha: float = 1.0):
        num = [f for f in FEATURES if f not in self.CAT]
        self.cols = self.CAT + num
        self.model = make_pipeline(
            ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), self.CAT + ["hour_dow"]),
                               ("num", StandardScaler(), num)]),
            Ridge(alpha=alpha))

    def _x(self, df):
        x = df[self.cols].copy()
        x["hour_dow"] = df["hour"].astype(int) * 10 + df["dow"].astype(int)
        return x

    def fit(self, df):
        self.model.fit(self._x(df), df["y"])
        return self

    def predict(self, df):
        return self.model.predict(self._x(df))


def _lgbm(seed=0, **kw):
    return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                         verbose=-1, **kw)


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, seed: int = 0):
        self.model = _lgbm(seed)

    def fit(self, df):
        self.model.fit(df[FEATURES], df["y"])
        return self

    def predict(self, df):
        return self.model.predict(df[FEATURES])


class ConformalQuantileLGBM:
    """Conformalized quantile regression (Romano et al. 2019).

    Fit the 10% and 90% quantiles, then widen both ends by however much the
    calibration set says is needed to actually cover 80%.
    """
    name = "lightgbm_cqr"

    def __init__(self, coverage: float = 0.8, seed: int = 0):
        a = (1 - coverage) / 2
        self.lo = _lgbm(seed, objective="quantile", alpha=a)
        self.hi = _lgbm(seed, objective="quantile", alpha=1 - a)
        self.coverage, self.q = coverage, 0.0

    def fit(self, train, calib):
        self.lo.fit(train[FEATURES], train["y"])
        self.hi.fit(train[FEATURES], train["y"])
        y = calib["y"].values
        scores = np.maximum(self.lo.predict(calib[FEATURES]) - y, y - self.hi.predict(calib[FEATURES]))
        n = len(scores)
        self.q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * self.coverage) / n)))
        return self

    def predict_interval(self, df):
        return self.lo.predict(df[FEATURES]) - self.q, self.hi.predict(df[FEATURES]) + self.q


def make_windows(L: pd.DataFrame, holiday: pd.Series, dates: pd.DatetimeIndex):
    """MLP inputs for each target day T.

    X: the week before T-1 (168 h) plus 00:00-09:00 on T-1, divided by T-2's mean, plus calendar flags.
    Y: the 24 hours of T on the same scale. Returns X, Y, scale, kept dates.
    """
    idx = {d: i for i, d in enumerate(L.index)}
    vals = L.values
    one_day = pd.Timedelta("1D")
    X, Y, S, keep = [], [], [], []
    for d in dates:
        i = idx.get(d)
        if i is None or i < 8:
            continue
        hist, morning, scale = vals[i - 8:i - 1].ravel(), vals[i - 1, :ISSUE_HOUR], vals[i - 2].mean()
        if np.isnan(hist).any() or np.isnan(morning).any() or np.isnan(vals[i]).any():
            continue
        doy = 2 * np.pi * d.dayofyear / 365.25
        cal = np.concatenate([np.eye(7)[d.dayofweek],
                              [holiday.get(d, 0.0), holiday.get(d - one_day, 0.0), holiday.get(d + one_day, 0.0),
                               np.sin(doy), np.cos(doy)]])
        X.append(np.concatenate([hist / scale - 1, morning / scale - 1, cal]))
        Y.append(vals[i] / scale - 1)
        S.append(scale)
        keep.append(d)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(S), pd.DatetimeIndex(keep)


class TorchMLP:
    """Predicts all 24 hours of the next day at once. Early stopping on validation L1."""
    name = "torch_mlp"

    def __init__(self, hidden: int = 256, epochs: int = 200, lr: float = 1e-3, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, Xtr, Ytr, Xva, Yva, patience: int = 20):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.net = nn.Sequential(nn.Linear(Xtr.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, 24))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        xtr, ytr, xva, yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xtr))
            for b in range(0, len(xtr), 64):
                j = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.l1_loss(self.net(xtr[j]), ytr[j]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = nn.functional.l1_loss(self.net(xva), yva).item()
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_l1 = best
        return self

    def predict(self, X, scale):
        import torch
        self.net.eval()
        with torch.no_grad():
            r = self.net(torch.from_numpy(X)).numpy()
        return (r + 1) * scale[:, None]
