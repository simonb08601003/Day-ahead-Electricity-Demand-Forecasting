"""Day-ahead load forecast API."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loadforecast.data import MARKETS
from loadforecast.serving import forecast_day, load_artifact

app = FastAPI(title="Day-ahead load forecast", version="1.0")


class ForecastRequest(BaseModel):
    market: str = Field(examples=["spain"])
    target_date: str = Field(examples=["2018-06-15"])
    timestamps: list[str] = Field(description="hourly local timestamps")
    load_mw: list[float]


class HourForecast(BaseModel):
    hour: int
    point: float
    lo80: float
    hi80: float


class ForecastResponse(BaseModel):
    market: str
    target_date: str
    model_trained_through: str
    forecast: list[HourForecast]


@app.get("/health")
def health():
    return {"status": "ok", "markets": list(MARKETS)}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.market not in MARKETS:
        raise HTTPException(404, f"unknown market {req.market}")
    if len(req.timestamps) != len(req.load_mw):
        raise HTTPException(422, "timestamps and load_mw must have equal length")
    hist = pd.Series(req.load_mw, index=pd.to_datetime(req.timestamps))
    try:
        out = forecast_day(req.market, hist, req.target_date)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))
    return ForecastResponse(market=req.market, target_date=req.target_date,
                            model_trained_through=load_artifact(req.market)["trained_through"],
                            forecast=[HourForecast(**r) for r in out.to_dict("records")])
