# app.py
from typing import Optional, List, Dict, Any
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from signal_engine import (
    DEFAULTS, backtest_stats, evaluate_signal, make_signal_text
)

app = FastAPI(title="Sweep + Displaced EMA Backtest API", version="1.1.0")

class BacktestRequest(BaseModel):
    ticker: str
    period: str = DEFAULTS["intraday_period"]
    interval: str = DEFAULTS["intraday_interval"]
    rr: float = DEFAULTS["default_rr"]
    use_displacement: bool = DEFAULTS["use_displacement"]
    displacement_factor: float = DEFAULTS["displacement_factor"]
    sweep_lookback_bars: int = DEFAULTS["sweep_lookback_bars"]
    recent_sweep_window: int = DEFAULTS["recent_sweep_window"]
    ema_len: int = DEFAULTS["ema_len"]
    risk_per_trade_pct: float = DEFAULTS["risk_per_trade_pct"]
    commission_pct: float = DEFAULTS["commission_pct"]
    cash: int = 100_000
    return_trades: bool = False
    max_trades: int = 200

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return DEFAULTS

def series_to_safe_dict(s: pd.Series) -> Dict[str, Any]:
    out = {}
    for k, v in s.items():
        if pd.isna(v):
            out[k] = None
        else:
            try:
                out[k] = v.item() if hasattr(v, "item") else v
            except Exception:
                out[k] = str(v)
    return out

def trades_df_to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    df2 = df.head(limit).copy()
    for c in df2.columns:
        if str(df2[c].dtype).startswith("float"):
            df2[c] = df2[c].astype(float)
        elif str(df2[c].dtype).startswith("int"):
            df2[c] = df2[c].astype(int)
        elif "datetime" in str(df2[c].dtype):
            df2[c] = pd.to_datetime(df2[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df2.to_dict(orient="records")

@app.post("/backtest")
def backtest_endpoint(req: BacktestRequest):
    try:
        stats = backtest_stats(
            ticker=req.ticker, period=req.period, interval=req.interval,
            rr=req.rr, use_displacement=req.use_displacement,
            displacement_factor=req.displacement_factor,
            sweep_lookback_bars=req.sweep_lookback_bars,
            recent_sweep_window=req.recent_sweep_window,
            ema_len=req.ema_len, risk_per_trade_pct=req.risk_per_trade_pct,
            commission_pct=req.commission_pct, cash=req.cash
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = series_to_safe_dict(stats)
    trades = None
    if req.return_trades:
        trades_df = getattr(stats, "_trades", None)
        if trades_df is not None:
            trades = trades_df_to_records(trades_df, req.max_trades)

    return {"summary": summary, "trades": trades}

class SignalRequest(BaseModel):
    ticker: str
    period: str = DEFAULTS["intraday_period"]
    interval: str = DEFAULTS["intraday_interval"]
    rr: float = DEFAULTS["default_rr"]
    use_displacement: bool = DEFAULTS["use_displacement"]
    displacement_factor: float = DEFAULTS["displacement_factor"]
    sweep_lookback_bars: int = DEFAULTS["sweep_lookback_bars"]
    recent_sweep_window: int = DEFAULTS["recent_sweep_window"]
    ema_len: int = DEFAULTS["ema_len"]

@app.post("/signal")
def signal_endpoint(req: SignalRequest):
    try:
        signal, info = evaluate_signal(
            ticker=req.ticker, period=req.period, interval=req.interval,
            rr=req.rr, use_displacement=req.use_displacement,
            displacement_factor=req.displacement_factor,
            sweep_lookback_bars=req.sweep_lookback_bars,
            recent_sweep_window=req.recent_sweep_window,
            ema_len=req.ema_len
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"signal": signal, "info": info, "text": make_signal_text(signal, info)}

@app.post("/backtest/plot")
def backtest_plot_endpoint(req: BacktestRequest):
    # Basit bir HTML ile dışarı embed etmek istersen burada string döndürüyoruz
    # (Grafik dosyasını üretmek yerine sadece JSON + sinyal endpoint'ini öneririm)
    return Response(
        content="<html><body><h3>Plot endpoint’i bu sürümde devre dışı.</h3></body></html>",
        media_type="text/html"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
