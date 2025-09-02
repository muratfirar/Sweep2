# signal_engine.py
# Strateji + veri çekme + sinyal hesaplama (API ve Bot ortak kullanır)

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy, set_bokeh_output

# Server ortamında Bokeh notebook gereksiz
set_bokeh_output(notebook=False)

DEFAULTS = {
    "intraday_interval": "15m",
    "intraday_period": "60d",
    "default_rr": 2.0,
    "use_displacement": True,
    "displacement_factor": 1.8,
    "sweep_lookback_bars": 192,
    "recent_sweep_window": 60,
    "ema_len": 50,
    "risk_per_trade_pct": 1.0,
    "commission_pct": 0.0005,
}

def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False, group_by="column"
    )
    if df is None or df.empty:
        raise ValueError(f"{ticker} için veri çekilemedi. period/interval kombinasyonunu kontrol et.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {missing}")

    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Veri çekildi ama temizleme sonrası boş kaldı.")
    return df

def displaced_ema(close: pd.Series, length: int, use_disp: bool, disp_factor: float) -> pd.Series:
    ema = close.ewm(span=length, adjust=False).mean()
    if use_disp:
        disp_bars = max(1, int(round(disp_factor * 3)))  # 1.8 ~ 5 bar
        return ema.shift(disp_bars)  # geçmiş EMA'yı ileri kaydırıyoruz (causal)
    return ema

def rolling_swing_high(high: pd.Series, lookback: int) -> pd.Series:
    return high.shift(1).rolling(lookback).max()

def rolling_swing_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.shift(1).rolling(lookback).min()

def find_sweep_signals(
    df: pd.DataFrame,
    sweep_lookback_bars: int,
    ema_len: int,
    use_displacement: bool,
    displacement_factor: float,
) -> pd.DataFrame:
    out = df.copy()
    out["swing_hi"] = rolling_swing_high(out["High"], sweep_lookback_bars)
    out["swing_lo"] = rolling_swing_low(out["Low"], sweep_lookback_bars)
    out["swept_high"] = (out["High"] > out["swing_hi"])
    out["swept_low"] = (out["Low"] < out["swing_lo"])
    out["dema"] = displaced_ema(out["Close"], ema_len, use_displacement, displacement_factor)

    out["bars_since_high_sweep"] = np.nan
    out["bars_since_low_sweep"] = np.nan
    last_high_sweep = None
    last_low_sweep = None
    for i in range(len(out)):
        if out["swept_high"].iat[i]:
            last_high_sweep = i
        if out["swept_low"].iat[i]:
            last_low_sweep = i
        if last_high_sweep is not None:
            out["bars_since_high_sweep"].iat[i] = i - last_high_sweep
        if last_low_sweep is not None:
            out["bars_since_low_sweep"].iat[i] = i - last_low_sweep
    return out

# ==== Backtest Strategy ====
class SweepDisplacementStrategy(Strategy):
    rr = DEFAULTS["default_rr"]
    risk_pct = DEFAULTS["risk_per_trade_pct"]
    recent_window = DEFAULTS["recent_sweep_window"]
    sweep_lookback_bars = DEFAULTS["sweep_lookback_bars"]
    ema_len = DEFAULTS["ema_len"]
    use_displacement = DEFAULTS["use_displacement"]
    displacement_factor = DEFAULTS["displacement_factor"]

    def init(self):
        df = self.data._df
        enriched = find_sweep_signals(
            df, self.sweep_lookback_bars, self.ema_len,
            self.use_displacement, self.displacement_factor
        )
        self.dema = self.I(lambda _: enriched["dema"].values, df["Close"])
        self.bars_since_high_sweep = self.I(lambda _: enriched["bars_since_high_sweep"].values, df["Close"])
        self.bars_since_low_sweep  = self.I(lambda _: enriched["bars_since_low_sweep"].values, df["Close"])

    def next(self):
        price = float(self.data.Close[-1])
        if not self.position:
            # SHORT
            if (
                (not np.isnan(self.bars_since_high_sweep[-1])) and
                (self.bars_since_high_sweep[-1] <= self.recent_window) and
                (not np.isnan(self.dema[-1])) and price < float(self.dema[-1])
            ):
                stop = float(np.nanmax(self.data.High[-int(self.recent_window):]))
                if stop > price:
                    risk_per_unit = stop - price
                    if risk_per_unit > 0:
                        risk_cash = self.risk_pct / 100.0 * self.equity
                        size = max(1, int(risk_cash / risk_per_unit))
                        tp = price - self.rr * (stop - price)
                        self.sell(size=size, sl=stop, tp=tp)

            # LONG
            if (
                (not np.isnan(self.bars_since_low_sweep[-1])) and
                (self.bars_since_low_sweep[-1] <= self.recent_window) and
                (not np.isnan(self.dema[-1])) and price > float(self.dema[-1])
            ):
                stop = float(np.nanmin(self.data.Low[-int(self.recent_window):]))
                if stop < price:
                    risk_per_unit = price - stop
                    if risk_per_unit > 0:
                        risk_cash = self.risk_pct / 100.0 * self.equity
                        size = max(1, int(risk_cash / risk_per_unit))
                        tp = price + self.rr * (price - stop)
                        self.buy(size=size, sl=stop, tp=tp)

def backtest_stats(
    ticker: str, period: str, interval: str,
    rr: float, use_displacement: bool, displacement_factor: float,
    sweep_lookback_bars: int, recent_sweep_window: int,
    ema_len: int, risk_per_trade_pct: float, commission_pct: float,
    cash: int
) -> pd.Series:
    data = fetch_data(ticker, period, interval)
    StrategyCls = type("CustomStrategy", (SweepDisplacementStrategy,), dict(
        rr=rr, risk_pct=risk_per_trade_pct, recent_window=recent_sweep_window,
        sweep_lookback_bars=sweep_lookback_bars, ema_len=ema_len,
        use_displacement=use_displacement, displacement_factor=displacement_factor
    ))
    bt = Backtest(
        data, StrategyCls, cash=cash, commission=commission_pct,
        trade_on_close=False, exclusive_orders=True, hedging=False
    )
    return bt.run()

def evaluate_signal(
    ticker: str, period: str, interval: str,
    rr: float, use_displacement: bool, displacement_factor: float,
    sweep_lookback_bars: int, recent_sweep_window: int,
    ema_len: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Canlı sinyal: 'LONG_ENTRY' | 'SHORT_ENTRY' | 'NONE'
    """
    df = fetch_data(ticker, period, interval)
    enriched = find_sweep_signals(
        df, sweep_lookback_bars, ema_len, use_displacement, displacement_factor
    )
    last = enriched.iloc[-1]
    price = float(df["Close"].iloc[-1])
    ts = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")

    long_setup = (
        (not np.isnan(last["bars_since_low_sweep"])) and
        (last["bars_since_low_sweep"] <= recent_sweep_window) and
        (not np.isnan(last["dema"])) and price > float(last["dema"])
    )
    short_setup = (
        (not np.isnan(last["bars_since_high_sweep"])) and
        (last["bars_since_high_sweep"] <= recent_sweep_window) and
        (not np.isnan(last["dema"])) and price < float(last["dema"])
    )

    signal = "LONG_ENTRY" if long_setup else "SHORT_ENTRY" if short_setup else "NONE"
    info = {
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "price": price,
        "time": ts,
        "dema": None if np.isnan(last["dema"]) else float(last["dema"]),
        "bars_since_low_sweep": None if np.isnan(last["bars_since_low_sweep"]) else int(last["bars_since_low_sweep"]),
        "bars_since_high_sweep": None if np.isnan(last["bars_since_high_sweep"]) else int(last["bars_since_high_sweep"]),
        "recent_window": recent_sweep_window,
        "ema_len": ema_len,
        "displacement_factor": displacement_factor,
        "use_displacement": use_displacement,
        "rr": rr,
    }
    return signal, info

def make_signal_text(signal: str, info: Dict[str, Any]) -> str:
    if signal == "LONG_ENTRY":
        tag = "🟢 LONG"
    elif signal == "SHORT_ENTRY":
        tag = "🔴 SHORT"
    else:
        tag = "⚪ NO SIGNAL"
    return (
        f"{tag} | {info['ticker']} ({info['interval']})\n"
        f"t: {info['time']} | px: {info['price']:.4f}\n"
        f"dEMA: {info['dema']}\n"
        f"since_low_sweep: {info['bars_since_low_sweep']} | since_high_sweep: {info['bars_since_high_sweep']}\n"
        f"win:{info['recent_window']} ema:{info['ema_len']} disp:{info['displacement_factor']}"
    )
