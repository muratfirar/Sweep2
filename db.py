# db.py
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

DB_PATH = Path("./signals.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        period TEXT NOT NULL,
        interval TEXT NOT NULL,
        rr REAL NOT NULL,
        use_displacement INTEGER NOT NULL,
        displacement_factor REAL NOT NULL,
        sweep_lookback_bars INTEGER NOT NULL,
        recent_sweep_window INTEGER NOT NULL,
        ema_len INTEGER NOT NULL,
        UNIQUE(chat_id, ticker, period, interval)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS last_signals (
        chat_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        last_signal TEXT NOT NULL,
        last_time TEXT,
        PRIMARY KEY (chat_id, ticker, interval)
    )
    """)
    conn.commit()
    conn.close()

def add_subscription(chat_id: int, params: Dict[str, Any]) -> str:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        INSERT INTO subscriptions
        (chat_id, ticker, period, interval, rr, use_displacement, displacement_factor,
         sweep_lookback_bars, recent_sweep_window, ema_len)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chat_id, params["ticker"], params["period"], params["interval"],
            params["rr"], int(params["use_displacement"]), params["displacement_factor"],
            params["sweep_lookback_bars"], params["recent_sweep_window"], params["ema_len"]
        ))
        conn.commit()
        return "OK"
    except sqlite3.IntegrityError:
        return "ALREADY_EXISTS"
    finally:
        conn.close()

def remove_subscription(chat_id: int, ticker: str, period: str, interval: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM subscriptions WHERE chat_id=? AND ticker=? AND period=? AND interval=?
    """, (chat_id, ticker, period, interval))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted

def list_subscriptions(chat_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM subscriptions WHERE chat_id=? ORDER BY ticker, interval", (chat_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def all_subscriptions() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM subscriptions")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_last_signal(chat_id: int, ticker: str, interval: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT last_signal FROM last_signals WHERE chat_id=? AND ticker=? AND interval=?
    """, (chat_id, ticker, interval))
    row = cur.fetchone()
    conn.close()
    return row["last_signal"] if row else None

def set_last_signal(chat_id: int, ticker: str, interval: str, signal: str, when: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO last_signals (chat_id, ticker, interval, last_signal, last_time)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, ticker, interval)
    DO UPDATE SET last_signal=excluded.last_signal, last_time=excluded.last_time
    """, (chat_id, ticker, interval, signal, when))
    conn.commit()
    conn.close()
