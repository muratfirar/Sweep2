# Sweep + Displaced EMA — API & Telegram Signal Bot

Bu repo, **FastAPI** ile backtest/sinyal endpoint’leri ve **Telegram** üzerinden canlı **LONG/SHORT** sinyal bildirimi yapan bir bot içerir.

## Özellikler
- **/backtest**: backtesting.py ile RR + EMA + displacement + sweep kurallarına göre backtest
- **/signal**: son bar'a göre LONG/SHORT/NONE tespiti
- **Telegram Bot**: `/subscribe <TICKER> [period] [interval]` ile abone ol, sinyal değişince mesaj gelsin

## Kurulum (Lokal)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # TELEGRAM_BOT_TOKEN gir
python app.py  # API: http://127.0.0.1:8000
python tg_bot.py  # Bot
