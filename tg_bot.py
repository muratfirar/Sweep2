# tg_bot.py
import asyncio
import os
from typing import Dict, Any

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from signal_engine import DEFAULTS, evaluate_signal, make_signal_text
from db import init_db, add_subscription, remove_subscription, list_subscriptions, all_subscriptions, get_last_signal, set_last_signal

load_dotenv()  # Lokal geliştirme için .env'den al (Render'da env vars panelinden)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env değişkenini ayarla.")

def parse_args(text: str) -> Dict[str, Any]:
    # basit: "/subscribe BTC-USD 60d 15m" formatını bekler
    parts = text.strip().split()
    if len(parts) < 2:
        raise ValueError("Eksik argüman. Örn: /subscribe BTC-USD 60d 15m")
    cmd = parts[0].lower()
    ticker = parts[1]
    period = parts[2] if len(parts) > 2 else DEFAULTS["intraday_period"]
    interval = parts[3] if len(parts) > 3 else DEFAULTS["intraday_interval"]
    return {
        "cmd": cmd, "ticker": ticker, "period": period, "interval": interval,
        "rr": DEFAULTS["default_rr"],
        "use_displacement": DEFAULTS["use_displacement"],
        "displacement_factor": DEFAULTS["displacement_factor"],
        "sweep_lookback_bars": DEFAULTS["sweep_lookback_bars"],
        "recent_sweep_window": DEFAULTS["recent_sweep_window"],
        "ema_len": DEFAULTS["ema_len"],
    }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Merhaba! Chat ID: {chat_id}\n"
        f"/subscribe <TICKER> [period] [interval]\n"
        f"/unsubscribe <TICKER> [period] [interval]\n"
        f"/list\n"
        f"/signal <TICKER> [period] [interval]\n\n"
        f"Varsayılanlar -> period:{DEFAULTS['intraday_period']} interval:{DEFAULTS['intraday_interval']}"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        params = parse_args(update.message.text)
        res = add_subscription(chat_id, params)
        if res == "OK":
            await update.message.reply_text(f"✅ Abone olundu: {params['ticker']} {params['period']} {params['interval']}")
        else:
            await update.message.reply_text("ℹ️ Zaten kayıtlı.")
    except Exception as e:
        await update.message.reply_text(f"❌ Hata: {e}")

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        params = parse_args(update.message.text)
        ok = remove_subscription(chat_id, params["ticker"], params["period"], params["interval"])
        await update.message.reply_text("✅ Silindi" if ok else "Bulunamadı.")
    except Exception as e:
        await update.message.reply_text(f"❌ Hata: {e}")

async def lst(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subs = list_subscriptions(chat_id)
    if not subs:
        await update.message.reply_text("Abonelik yok.")
        return
    lines = [f"- {s['ticker']} {s['period']} {s['interval']}" for s in subs]
    await update.message.reply_text("Abonelikler:\n" + "\n".join(lines))

async def inline_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = parse_args(update.message.text)
        signal, info = evaluate_signal(
            ticker=params["ticker"], period=params["period"], interval=params["interval"],
            rr=params["rr"], use_displacement=params["use_displacement"],
            displacement_factor=params["displacement_factor"],
            sweep_lookback_bars=params["sweep_lookback_bars"],
            recent_sweep_window=params["recent_sweep_window"],
            ema_len=params["ema_len"]
        )
        await update.message.reply_text(make_signal_text(signal, info))
    except Exception as e:
        await update.message.reply_text(f"❌ Hata: {e}")

async def scheduler(app: Application):
    # Her POLL_INTERVAL_SECONDS'ta tüm abonelikleri kontrol et
    while True:
        subs = all_subscriptions()
        for s in subs:
            try:
                signal, info = evaluate_signal(
                    ticker=s["ticker"], period=s["period"], interval=s["interval"],
                    rr=s["rr"], use_displacement=bool(s["use_displacement"]),
                    displacement_factor=s["displacement_factor"],
                    sweep_lookback_bars=s["sweep_lookback_bars"],
                    recent_sweep_window=s["recent_sweep_window"],
                    ema_len=s["ema_len"]
                )
                last = get_last_signal(s["chat_id"], s["ticker"], s["interval"])
                # Yalnızca değişim olduğunda mesaj at
                if signal != "NONE" and signal != last:
                    text = make_signal_text(signal, info)
                    await app.bot.send_message(chat_id=s["chat_id"], text=text)
                    set_last_signal(s["chat_id"], s["ticker"], s["interval"], signal, info["time"])
                elif signal == "NONE" and last not in (None, "NONE"):
                    # İsterseniz 'NONE'a geçişleri de bildirin:
                    set_last_signal(s["chat_id"], s["ticker"], s["interval"], "NONE", info["time"])
            except Exception as e:
                # Sessizce geçebiliriz veya loglayabiliriz
                print("Scheduler error:", e)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

async def main():
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(CommandHandler("list", lst))
    application.add_handler(CommandHandler("signal", inline_signal))

    # Scheduler'ı background task olarak ekle
    application.post_init.append(lambda app: asyncio.create_task(scheduler(app)))

    await application.initialize()
    await application.start()
    print("Telegram bot started. Listening for commands...")
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    # Uygulamayı sonsuza kadar çalıştır
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
