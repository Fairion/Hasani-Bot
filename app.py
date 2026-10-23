import os
import sys
import io
import json
import time
import threading
import subprocess

# Use thread-safe headless Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Clone shiyu-coder/Kronos into /tmp on container startup
if not os.path.exists("/tmp/Kronos"):
    print("📦 Cloning shiyu-coder/Kronos repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/shiyu-coder/Kronos.git", "/tmp/Kronos"],
        check=True
    )
if "/tmp/Kronos" not in sys.path:
    sys.path.insert(0, "/tmp/Kronos")

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf
import gradio as gr
from google import genai
from model import Kronos, KronosTokenizer, KronosPredictor

# ==============================================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==============================================================================
DEFAULT_INTERVAL   = "15m"
LOOKBACK           = 512
PRED_LEN           = 24
SAMPLE_COUNT       = 100

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "").strip()

MODEL_LOCK = threading.Lock()
BOT_STATUS = "⏳ Initializing Kronos-small..."

# ==============================================================================
# LOAD & CACHE KRONOS-SMALL ONCE AT STARTUP
# ==============================================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"⚡ Loading NeoQuasar/Kronos-small on {DEVICE}...")
TOKENIZER = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
MODEL = Kronos.from_pretrained("NeoQuasar/Kronos-small")
PREDICTOR = KronosPredictor(MODEL, TOKENIZER, device=DEVICE, max_context=512)
print("✅ Kronos-small loaded and cached!")


# ==============================================================================
# CORE FORECASTING, BACKTEST & VERIFIED STOP-LOSS ENGINE
# ==============================================================================
def fetch_candles(symbol: str, interval: str = "15m", lookback: int = 512, source: str = "stocks") -> pd.DataFrame:
    symbol = symbol.strip().upper()
    if source == "binance":
        endpoints = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
            "https://api.binance.us/api/v3/klines",
        ]
        data = None
        for url in endpoints:
            try:
                resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": lookback}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    break
            except Exception:
                continue
        if data is None:
            raise RuntimeError(f"Could not fetch {symbol} from Binance endpoints.")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["amount"] = df["quote_volume"]
        return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    else:
        yf_interval_map = {
            "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "60m", "1d": "1d"
        }
        yf_int = yf_interval_map.get(interval, "15m")
        period = "60d" if yf_int in ["2m", "5m", "15m", "30m", "60m"] else ("7d" if yf_int == "1m" else "2y")
        df = yf.Ticker(symbol).history(period=period, interval=yf_int)
        if df.empty or len(df) < 10:
            raise ValueError(f"Not enough stock data returned for '{symbol}' on {interval}.")
        df = df.tail(lookback).reset_index()
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df["timestamps"] = pd.to_datetime(df[time_col]).dt.tz_localize(None)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["amount"] = df["close"] * df["volume"]
        return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def run_kronos_forecast(df: pd.DataFrame, pred_len: int = 24, sample_count: int = 100):
    time_delta = df["timestamps"].diff().median()
    if pd.isna(time_delta) or time_delta.total_seconds() <= 0:
        time_delta = pd.Timedelta(minutes=15)
    future_timestamps = pd.Series([df["timestamps"].iloc[-1] + time_delta * (i + 1) for i in range(pred_len)])

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df["timestamps"]

    all_paths_close = []
    with MODEL_LOCK:
        try:
            batch_size = 25
            remaining = sample_count
            while remaining > 0:
                cur_batch = min(batch_size, remaining)
                preds = PREDICTOR.predict_batch(
                    df_list=[x_df.copy() for _ in range(cur_batch)],
                    x_timestamp_list=[x_timestamp.copy() for _ in range(cur_batch)],
                    y_timestamp_list=[future_timestamps.copy() for _ in range(cur_batch)],
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )
                for p_df in preds:
                    all_paths_close.append(p_df["close"].values)
                remaining -= cur_batch
        except Exception:
            all_paths_close = []
            for _ in range(sample_count):
                pred_df = PREDICTOR.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=future_timestamps,
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )
                all_paths_close.append(pred_df["close"].values)

    return future_timestamps, np.array(all_paths_close)


def analyze_paths(df: pd.DataFrame, paths_matrix: np.ndarray) -> dict:
    last_close = float(df["close"].iloc[-1])
    final_closes = paths_matrix[:, -1]
    paths_up = int(np.sum(final_closes > last_close))
    total_paths = paths_matrix.shape[0]
    up_ratio = paths_up / total_paths

    median_path = np.median(paths_matrix, axis=0)
    p10_path = np.percentile(paths_matrix, 10, axis=0)
    p90_path = np.percentile(paths_matrix, 90, axis=0)

    if up_ratio >= 0.65:
        regime = "TRENDING UP"
    elif up_ratio <= 0.35:
        regime = "TRENDING DOWN"
    else:
        regime = "RANGING"

    return {
        "last_close": last_close,
        "paths_up": paths_up,
        "total_paths": total_paths,
        "regime": regime,
        "median_final": float(median_path[-1]),
        "p10_final": float(p10_path[-1]),
        "p90_final": float(p90_path[-1]),
        "dispersion_pct": ((float(p90_path[-1]) - float(p10_path[-1])) / last_close) * 100.0,
        "median_path": median_path.tolist(),
        "p10_path": p10_path.tolist(),
        "p90_path": p90_path.tolist(),
    }


def run_backtest_with_fees(symbol: str, interval: str, source: str, fee_bps: float = 10.0, slippage_bps: float = 5.0) -> dict:
    df_hist = fetch_candles(symbol, interval=interval, lookback=1000, source=source)
    closes = df_hist["close"].values
    round_trip_cost = 2.0 * ((fee_bps + slippage_bps) / 10000.0)
    trades = []

    for i in range(50, len(closes) - 24, 24):
        window = closes[i - 50:i]
        momentum = (window[-1] - window[0]) / window[0]
        volatility = np.std(np.diff(window) / window[:-1])
        if volatility > 0 and abs(momentum) > volatility * 3.5:
            direction = 1 if momentum > 0 else -1
            gross_ret = direction * ((closes[i + 24] - closes[i]) / closes[i])
            trades.append(gross_ret - round_trip_cost)

    if not trades:
        return {"trades": 0, "win_rate_pct": 0.0, "net_return_pct": 0.0}
    trades_arr = np.array(trades)
    return {
        "trades": len(trades),
        "win_rate_pct": round(float(np.mean(trades_arr > 0) * 100.0), 1),
        "net_return_pct": round(float((np.prod(1.0 + trades_arr) - 1.0) * 100.0), 2),
    }


def build_verified_trade_plan(symbol: str, interval: str, df: pd.DataFrame, stats: dict, backtest: dict, api_key: str) -> dict:
    last_c = float(stats["last_close"])
    p10_path_min = float(np.min(stats["p10_path"]))
    p90_path_max = float(np.max(stats["p90_path"]))

    recent = df.tail(14)
    atr = float((recent["high"] - recent["low"]).mean())
    if atr <= 0 or np.isnan(atr):
        atr = last_c * 0.015

    recent_swing_low = float(df["low"].tail(5).min())
    recent_swing_high = float(df["high"].tail(5).max())

    if stats["regime"] == "TRENDING DOWN":
        action = "WATCH SHORT"
        default_entry = round(last_c * 0.999, 2)
        raw_stop = max(p90_path_max, recent_swing_high + 0.5 * atr, default_entry + 1.5 * atr)
        default_stop = round(raw_stop, 2)
        risk = default_stop - default_entry
        default_target = round(min(float(stats["p10_final"]), default_entry - 2.0 * risk), 2)
    else:
        action = "WATCH LONG" if stats["regime"] == "TRENDING UP" else "WAIT (RANGING)"
        default_entry = round(last_c * 1.001, 2)
        candidates_below = [
            val for val in [p10_path_min, recent_swing_low - 0.5 * atr, default_entry - 1.5 * atr]
            if val < default_entry - 0.5 * atr
        ]
        raw_stop = min(candidates_below) if candidates_below else (default_entry - 1.5 * atr)
        default_stop = round(raw_stop, 2)
        risk = default_entry - default_stop
        default_target = round(max(float(stats["p90_final"]), default_entry + 2.0 * risk), 2)

    plan = {"action": action, "entry": default_entry, "stop_loss": default_stop, "target": default_target}

    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            You are a quantitative trading engine analyzing Kronos-small (100 Monte Carlo candlestick futures).
            Asset: {symbol} ({interval}) | Last Close: {last_c} | ATR(14): {atr:.4f}
            Regime: {stats['regime']} ({stats['paths_up']}/{stats['total_paths']} paths up)
            Lowest p10 dip across 24 bars: {p10_path_min:.2f} | Highest p90 peak: {p90_path_max:.2f}
            24th-Bar Forecast: Median={stats['median_final']:.2f}, p10={stats['p10_final']:.2f}, p90={stats['p90_final']:.2f}
            Baseline Validated Levels: Entry={default_entry}, Stop={default_stop}, Target={default_target}

            RULES:
            - If Regime is TRENDING UP or RANGING: stop_loss MUST be strictly LESS than entry, and target MUST be strictly GREATER than entry.
            - If Regime is TRENDING DOWN: stop_loss MUST be strictly GREATER than entry, and target MUST be strictly LESS than entry.
            Return ONLY valid JSON:
            {{
              "action": "WATCH LONG" | "WATCH SHORT" | "WAIT (RANGING)",
              "entry": <number>,
              "stop_loss": <number>,
              "target": <number>
            }}
            """
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            ai_plan = json.loads(resp.text)
            plan["entry"] = round(float(ai_plan.get("entry", default_entry)), 2)
            plan["stop_loss"] = round(float(ai_plan.get("stop_loss", default_stop)), 2)
            plan["target"] = round(float(ai_plan.get("target", default_target)), 2)
        except Exception:
            pass

    if stats["regime"] == "TRENDING DOWN":
        if plan["stop_loss"] <= plan["entry"]:
            plan["stop_loss"] = round(max(default_stop, plan["entry"] + 1.5 * atr), 2)
        if plan["target"] >= plan["entry"]:
            plan["target"] = round(min(default_target, plan["entry"] - 2.0 * (plan["stop_loss"] - plan["entry"])), 2)
        risk_amt = abs(plan["stop_loss"] - plan["entry"])
        reward_amt = abs(plan["entry"] - plan["target"])
    else:
        if plan["stop_loss"] >= plan["entry"]:
            plan["stop_loss"] = round(min(default_stop, plan["entry"] - 1.5 * atr), 2)
        if plan["target"] <= plan["entry"]:
            plan["target"] = round(max(default_target, plan["entry"] + 2.0 * (plan["entry"] - plan["stop_loss"])), 2)
        risk_amt = abs(plan["entry"] - plan["stop_loss"])
        reward_amt = abs(plan["target"] - plan["entry"])

    rr_ratio = round(reward_amt / risk_amt, 2) if risk_amt > 0 else 0.0
    plan["rr_ratio"] = rr_ratio
    plan["summary"] = (
        f"{plan['action']} near {plan['entry']:,.2f}, "
        f"stop {plan['stop_loss']:,.2f}, "
        f"target {plan['target']:,.2f} (R:R 1:{rr_ratio}). Mode: PAPER."
    )
    return plan


def create_telegram_chart_png(symbol: str, interval: str, df: pd.DataFrame, paths_matrix: np.ndarray, stats: dict, plan: dict) -> io.BytesIO:
    with MODEL_LOCK:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        fig.patch.set_facecolor("#0b0e14")
        ax.set_facecolor("#0b0e14")

        hist_closes = df["close"].tail(80).values
        hist_x = np.arange(len(hist_closes))
        fut_x = np.arange(len(hist_closes) - 1, len(hist_closes) - 1 + paths_matrix.shape[1] + 1)

        ax.plot(hist_x, hist_closes, color="#10b981", linewidth=2, label="Historical Close")

        last_close = hist_closes[-1]
        for path in paths_matrix:
            ax.plot(fut_x, np.concatenate([[last_close], path]), color="#94a3b8", alpha=0.08, linewidth=0.8)

        ax.plot(fut_x, np.concatenate([[last_close], stats["p90_path"]]), color="#38bdf8", linestyle="--", linewidth=1.8, label="p90 (Upper)")
        ax.plot(fut_x, np.concatenate([[last_close], stats["median_path"]]), color="#facc15", linewidth=2.4, label="Median Forecast")
        ax.plot(fut_x, np.concatenate([[last_close], stats["p10_path"]]), color="#f87171", linestyle="--", linewidth=1.8, label="p10 (Lower)")

        ax.hlines(plan["stop_loss"], fut_x[0], fut_x[-1], colors="#ef4444", linestyles=":", linewidth=1.5, label=f"Stop ({plan['stop_loss']:,.2f})")
        ax.hlines(plan["target"], fut_x[0], fut_x[-1], colors="#22c55e", linestyles=":", linewidth=1.5, label=f"Target ({plan['target']:,.2f})")

        ax.set_title(f"KRONOS 100-PATH FORECAST: {symbol} ({interval}) — {stats['regime']}", color="white", fontsize=12, fontweight="bold")
        ax.grid(color="#1f2937", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=8, facecolor="#111827", edgecolor="#374151")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf


# ==============================================================================
# BACKGROUND 24/7 TELEGRAM BOT THREAD
# ==============================================================================
def telegram_bot_worker():
    global BOT_STATUS
    if not TELEGRAM_BOT_TOKEN:
        BOT_STATUS = "⚠️ TELEGRAM_BOT_TOKEN secret is not set."
        print(BOT_STATUS)
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        requests.get(f"{base_url}/deleteWebhook", timeout=10)
        me = requests.get(f"{base_url}/getMe", timeout=10).json()
        if not me.get("ok"):
            BOT_STATUS = f"❌ Invalid TELEGRAM_BOT_TOKEN: {me}"
            return
        bot_username = me["result"].get("username", "Bot")
    except Exception as e:
        BOT_STATUS = f"❌ Telegram connection error: {e}"
        return

    offset = None
    try:
        init_updates = requests.get(f"{base_url}/getUpdates", timeout=10).json()
        if init_updates.get("ok") and init_updates.get("result"):
            offset = init_updates["result"][-1]["update_id"] + 1
    except Exception:
        pass

    BOT_STATUS = f"🟢 Online 24/7 on Cloud (@{bot_username})"
    if TELEGRAM_CHAT_ID:
        requests.post(f"{base_url}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": (
                f"☁️ <b>@{bot_username} is now LIVE 24/7 on Cloud!</b>\n"
                f"Your PC can stay off. Text any ticker like <code>NVDA 15m</code> or <code>BTCUSDT 1h</code>."
            ),
            "parse_mode": "HTML"
        })

    valid_stock_tf = {"1m", "2m", "5m", "15m", "30m", "1h", "1d"}
    valid_crypto_tf = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}

    while True:
        try:
            params = {"timeout": 20}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{base_url}/getUpdates", params=params, timeout=25).json()

            if not resp.get("ok"):
                time.sleep(4)
                continue

            for item in resp.get("result", []):
                offset = item["update_id"] + 1
                msg = item.get("message") or item.get("edited_message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = msg.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue

                if TELEGRAM_CHAT_ID and str(chat_id) != str(TELEGRAM_CHAT_ID).strip():
                    continue

                if text.lower() in ["/start", "help", "/help"]:
                    requests.post(f"{base_url}/sendMessage", json={
                        "chat_id": chat_id,
                        "text": "📈 Send a ticker + timeframe (e.g. <code>NVDA 5m</code>, <code>SHFS 15m</code>, <code>BTCUSDT 15m</code>).",
                        "parse_mode": "HTML"
                    })
                    continue

                parts = text.upper().split()
                req_symbol = parts[0].replace("/", "").lstrip("$")
                raw_tf = parts[1].lower() if len(parts) > 1 else DEFAULT_INTERVAL
                req_source = "binance" if req_symbol.endswith(("USDT", "USDC", "BUSD")) else "stocks"

                tf_note = ""
                if req_source == "stocks":
                    if raw_tf == "3m":
                        req_interval = "5m"
                        tf_note = " <i>(Yahoo Finance doesn't have 3m stock bars — switched to 5m)</i>"
                    elif raw_tf in valid_stock_tf:
                        req_interval = raw_tf
                    else:
                        req_interval = DEFAULT_INTERVAL
                else:
                    req_interval = raw_tf if raw_tf in valid_crypto_tf else DEFAULT_INTERVAL

                requests.post(f"{base_url}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": f"⏳ Running 100 Kronos futures on <b>{req_symbol}</b> (<code>{req_interval}</code>)...{tf_note}",
                    "parse_mode": "HTML"
                })

                try:
                    df = fetch_candles(req_symbol, req_interval, LOOKBACK, req_source)
                    _, paths_matrix = run_kronos_forecast(df, PRED_LEN, SAMPLE_COUNT)
                    stats = analyze_paths(df, paths_matrix)
                    backtest = run_backtest_with_fees(req_symbol, req_interval, req_source)
                    plan = build_verified_trade_plan(req_symbol, req_interval, df, stats, backtest, GEMINI_API_KEY)

                    p10_min = float(np.min(stats["p10_path"]))
                    caption = (
                        f"📊 <b>KRONOS + GEMINI: {req_symbol} ({req_interval})</b>\n"
                        f"• <b>Last Close:</b> <code>{stats['last_close']:,.2f}</code>\n"
                        f"• <b>Regime:</b> <code>{stats['regime']}</code> ({stats['paths_up']}/{stats['total_paths']} paths up)\n"
                        f"• <b>Median (+{PRED_LEN} bars):</b> <code>{stats['median_final']:,.2f}</code>\n"
                        f"• <b>24-Bar p10 – p90:</b> <code>{stats['p10_final']:,.2f}</code> – <code>{stats['p90_final']:,.2f}</code>\n"
                        f"• <b>Max Path Pullback (p10 low):</b> <code>{p10_min:,.2f}</code>\n"
                        f"• <b>Backtest (net fees):</b> <code>{backtest['win_rate_pct']}% win</code> (<code>{backtest['net_return_pct']}% net</code>)\n\n"
                        f"🎯 <b>Plan:</b> {plan['summary']}\n"
                        f"<code>Entry: {plan['entry']:,.2f} | Stop: {plan['stop_loss']:,.2f} | Target: {plan['target']:,.2f}</code>"
                    )

                    chart_buf = create_telegram_chart_png(req_symbol, req_interval, df, paths_matrix, stats, plan)
                    requests.post(
                        f"{base_url}/sendPhoto",
                        data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                        files={"photo": ("forecast.png", chart_buf, "image/png")},
                        timeout=30
                    )
                except Exception as err:
                    requests.post(f"{base_url}/sendMessage", json={
                        "chat_id": chat_id,
                        "text": f"❌ Could not analyze <code>{req_symbol}</code> ({req_interval}): {err}",
                        "parse_mode": "HTML"
                    })

        except Exception:
            time.sleep(4)

# Start background Telegram listener
threading.Thread(target=telegram_bot_worker, daemon=True).start()

# Gradio status page
def get_status():
    return f"**Bot Status:** {BOT_STATUS}  \n**Compute Device:** `{DEVICE}` | **Samples:** `{SAMPLE_COUNT}`"

with gr.Blocks(title="Kronos Telegram Bot") as demo:
    gr.Markdown("## 📈 Kronos + Gemini Cloud Telegram Bot")
    status_md = gr.Markdown(get_status)
    refresh_btn = gr.Button("Refresh Status")
    refresh_btn.click(fn=get_status, outputs=status_md)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
