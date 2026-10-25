import os
# Limit glibc memory arenas BEFORE importing numpy/torch to prevent RAM fragmentation on Render 512MB
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import io
import gc
import re
import json
import time
import ctypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

# ==============================================================================
# CONFIGURATION & MEMORY HELPERS
# ==============================================================================
def clean_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip().strip('"').strip("'")

DEFAULT_INTERVAL    = "15m"
LOOKBACK            = int(clean_env("LOOKBACK", "160"))
PRED_LEN            = int(clean_env("PRED_LEN", "24"))
SAMPLE_COUNT        = int(clean_env("SAMPLE_COUNT", "25"))

TELEGRAM_BOT_TOKEN  = clean_env("TELEGRAM_BOT_TOKEN")
RAW_CHAT_ID         = clean_env("TELEGRAM_CHAT_ID")
CLEAN_CHAT_ID_MATCH = re.search(r"-?\d+", RAW_CHAT_ID)
TELEGRAM_CHAT_ID    = CLEAN_CHAT_ID_MATCH.group(0) if CLEAN_CHAT_ID_MATCH else ""

GEMINI_API_KEY      = clean_env("GEMINI_API_KEY")
RENDER_EXTERNAL_URL = clean_env("RENDER_EXTERNAL_URL").rstrip("/")

LOAD_LOCK         = threading.Lock()
INFER_LOCK        = threading.Lock()
PREDICTOR         = None
PROCESSED_UPDATES = set()

BOT_STATUS   = "⏳ Connecting to Telegram..."
MODEL_STATUS = "💤 Standby (loads on first forecast)"
LAST_EVENT   = "None yet"


def free_memory():
    """Forces Python GC and returns unused Linux heap pages back to OS (<512 MB guard)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_ram_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return 0.0


# ==============================================================================
# 1. CPU-OPTIMIZED KRONOS-MINI LOADER (~220 MB Total Process RAM)
# ==============================================================================
def ensure_kronos_loaded():
    global PREDICTOR, MODEL_STATUS
    if PREDICTOR is not None:
        return PREDICTOR

    with LOAD_LOCK:
        if PREDICTOR is not None:
            return PREDICTOR

        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        pkg_root = "/tmp/kronos_pkg"
        model_dir = os.path.join(pkg_root, "model")
        os.makedirs(model_dir, exist_ok=True)

        MODEL_STATUS = "📦 Downloading Kronos model scripts..."
        for fname in ["__init__.py", "module.py", "kronos.py"]:
            fpath = os.path.join(model_dir, fname)
            if not os.path.exists(fpath):
                fetched = False
                for branch in ["master", "main"]:
                    url = f"https://raw.githubusercontent.com/shiyu-coder/Kronos/{branch}/model/{fname}"
                    r = requests.get(url, timeout=20)
                    if r.status_code == 200:
                        with open(fpath, "wb") as f:
                            f.write(r.content)
                        fetched = True
                        break
                if not fetched:
                    raise RuntimeError(f"Failed to download Kronos {fname}.")

        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)

        from model import Kronos, KronosTokenizer, KronosPredictor

        MODEL_STATUS = f"⚡ Loading Kronos-mini on CPU (torch {torch.__version__})..."
        print(f"{MODEL_STATUS} | RAM before load: {get_ram_mb()} MB", flush=True)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        model.eval()
        PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        free_memory()

        MODEL_STATUS = f"✅ Kronos-mini Ready (RAM: {get_ram_mb()} MB / 512 MB)"
        print(MODEL_STATUS, flush=True)
        return PREDICTOR


# ==============================================================================
# 2. ZERO-OVERHEAD YAHOO FINANCE + BINANCE FETCHER (Supports 4h Stocks!)
# ==============================================================================
def fetch_candles(symbol: str, interval: str = "15m", lookback: int = 160, source: str = "stocks"):
    import numpy as np
    import pandas as pd

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

    # Direct Yahoo Finance v8 JSON API (saves ~100 MB RAM vs importing yfinance)
    resample_4h = (interval == "4h")
    yf_interval_map = {
        "1m": ("1m", "7d"),
        "2m": ("2m", "60d"),
        "5m": ("5m", "60d"),
        "15m": ("15m", "60d"),
        "30m": ("30m", "60d"),
        "1h": ("60m", "60d"),
        "4h": ("60m", "1y"),   # Fetch 1h bars over 1 year and aggregate into 4h bars
        "1d": ("1d", "2y"),
    }
    yf_int, yf_range = yf_interval_map.get(interval, ("15m", "60d"))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    payload = None
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            url = f"https://{host}/v8/finance/chart/{symbol}"
            r = requests.get(url, params={"interval": yf_int, "range": yf_range}, headers=headers, timeout=12)
            if r.status_code == 200:
                payload = r.json()
                break
        except Exception:
            continue

    if not payload or not payload.get("chart", {}).get("result"):
        raise ValueError(f"Ticker '{symbol}' not found on Yahoo Finance.")

    res = payload["chart"]["result"][0]
    ts_list = res.get("timestamp")
    quote = res.get("indicators", {}).get("quote", [{}])[0]
    if not ts_list or not quote.get("close"):
        raise ValueError(f"No candle data returned for '{symbol}' ({interval}).")

    df = pd.DataFrame({
        "timestamps": pd.to_datetime(ts_list, unit="s"),
        "open": pd.to_numeric(quote.get("open"), errors="coerce"),
        "high": pd.to_numeric(quote.get("high"), errors="coerce"),
        "low": pd.to_numeric(quote.get("low"), errors="coerce"),
        "close": pd.to_numeric(quote.get("close"), errors="coerce"),
        "volume": pd.to_numeric(quote.get("volume"), errors="coerce").fillna(0.0),
    }).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if resample_4h and len(df) >= 40:
        df["grp"] = np.arange(len(df)) // 4
        df = df.groupby("grp", as_index=False).agg({
            "timestamps": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

    if len(df) < 10:
        raise ValueError(f"Not enough historical bars ({len(df)}) for '{symbol}' on {interval}.")

    df = df.tail(lookback).reset_index(drop=True)
    df["amount"] = df["close"] * df["volume"]
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


# ==============================================================================
# 3. KRONOS FORECAST, BACKTEST & HARD STOP-LOSS GUARDRAIL
# ==============================================================================
def run_kronos_forecast(df, pred_len: int = 24, sample_count: int = 25):
    import numpy as np
    import pandas as pd
    import torch

    predictor = ensure_kronos_loaded()

    time_delta = df["timestamps"].diff().median()
    if pd.isna(time_delta) or time_delta.total_seconds() <= 0:
        time_delta = pd.Timedelta(minutes=15)
    future_timestamps = pd.Series([df["timestamps"].iloc[-1] + time_delta * (i + 1) for i in range(pred_len)])

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df["timestamps"]

    all_paths_close = []
    with INFER_LOCK, torch.inference_mode():
        try:
            # Tiny batch size (5) keeps CPU activation RAM < 15 MB
            batch_size = 5
            remaining = sample_count
            while remaining > 0:
                cur_batch = min(batch_size, remaining)
                preds = predictor.predict_batch(
                    df_list=[x_df for _ in range(cur_batch)],
                    x_timestamp_list=[x_timestamp for _ in range(cur_batch)],
                    y_timestamp_list=[future_timestamps for _ in range(cur_batch)],
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
                pred_df = predictor.predict(
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

    free_memory()
    return future_timestamps, np.array(all_paths_close)


def analyze_paths(df, paths_matrix) -> dict:
    import numpy as np
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
    import numpy as np
    df_hist = fetch_candles(symbol, interval=interval, lookback=500, source=source)
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


def build_verified_trade_plan(symbol: str, interval: str, df, stats: dict, backtest: dict, api_key: str) -> dict:
    import numpy as np
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
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"""
            You are a quantitative trading engine analyzing Kronos ({stats['total_paths']} Monte Carlo candlestick futures).
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
            for model_id in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
                try:
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    ai_plan = json.loads(resp.text)
                    plan["entry"] = round(float(ai_plan.get("entry", default_entry)), 2)
                    plan["stop_loss"] = round(float(ai_plan.get("stop_loss", default_stop)), 2)
                    plan["target"] = round(float(ai_plan.get("target", default_target)), 2)
                    break
                except Exception:
                    continue
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


# ==============================================================================
# 4. ULTRA-LIGHTWEIGHT PILLOW CHART RENDERER (~6 MB RAM vs 85 MB Matplotlib)
# ==============================================================================
def create_telegram_chart_png(symbol: str, interval: str, df, paths_matrix, stats: dict, plan: dict) -> io.BytesIO:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1000, 520
    PAD_L, PAD_R, PAD_T, PAD_B = 28, 92, 54, 40
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    base = Image.new("RGBA", (W, H), (11, 14, 20, 255))
    draw = ImageDraw.Draw(base)

    try:
        font_title = ImageFont.load_default(size=17)
        font_sm = ImageFont.load_default(size=13)
    except TypeError:
        font_title = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    hist_closes = df["close"].tail(80).values
    last_close = float(hist_closes[-1])
    total_pts = len(hist_closes) + paths_matrix.shape[1]

    all_vals = np.concatenate([
        hist_closes,
        paths_matrix.flatten(),
        np.array([plan["stop_loss"], plan["target"]])
    ])
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad_y = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_min -= pad_y
    y_max += pad_y

    def to_xy(idx: int, val: float):
        x = PAD_L + (idx / max(total_pts - 1, 1)) * plot_w
        y = PAD_T + (1.0 - (val - y_min) / (y_max - y_min)) * plot_h
        return (round(x, 1), round(y, 1))

    # Grid & right-side price scale
    for i in range(6):
        frac = i / 5.0
        gy = round(PAD_T + frac * plot_h)
        price_val = y_max - frac * (y_max - y_min)
        draw.line([(PAD_L, gy), (W - PAD_R, gy)], fill=(31, 41, 55, 255), width=1)
        draw.text((W - PAD_R + 8, gy - 7), f"{price_val:,.2f}", fill=(156, 163, 175, 255), font=font_sm)

    split_x, _ = to_xy(len(hist_closes) - 1, last_close)
    draw.line([(split_x, PAD_T), (split_x, H - PAD_B)], fill=(55, 65, 81, 255), width=1)

    # Semi-transparent Monte Carlo fan paths
    fan_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fan_draw = ImageDraw.Draw(fan_layer)
    start_idx = len(hist_closes) - 1
    for path in paths_matrix:
        full_p = np.concatenate([[last_close], path])
        pts = [to_xy(start_idx + j, float(v)) for j, v in enumerate(full_p)]
        fan_draw.line(pts, fill=(148, 163, 184, 42), width=2)
    base = Image.alpha_composite(base, fan_layer)
    draw = ImageDraw.Draw(base)

    # Historical Close, p90, Median, p10
    hist_pts = [to_xy(i, float(v)) for i, v in enumerate(hist_closes)]
    draw.line(hist_pts, fill=(16, 185, 129, 255), width=3)

    p90_pts = [to_xy(start_idx + j, float(v)) for j, v in enumerate([last_close] + stats["p90_path"])]
    med_pts = [to_xy(start_idx + j, float(v)) for j, v in enumerate([last_close] + stats["median_path"])]
    p10_pts = [to_xy(start_idx + j, float(v)) for j, v in enumerate([last_close] + stats["p10_path"])]

    draw.line(p90_pts, fill=(56, 189, 248, 255), width=2)
    draw.line(p10_pts, fill=(248, 113, 113, 255), width=2)
    draw.line(med_pts, fill=(250, 204, 21, 255), width=3)

    # Stop & Target dashed lines
    for level, color in [(plan["stop_loss"], (239, 68, 68, 255)), (plan["target"], (34, 197, 94, 255))]:
        _, ly = to_xy(start_idx, level)
        for seg_x in range(int(split_x), W - PAD_R, 12):
            draw.line([(seg_x, ly), (min(seg_x + 6, W - PAD_R), ly)], fill=color, width=2)

    # Header & Legend
    draw.text((PAD_L, 16), f"KRONOS FORECAST: {symbol} ({interval})  |  {stats['regime']} ({stats['paths_up']}/{stats['total_paths']} up)", fill=(255, 255, 255, 255), font=font_title)
    legend_box = (PAD_L + 8, PAD_T + 8, PAD_L + 225, PAD_T + 108)
    draw.rectangle(legend_box, fill=(17, 24, 39, 220), outline=(55, 65, 81, 255))
    legend_items = [
        ("Historical Close", (16, 185, 129, 255)),
        ("Median Forecast", (250, 204, 21, 255)),
        ("p90 / p10 Range", (56, 189, 248, 255)),
        (f"Target: {plan['target']:,.2f}", (34, 197, 94, 255)),
        (f"Stop: {plan['stop_loss']:,.2f}", (239, 68, 68, 255)),
    ]
    for idx, (label, col) in enumerate(legend_items):
        ly = PAD_T + 14 + idx * 18
        draw.line([(PAD_L + 16, ly + 7), (PAD_L + 34, ly + 7)], fill=col, width=3)
        draw.text((PAD_L + 42, ly), label, fill=(229, 231, 235, 255), font=font_sm)

    buf = io.BytesIO()
    base.convert("RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    free_memory()
    return buf


# ==============================================================================
# 5. TELEGRAM MESSAGE HANDLER & WEBHOOK SERVER
# ==============================================================================
def process_telegram_update(item: dict):
    global LAST_EVENT
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    msg = item.get("message") or item.get("edited_message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = msg.get("chat", {}).get("id")

    if not text or not chat_id:
        return

    LAST_EVENT = f"'{text}' from {chat_id} (RAM: {get_ram_mb()} MB)"
    print(f"📩 {LAST_EVENT}", flush=True)

    if text.lower() in ["/", "/start", "start", "/help", "help", "/ping", "ping"]:
        requests.post(f"{base_url}/sendMessage", json={
            "chat_id": chat_id,
            "text": (
                f"🟢 <b>Hasani-Bot is Awake & Memory-Optimized!</b>\n"
                f"• <b>RAM Usage:</b> <code>{get_ram_mb()} MB / 512 MB</code>\n"
                f"• <b>Try sending:</b> <code>NVDA 4h</code>, <code>NVDA 15m</code>, <code>SHFS 15m</code>, or <code>BTCUSDT 1h</code>"
            ),
            "parse_mode": "HTML"
        }, timeout=15)
        return

    valid_stock_tf = {"1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"}
    valid_crypto_tf = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}

    parts = text.upper().split()
    req_symbol = parts[0].replace("/", "").lstrip("$")
    raw_tf = parts[1].lower() if len(parts) > 1 else DEFAULT_INTERVAL
    req_source = "binance" if req_symbol.endswith(("USDT", "USDC", "BUSD")) else "stocks"

    tf_note = ""
    if req_source == "stocks":
        if raw_tf == "3m":
            req_interval = "5m"
            tf_note = " <i>(switched 3m &rarr; 5m)</i>"
        elif raw_tf in valid_stock_tf:
            req_interval = raw_tf
        else:
            req_interval = DEFAULT_INTERVAL
    else:
        req_interval = raw_tf if raw_tf in valid_crypto_tf else DEFAULT_INTERVAL

    step_resp = requests.post(f"{base_url}/sendMessage", json={
        "chat_id": chat_id,
        "text": f"⏳ <b>[1/3]</b> Fetching candles for <b>{req_symbol}</b> (<code>{req_interval}</code>)...{tf_note}",
        "parse_mode": "HTML"
    }, timeout=15).json()
    status_msg_id = step_resp.get("result", {}).get("message_id")

    def update_progress(html_text: str):
        if status_msg_id:
            try:
                requests.post(f"{base_url}/editMessageText", json={
                    "chat_id": chat_id,
                    "message_id": status_msg_id,
                    "text": html_text,
                    "parse_mode": "HTML"
                }, timeout=10)
            except Exception:
                pass

    try:
        import numpy as np
        df = fetch_candles(req_symbol, req_interval, LOOKBACK, req_source)

        cold_hint = " <i>(loading CPU model ~12s)</i>" if PREDICTOR is None else ""
        update_progress(f"🧠 <b>[2/3]</b> Running {SAMPLE_COUNT} Kronos futures on <b>{req_symbol}</b> (<code>{req_interval}</code>)...{cold_hint}")

        _, paths_matrix = run_kronos_forecast(df, PRED_LEN, SAMPLE_COUNT)
        stats = analyze_paths(df, paths_matrix)

        update_progress(f"📊 <b>[3/3]</b> Building Gemini trade plan & chart for <b>{req_symbol}</b>...")
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

        if status_msg_id:
            requests.post(f"{base_url}/deleteMessage", json={"chat_id": chat_id, "message_id": status_msg_id}, timeout=10)

        LAST_EVENT = f"✅ Sent {req_symbol} ({req_interval}) | RAM: {get_ram_mb()} MB / 512 MB"
        print(LAST_EVENT, flush=True)

    except Exception as err:
        LAST_EVENT = f"❌ Error on {req_symbol}: {err}"
        print(traceback.format_exc(), flush=True)
        requests.post(f"{base_url}/sendMessage", json={
            "chat_id": chat_id,
            "text": f"❌ Could not analyze <code>{req_symbol}</code> ({req_interval}): {err}",
            "parse_mode": "HTML"
        }, timeout=15)


class BotHTTPHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = (
            "<html><body style='background:#0b0e14;color:#e5e7eb;font-family:monospace;padding:24px;line-height:1.8;'>"
            "<h2>📈 Hasani-Bot Live Diagnostic Page</h2>"
            f"<div><b>Telegram Status:</b> {BOT_STATUS}</div>"
            f"<div><b>Kronos Status:</b> {MODEL_STATUS}</div>"
            f"<div><b>Live RAM Usage:</b> {get_ram_mb()} MB / 512 MB</div>"
            f"<div><b>Last Event:</b> {LAST_EVENT}</div>"
            "</body></html>"
        )
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/webhook"):
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

            try:
                update = json.loads(raw_body.decode("utf-8"))
                uid = update.get("update_id")
                if uid is not None:
                    if uid in PROCESSED_UPDATES:
                        return
                    PROCESSED_UPDATES.add(uid)
                    if len(PROCESSED_UPDATES) > 500:
                        PROCESSED_UPDATES.clear()
                threading.Thread(target=process_telegram_update, args=(update,), daemon=True).start()
            except Exception as e:
                print(f"⚠️ Webhook parse error: {e}", flush=True)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def startup_telegram():
    global BOT_STATUS
    if not TELEGRAM_BOT_TOKEN:
        BOT_STATUS = "❌ TELEGRAM_BOT_TOKEN is missing!"
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        me = requests.get(f"{base_url}/getMe", timeout=15).json()
        if not me.get("ok"):
            BOT_STATUS = f"❌ Invalid TELEGRAM_BOT_TOKEN: {me.get('description')}"
            return
        bot_username = me["result"].get("username", "Bot")
    except Exception as e:
        BOT_STATUS = f"❌ Telegram connection error: {e}"
        return

    webhook_url = f"{RENDER_EXTERNAL_URL or 'https://hasani-bot.onrender.com'}/webhook"
    wh = requests.post(f"{base_url}/setWebhook", json={"url": webhook_url}, timeout=15).json()
    BOT_STATUS = f"🟢 Webhook Active on @{bot_username}" if wh.get("ok") else f"⚠️ Webhook error: {wh}"

    if TELEGRAM_CHAT_ID:
        requests.post(f"{base_url}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": (
                f"☁️ <b>@{bot_username} is LIVE (<250 MB RAM Mode)!</b>\n"
                f"Send <code>NVDA 4h</code> or <code>NVDA 15m</code> now to test."
            ),
            "parse_mode": "HTML"
        }, timeout=15)


if __name__ == "__main__":
    threading.Thread(target=startup_telegram, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Server listening on 0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), BotHTTPHandler).serve_forever()
