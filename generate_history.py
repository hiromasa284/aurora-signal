import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Aurora Signal と同じロジックを再利用する ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def detect_signal(rsi):
    if rsi > 70:
        return "SELL"
    elif rsi < 30:
        return "BUY"
    else:
        return "HOLD"

# --- 過去データから signal_history.json を作る ---
def generate_signal_history(tickers, start="2020-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    history = []

    for ticker in tickers:
        print(f"[処理中] {ticker}")

        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"  → データなし")
            continue

        df["RSI"] = calculate_rsi(df["Close"])
        df["Signal"] = df["RSI"].apply(detect_signal)

        # 翌日の終値をずらして追加
        df["Next_Close"] = df["Close"].shift(-1)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            if row["Signal"] == "HOLD":
                continue

            entry_price = row["Close"]
            next_price = row["Next_Close"]

            if pd.isna(next_price):
                continue

            # BUY/SELL の勝敗判定
            if row["Signal"] == "BUY":
                result = "WIN" if next_price > entry_price else "LOSE"
            else:
                result = "WIN" if next_price < entry_price else "LOSE"

            history.append({
                "ticker": ticker,
                "date": str(row.name.date()),
                "signal": row["Signal"],
                "entry_price": float(entry_price),
                "next_price": float(next_price),
                "result": result
            })

    # JSON 保存
    with open("signal_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("\n=== signal_history.json を生成しました ===")
    print(f"総件数: {len(history)} 件")

# --- 実行例 ---
if __name__ == "__main__":
    tickers = [
        "6758.T", "9984.T", "7203.T", "9432.T",
        "AAPL", "MSFT", "GOOGL", "AMZN"
    ]
    generate_signal_history(tickers)
