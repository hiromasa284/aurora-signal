import os
import json
import requests
import pandas as pd
from email.mime.text import MIMEText
from datetime import datetime
import smtplib

# 株価取得（Alpha Vantage）
def get_price(symbol):
    key = os.getenv("ALPHA_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}"
    r = requests.get(url).json()
    data = r.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df = df.astype(float)
    return df

# RSI計算
def calculate_rsi(data, window=14):
    delta = data["4. close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# シグナル判定
def check_signal(data):
    rsi = data["rsi"]
    price = data["close"]
    moving_avg = data.get("moving_avg", 150)  # 仮の移動平均値

    if rsi <= 30 and price < moving_avg:
        return "BUY"
    elif rsi >= 70 and price > moving_avg:
        return "SELL"
    else:
        return "HOLD"

# 勝率と期待値を計算
def calculate_expected_value(data):
    win_prob = 1 / data["rsi"]  # 仮の勝率（RSIが低いほど勝率が高いと仮定）
    expected_value = win_prob * data["close"]  # 仮の期待値計算
    return expected_value

# アラート結果をフィルタリング
def filter_alerts(alerts):
    return {ticker: info for ticker, info in alerts.items() if info["signal"] in ["BUY", "SELL"]}

# メール本文を整形
def format_alerts_for_email(signals):
    body = "以下は最新のアラート情報です：\n\n"
    for ticker, info in signals.items():
        body += f"銘柄: {ticker}\n"
        body += f"  シグナル: {info['signal']}\n"
        body += f"  RSI: {info['rsi']}\n"
        body += f"  価格: {info['close']}\n"
        body += f"  期待値: {info['expected_value']:.2f}\n"
        body += "-" * 20 + "\n"
    return body

# メール送信
def send_email(subject, body):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    send_to = os.getenv("SEND_TO", smtp_user)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = send_to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

# メインロジック
def main():
    TICKERS = ["AAPL", "MSFT", "TSLA"]
    signals = {}

    for ticker in TICKERS:
        # 株価データ取得
        price_data = get_price(ticker)
        latest_price = price_data.iloc[-1]["4. close"]
        rsi = calculate_rsi(price_data)
        moving_avg = price_data["4. close"].rolling(window=14).mean().iloc[-1]

        data = {
            "close": latest_price,
            "rsi": rsi,
            "moving_avg": moving_avg
        }

        # シグナル判定
        signal = check_signal(data)
        expected_value = calculate_expected_value(data)

        signals[ticker] = {
            "signal": signal,
            "rsi": rsi,
            "close": latest_price,
            "expected_value": expected_value
        }

    # 期待値でソート
    sorted_signals = sorted(signals.items(), key=lambda x: x[1]["expected_value"], reverse=True)
    top_signals = {k: v for k, v in sorted_signals[:3]}  # 上位3件を選択

    # フィルタリング（BUYとSELLのみ）
    filtered_signals = filter_alerts(top_signals)

    # メール本文を作成
    email_body = format_alerts_for_email(filtered_signals)

    # メール送信
    send_email("最新の株価アラート", email_body)

if __name__ == "__main__":
    main()
