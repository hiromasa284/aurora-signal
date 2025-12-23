import os
import json
import requests
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import smtplib

# CSV から銘柄リストを読み込む
def load_tickers():
    # 日本株はヘッダーなし
    jp = pd.read_csv("tickers_jp.csv", header=None)[0].dropna().tolist()

    # 米国株は symbol 列がある
    us = pd.read_csv("tickers_us.csv")["symbol"].dropna().tolist()

    # 重複除去して順番維持
    return list(dict.fromkeys(jp + us))

# 株価取得（Alpha Vantage）
def get_price(symbol):
    key = os.getenv("ALPHA_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}"
    r = requests.get(url).json()
    data = r.get("Time Series (Daily)", {})
    if not data:
        print(f"{symbol} のデータが取得できませんでした")
        return pd.DataFrame()
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
    moving_avg = data.get("moving_avg", 150)
    price_change = data.get("price_change", 0)

    if rsi <= 25 and price < moving_avg and price_change < -0.05:
        return "BUY"
    elif rsi >= 75 and price > moving_avg and price_change > 0.05:
        return "SELL"
    else:
        return "HOLD"

# 勝率と期待値
def calculate_expected_value(data):
    win_prob = 1 / data["rsi"]
    expected_value = win_prob * data["close"]
    return expected_value

# BUY/SELL のみ抽出
def filter_alerts(alerts):
    return {ticker: info for ticker, info in alerts.items() if info["signal"] in ["BUY", "SELL"]}

# メール本文整形
def format_alerts_for_email(signals):
    body = "以下は最新のアラート情報です：\n\n"
    for ticker, info in signals.items():
        body += f"銘柄: {ticker}\n"
        body += f"  シグナル: {info['signal']}\n"
        body += f"  RSI: {info['rsi']}\n"
        body += f"  価格: {info['close']}\n"
        body += f"  移動平均: {info.get('moving_avg', 'N/A')}\n"
        body += f"  期待値: {info['expected_value']:.2f}\n"
        body += "-" * 20 + "\n"
    return body

# メール送信
def send_email(subject, body, to_email=None):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = to_email or os.getenv("SEND_TO", smtp_user)

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())

# ★ CSV から銘柄リストを読み込む（ここが重要）
TICKERS = load_tickers()

# メインロジック
def main():
    signals = {}

    for ticker in TICKERS:
        try:
            price_data = get_price(ticker)

            if price_data.empty or len(price_data) < 2:
                continue

            latest_price = price_data.iloc[-1]["4. close"]
            rsi = calculate_rsi(price_data)
            moving_avg = price_data["4. close"].rolling(window=14).mean().iloc[-1]
            price_change = (latest_price - price_data.iloc[-2]["4. close"]) / price_data.iloc[-2]["4. close"]

            data = {
                "close": latest_price,
                "rsi": rsi,
                "moving_avg": moving_avg,
                "price_change": price_change
            }

            signal = check_signal(data)
            expected_value = calculate_expected_value(data)

            signals[ticker] = {
                "signal": signal,
                "rsi": rsi,
                "close": latest_price,
                "moving_avg": moving_avg,
                "expected_value": expected_value
            }
        except Exception as e:
            print(f"エラーが発生しました（{ticker}）: {e}")

    sorted_signals = sorted(signals.items(), key=lambda x: x[1]["expected_value"], reverse=True)
    top_signals = {k: v for k, v in sorted_signals[:3]}

    filtered_signals = filter_alerts(top_signals)

    if filtered_signals:
        email_body = format_alerts_for_email(filtered_signals)
    else:
        email_body = "該当するアラート情報はありませんでした。"

    send_email("最新の株価アラート", email_body)

if __name__ == "__main__":
    main()
