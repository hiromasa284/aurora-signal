TICKERS = ["AAPL", "MSFT", "TSLA"]
def get_stock_data(ticker):
    # 仮のデータ（本番では API から取得）
    return {
        "close": 150.0,
        "rsi": 55.0
    }

def check_signal(data):
    if data["rsi"] > 70:
        return "SELL"
    elif data["rsi"] < 30:
        return "BUY"
    else:
        return "HOLD"
        TICKERS = ["AAPL", "MSFT", "TSLA"]
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from google.oauth2 import service_account
from googleapiclient.discovery import build
import base64

# 株価取得（Alpha Vantage）
def get_price(symbol):
    key = os.getenv("ALPHA_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}"
    r = requests.get(url).json()
    data = r.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df = df.astype(float)
    return df

# ニュース取得（NewsAPI）
def get_news(symbol):
    key = os.getenv("NEWS_KEY")
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&apiKey={key}"
    r = requests.get(url).json()
    return r.get("articles", [])

# ニューススコア（簡易判定）
def news_score(articles):
    score = 0
    for a in articles[:5]:
        title = a.get("title", "").lower()
        if "upgrade" in title or "surge" in title or "beat" in title:
            score += 1
        elif "downgrade" in title or "miss" in title or "fall" in title:
            score -= 1
    return score

# 決算データ取得（FMP）
def get_fundamentals(symbol):
    key = os.getenv("FMP_KEY")
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=2&apikey={key}"
    r = requests.get(url).json()
    if len(r) < 2:
        return None
    latest, prev = r[0], r[1]
    growth = (latest["revenue"] - prev["revenue"]) / prev["revenue"]
    return {
        "eps": latest.get("eps"),
        "revenue_growth": growth,
        "netIncome": latest.get("netIncome")
    }

# Gmail API でメール送信
def send_email(subject, body):
    creds_json = os.getenv("GMAIL_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=["https://www.googleapis.com/auth/gmail.send"]
    )
    service = build("gmail", "v1", credentials=creds)

    message = MIMEText(body)
    message["to"] = os.getenv("SEND_TO")
    message["from"] = "me"
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

# メインロジック
def main():
    symbol = "AUR"
    price_df = get_price(symbol)
    latest = price_df.iloc[-1]["4. close"]
    prev = price_df.iloc[-2]["4. close"]
    price_change = (latest - prev) / prev

    articles = get_news(symbol)
    score = news_score(articles)

    fundamentals = get_fundamentals(symbol)
    growth = fundamentals["revenue_growth"] if fundamentals else 0

    if price_change > 0.2 or score >= 2 or growth > 0.3:
        body = f"""Aurora Signal Alert 🚨

Symbol: {symbol}
Price Change: {price_change:.2%}
News Score: {score}
Revenue Growth: {growth:.2%}

Triggered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        send_email(f"{symbol}: Signal Triggered", body)

if __name__ == "__main__":
    main()
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = smtp_user

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
import os
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    smtp_user = os.getenv("SMTP_USER")      # 送信元（Gmail）
    smtp_pass = os.getenv("SMTP_PASS")      # アプリパスワード
    send_to = os.getenv("SEND_TO", smtp_user)  # 送信先（SEND_TO が無ければ自分）

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = send_to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
import json

def save_signal_json(signals):
    with open("signal.json", "w") as f:
        json.dump(signals, f, indent=4)
signals = {}

for ticker in TICKERS:
    data = get_stock_data(ticker)
    signal = check_signal(data)
    signals[ticker] = signal

save_signal_json(signals)
def get_stock_data(ticker):
    # 仮のデータ構造（本番では API から取得）
    return {
        "close": 150.0,
        "rsi": 55.0
    }

def check_signal(data):
    # 仮のシグナル判定
    if data["rsi"] > 70:
        return "SELL"
    elif data["rsi"] < 30:
        return "BUY"
    else:
        return "HOLD"
import json

signals = {
    "AAPL": {"signal": "BUY", "rsi": 28.5, "price": 147.23},
    "MSFT": {"signal": "HOLD", "rsi": 52.1, "price": 310.45},
    "TSLA": {"signal": "SELL", "rsi": 72.8, "price": 245.67}
}

with open("signal.json", "w") as f:
    json.dump(signals, f, indent=2)
