import os
import requests
import pandas as pd

# 環境変数からAPIキーと送信先メールを取得
ALPHA_KEY = os.getenv("ALPHA_KEY")
NEWS_KEY = os.getenv("NEWS_KEY")
FMP_KEY = os.getenv("FMP_KEY")
SENDGRID_KEY = os.getenv("SENDGRID_KEY")
SEND_TO = os.getenv("SEND_TO")

# 株価データ取得（Alpha Vantage）
def get_price(symbol="AUR"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_KEY}"
    data = requests.get(url).json()["Time Series (Daily)"]
    df = pd.DataFrame(data).T.astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# ニュース取得（NewsAPI）
def get_news():
    url = f"https://newsapi.org/v2/everything?q=Aurora+Innovation&apiKey={NEWS_KEY}"
    return requests.get(url).json()["articles"]

# ニューススコア判定
def news_score(articles):
    pos = ["approval", "partnership", "contract", "launch", "expansion"]
    neg = ["accident", "ban", "suspend", "investigation", "recall"]
    score = 0
    for a in articles:
        text = (a["title"] + " " + a.get("description", "")).lower()
        if any(p in text for p in pos):
            score += 1
        if any(n in text for n in neg):
            score -= 1
    return score

# 決算データ取得（FMP）
def get_financials(symbol="AUR"):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?apikey={FMP_KEY}"
    return requests.get(url).json()

# シグナル判定ロジック
def check_signal():
    price = get_price()
    last = price.iloc[-1]
    high_30 = price["2. high"].tail(30).max()
    avg_vol = price["5. volume"].tail(30).mean()

    cond_price = last["4. close"] > high_30
    cond_vol = last["5. volume"] > 2 * avg_vol

    score = news_score(get_news())
    cond_news = score >= 1

    fin = get_financials()
    rev_now = fin[0]["revenue"]
    rev_prev = fin[1]["revenue"]
    cond_fund = (rev_now - rev_prev) / rev_prev > 0.5

    return cond_price and cond_vol and cond_news and cond_fund

# メール送信（SendGrid）
def send_email(msg):
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {SENDGRID_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "personalizations": [{"to": [{"email": SEND_TO}]}],
        "from": {"email": "alert@aurora-signal.com"},
        "subject": "AURORA SIGNAL ALERT",
        "content": [{"type": "text/plain", "value": msg}]
    }
    requests.post(url, headers=headers, json=data)

# メイン処理
if check_signal():
    send_email("AUR: 20% UP シグナル点灯")
