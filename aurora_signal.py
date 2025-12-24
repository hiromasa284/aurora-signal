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
    # 日本株
    jp_df = pd.read_csv("tickers_jp.csv")
    jp_symbols = jp_df["symbol"].dropna().tolist()
    jp_names = dict(zip(jp_df["symbol"], jp_df["name"]))

    # 米国株
    us_df = pd.read_csv("tickers_us.csv")
    us_symbols = us_df["symbol"].dropna().tolist()
    us_names = dict(zip(us_df["symbol"], us_df["name"]))

    # 結合（順序維持＋重複除去）
    symbols = list(dict.fromkeys(jp_symbols + us_symbols))
    names = {**jp_names, **us_names}

    return symbols, names

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

# シグナル判定（勝ちに行くモード）
def check_signal(data):
    rsi = data["rsi"]
    price = data["close"]
    moving_avg = data.get("moving_avg", 150)

    # BUY条件（かなり絞る：売られすぎ＋中期トレンド下）
    if rsi <= 30 and price < moving_avg:
        return "BUY"

    # SELL条件（かなり絞る：買われすぎ＋中期トレンド上）
    elif rsi >= 70 and price > moving_avg:
        return "SELL"

    return "HOLD"

# 勝率と期待値（シンプルな形は一旦維持）
def calculate_expected_value(data):
    # RSI が極端なほど妙味が高いとみなす簡易モデル
    rsi = data["rsi"]
    edge = abs(50 - rsi) / 50  # 50 からの乖離率
    expected_value = edge * data["close"]
    return expected_value

# BUY/SELL のみ抽出
def filter_alerts(alerts):
    return {ticker: info for ticker, info in alerts.items() if info["signal"] in ["BUY", "SELL"]}

# メール本文整形
def format_alerts_for_email(signals):
    """
    アラート情報を整形してメール本文を作成する関数
    """
    body = "以下は最新のハイコンフィデンス・シグナルです：\n\n"
    for ticker, info in signals.items():
        # 銘柄名とティッカーシンボル
        name = NAMES.get(ticker, "N/A")  # 銘柄名を取得
        body += f"銘柄: {name} ({ticker})\n"

        # シグナル
        body += f"  シグナル: {info['signal']}\n"

        # RSI
        body += f"  RSI: {info['rsi']:.2f}\n"

        # 価格
        body += f"  価格: {info['close']}\n"

        # 移動平均
        body += f"  移動平均(50日): {info.get('moving_avg', 'N/A')}\n"

        # 期待値スコアと星の評価
        expected_value = info['expected_value']
        stars = calculate_stars(expected_value)  # 星を計算
        body += f"  期待値スコア: {expected_value:.2f} ({stars})\n"

        # 区切り線
        body += "-" * 20 + "\n"
    return body

def calculate_stars(expected_value):
    """
    期待値スコアに応じて星を付与する関数
    """
    if expected_value >= 50:
        return "★★★★★"
    elif expected_value >= 40:
        return "★★★★☆"
    elif expected_value >= 30:
        return "★★★☆☆"
    elif expected_value >= 20:
        return "★★☆☆☆"
    elif expected_value >= 10:
        return "★☆☆☆☆"
    else:
        return "☆☆☆☆☆"

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

# ★ CSV から銘柄リストを読み込む
TICKERS, NAMES = load_tickers()

# メインロジック
def main():
    signals = {}

    for ticker in TICKERS:
        try:
            price_data = get_price(ticker)

            if price_data.empty or len(price_data) < 50:
                # 50日移動平均を使うので、最低50本必要
                continue

            latest_price = price_data.iloc[-1]["4. close"]
            rsi = calculate_rsi(price_data)
            moving_avg = price_data["4. close"].rolling(window=50).mean().iloc[-1]

            data = {
                "close": latest_price,
                "rsi": rsi,
                "moving_avg": moving_avg,
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

    # まず BUY/SELL だけに絞る（ここが勝ちに行くポイント）
    filtered_signals = filter_alerts(signals)

    if filtered_signals:
        # その中から期待値スコア順に上位3つ
        sorted_signals = sorted(filtered_signals.items(), key=lambda x: x[1]["expected_value"], reverse=True)
        top_signals = dict(sorted_signals[:3])
        email_body = format_alerts_for_email(top_signals)
    else:
        # 本当に何も出なかった日は「今日は無理に触らない日」と割り切る
        email_body = "本日は高確度のシグナルは検出されませんでした。焦らず、チャンスを待ちましょう。"

    send_email("Aurora Signal: ハイコンフィデンス・シグナル", email_body)

if __name__ == "__main__":
    main()
