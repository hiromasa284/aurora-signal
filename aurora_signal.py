import os
import json
import requests
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import smtplib

HISTORY_FILE = "signal_history.json"  # リポジトリ直下に保存

def load_signal_history():
    """
    過去のシグナル履歴を読み込む。
    ファイルが存在しなければ空のリストを返す。
    """
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except Exception as e:
        print(f"signal_history.json の読み込み中にエラー: {e}")
        return []

def save_signal_history(signals, run_timestamp=None):
    """
    今回の実行で得られた全シグナルを signal_history.json に追記保存する。
    signals: {ticker: {signal, rsi, close, moving_avg, expected_value}}
    """
    if run_timestamp is None:
        run_timestamp = datetime.utcnow().isoformat()

    history = load_signal_history()

    for ticker, info in signals.items():
        record = {
            "timestamp": run_timestamp,
            "ticker": ticker,
            "signal": info.get("signal"),
            "rsi": info.get("rsi"),
            "close": info.get("close"),
            "moving_avg": info.get("moving_avg"),
            "expected_value": info.get("expected_value"),
        }
        history.append(record)

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"signal_history.json の書き込み中にエラー: {e}")

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
    print(f"[取得開始] {symbol}")
    key = os.getenv("ALPHA_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}"
    r = requests.get(url).json()

    # ★ ここに入れる
    if "Information" in r:
        print(f"[API制限] {symbol}: {r['Information']}")
        return pd.DataFrame()

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
    """
    RSI が極端なほど妙味が高いとみなすモデル。
    50 からの乖離を二乗して、極端値を強調。
    """
    rsi = data["rsi"]
    price = data["close"]

    edge = (abs(50 - rsi) / 50) ** 2
    expected_value = edge * price
    return expected_value

# BUY/SELL のみ抽出
def filter_alerts(alerts):
    return {ticker: info for ticker, info in alerts.items() if info["signal"] in ["BUY", "SELL"]}

def evaluate_past_signals():
    print("evaluate_past_signals: START")
    
    """
    過去のシグナル履歴を読み込み、
    翌日・3日後の価格を取得して、
    BUY/SELL の成否を判定する。
    """
    history = load_signal_history()
    updated = False

    for entry in history:
        # すでに評価済みならスキップ
        if "result_1d" in entry and "result_3d" in entry:
            continue

        symbol = entry["ticker"]
        signal = entry["signal"]
        timestamp = entry["timestamp"]

        try:
            price_data = get_price(symbol)
            if price_data.empty:
                continue

            # 日付の整形（UTC → 日付部分だけ）
            date_str = timestamp[:10]
            dates = sorted(price_data.index)

            # 翌日・3日後のインデックスを探す
            if date_str not in dates:
                continue

            idx = dates.index(date_str)
            if idx + 1 >= len(dates) or idx + 3 >= len(dates):
                continue

            price_0d = price_data.loc[dates[idx]]["4. close"]
            price_1d = price_data.loc[dates[idx + 1]]["4. close"]
            price_3d = price_data.loc[dates[idx + 3]]["4. close"]

            # 判定ロジック
            def judge(p0, pX, signal):
                if signal == "BUY":
                    return "WIN" if pX > p0 else "LOSE"
                elif signal == "SELL":
                    return "WIN" if pX < p0 else "LOSE"
                else:
                    return "N/A"

            entry["result_1d"] = judge(price_0d, price_1d, signal)
            entry["result_3d"] = judge(price_0d, price_3d, signal)

            # ★ これを追加
            entry["price_1d"] = price_1d
            entry["price_3d"] = price_3d
           
            updated = True

        except Exception as e:
            print(f"[追跡エラー] {symbol}: {e}")
            continue

    # 更新があったら保存
    if updated:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print("signal_history.json を更新しました（追跡結果付き）")
        except Exception as e:
            print(f"[保存エラー] signal_history.json: {e}")

    # 🔹 これが正しい位置
    print("evaluate_past_signals: END")

def calculate_win_rates():
    """
    signal_history.json から勝率と平均反発率を集計する。
    """
    history = load_signal_history()

    buy_total = sell_total = 0
    buy_win = sell_win = 0
    buy_gain_sum = sell_drop_sum = 0.0

    for entry in history:
        signal = entry.get("signal")
        r1 = entry.get("result_1d")

        # 翌日価格が記録されていない場合はスキップ
        if r1 not in ["WIN", "LOSE"]:
            continue

        price_0d = entry.get("close")
        price_1d = entry.get("price_1d", None)

        # price_1d を保存していない場合は計算できないのでスキップ
        if price_1d is None:
            continue

        change_pct = ((price_1d - price_0d) / price_0d) * 100

        if signal == "BUY":
            buy_total += 1
            if r1 == "WIN":
                buy_win += 1
            buy_gain_sum += change_pct

        elif signal == "SELL":
            sell_total += 1
            if r1 == "WIN":
                sell_win += 1
            sell_drop_sum += change_pct

    buy_win_rate = round((buy_win / buy_total * 100), 1) if buy_total else 0
    sell_win_rate = round((sell_win / sell_total * 100), 1) if sell_total else 0
    buy_avg_gain = round((buy_gain_sum / buy_total), 2) if buy_total else 0
    sell_avg_drop = round((sell_drop_sum / sell_total), 2) if sell_total else 0

    return {
        "buy_win_rate": buy_win_rate,
        "sell_win_rate": sell_win_rate,
        "buy_avg_gain": buy_avg_gain,
        "sell_avg_drop": sell_avg_drop
    }

def format_alerts_for_email(signals):
    body = "【Aurora Signal: ハイコンフィデンス・シグナル】\n\n"

    # 勝率データを取得
    win_rates = calculate_win_rates()
    buy_win = win_rates["buy_win_rate"]
    sell_win = win_rates["sell_win_rate"]

    # 銘柄ごとの表示
    for ticker, info in signals.items():
        win_rate = buy_win if info["signal"] == "BUY" else sell_win
        rank = rank_signal(info["expected_value"], win_rate)

        body += f"■ {ticker}（{rank}ランク）\n"
        body += f"  シグナル: {info['signal']}\n"
        body += f"  RSI: {info['rsi']:.2f}\n"
        body += f"  終値: {info['close']:.2f}\n"
        body += f"  移動平均(50日): {info['moving_avg']:.2f}\n"
        body += f"  期待値スコア: {info['expected_value']:.2f}\n"
        body += "--------------------\n"

    # 勝率サマリー
    body += "\n【過去シグナルの成績（1日後）】\n"
    body += f"BUY 勝率: {buy_win}%\n"
    body += f"SELL 勝率: {sell_win}%\n"
    body += f"平均反発率: +{win_rates['buy_avg_gain']}%\n"
    body += f"平均下落率: {win_rates['sell_avg_drop']}%\n"

    return body

def rank_signal(expected_value, win_rate):
    total_score = expected_value * (win_rate / 100)
    if total_score >= 300 and win_rate >= 70:
        return "S"
    elif total_score >= 150 and win_rate >= 55:
        return "A"
    else:
        return "B"

def send_email(subject, body):
    sender = os.getenv("EMAIL_SENDER")
    recipient = os.getenv("EMAIL_RECIPIENT")
    password = os.getenv("EMAIL_PASSWORD")

    if not sender or not recipient or not password:
        print("メール送信に必要な環境変数が不足しています")
        return

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 🔹 ここにログ出力を追加（インデント修正済み）
    print("送信者:", sender)
    print("宛先:", recipient)
    print("件名:", subject)
    print("本文:\n", body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("メール送信に成功しました")
    except Exception as e:
        print(f"メール送信中にエラー: {e}")

def main():
    signals = {}
    run_timestamp = datetime.utcnow().isoformat()

    # BUY/SELL のみ抽出
    filtered_signals = filter_alerts(signals)

    if filtered_signals:
        sorted_signals = sorted(
            filtered_signals.items(),
            key=lambda x: x[1]["expected_value"],
            reverse=True
        )
        top_signals = dict(sorted_signals[:3])
        email_body = format_alerts_for_email(top_signals)
    else:
        email_body = "本日は高確度のシグナルは検出されませんでした。焦らず、チャンスを待ちましょう。"

    send_email("Aurora Signal: ハイコンフィデンス・シグナル", email_body)

TICKERS, NAMES = load_tickers()
TICKERS = TICKERS[:25]

if __name__ == "__main__":
    evaluate_past_signals()
    main()
