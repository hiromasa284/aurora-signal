import os
import json
import requests
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import smtplib
import time

# 🔥 Secrets 読み込み（ここがベスト）
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SEND_TO = os.getenv("SEND_TO")

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

def get_price(symbol):
    print(f"[取得開始] {symbol}")
    key = os.getenv("FMP_KEY")

    symbol_clean = symbol.replace(".T", "")
    urls = [
        f"https://financialmodelingprep.com/api/v3/historical-chart/4hour/{symbol_clean}?apikey={key}",
        f"https://financialmodelingprep.com/api/v3/historical-chart/4hour/{symbol}?apikey={key}",
    ]

    for url in urls:
        try:
            r = requests.get(url).json()
        except Exception as e:
            print(f"[取得エラー] {symbol}: {e}")
            continue

        if isinstance(r, list) and len(r) > 0 and "date" in r[0]:
            df = pd.DataFrame(r)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            return df

    print(f"{symbol} のデータが取得できませんでした")
    return pd.DataFrame()

def calculate_rsi(data, window=14):
    delta = data["close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]

# シグナル判定（勝ちに行くモード）
def check_signal(row):
    rsi = row["rsi"]
    price = row["close"]
    moving_avg = row.get("moving_avg", 150)

    if rsi <= 30 and price < moving_avg:
        return "BUY"

    if rsi >= 70 and price > moving_avg:
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

            # 🔥 空データ対策
            if len(dates) == 0:
                continue

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

            # 追加情報
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

    print("evaluate_past_signals: END")

def append_signal_history(entry):
    history = load_signal_history()
    history.append(entry)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[保存エラー] signal_history.json: {e}")

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

def calculate_ranked_win_rates():
    """
    signal_history.json からランク別の勝率と平均反発率を集計する。
    """
    history = load_signal_history()

    # ランク別の集計用辞書
    rank_stats = {
        "S": {"buy_total": 0, "buy_win": 0, "buy_gain_sum": 0.0,
              "sell_total": 0, "sell_win": 0, "sell_drop_sum": 0.0},
        "A": {"buy_total": 0, "buy_win": 0, "buy_gain_sum": 0.0,
              "sell_total": 0, "sell_win": 0, "sell_drop_sum": 0.0},
        "B": {"buy_total": 0, "buy_win": 0, "buy_gain_sum": 0.0,
              "sell_total": 0, "sell_win": 0, "sell_drop_sum": 0.0},
    }

    for entry in history:
        signal = entry.get("signal")
        r1 = entry.get("result_1d")
        rank = entry.get("rank")  # ← main() で保存したランクを使う

        # ランクが保存されていない古いデータはスキップ
        if rank not in ["S", "A", "B"]:
            continue

        # 翌日結果がない場合はスキップ
        if r1 not in ["WIN", "LOSE"]:
            continue

        price_0d = entry.get("close")
        price_1d = entry.get("price_1d")

        if price_1d is None:
            continue

        change_pct = ((price_1d - price_0d) / price_0d) * 100

        stats = rank_stats[rank]

        if signal == "BUY":
            stats["buy_total"] += 1
            if r1 == "WIN":
                stats["buy_win"] += 1
            stats["buy_gain_sum"] += change_pct

        elif signal == "SELL":
            stats["sell_total"] += 1
            if r1 == "WIN":
                stats["sell_win"] += 1
            stats["sell_drop_sum"] += change_pct

    # 勝率と平均値を計算
    result = {}
    for rank, stats in rank_stats.items():
        result[rank] = {
            "buy_win_rate": round((stats["buy_win"] / stats["buy_total"] * 100), 1)
                              if stats["buy_total"] else 0,
            "sell_win_rate": round((stats["sell_win"] / stats["sell_total"] * 100), 1)
                              if stats["sell_total"] else 0,
            "buy_avg_gain": round((stats["buy_gain_sum"] / stats["buy_total"]), 2)
                              if stats["buy_total"] else 0,
            "sell_avg_drop": round((stats["sell_drop_sum"] / stats["sell_total"]), 2)
                              if stats["sell_total"] else 0,
        }

    return result

def format_alerts_for_email(signals):
    body = "【Aurora Signal: ハイコンフィデンス・シグナル】\n\n"

    # 勝率データを取得
    win_rates = calculate_win_rates()
    buy_win = win_rates["buy_win_rate"]
    sell_win = win_rates["sell_win_rate"]

    # 銘柄ごとの表示
    for ticker, info in signals.items():
        win_rate = buy_win if info["signal"] == "BUY" else sell_win
        rank = rank_signal(info["expected_value"], info["signal"])

        # 手じまいライン
        take_profit, stop_loss = calculate_exit_levels(
            info["close"],
            info["expected_value"],
            info["signal"],
            rank
        )

        # 銘柄ブロック
        body += f"■ {ticker}（{rank}ランク）\n"
        body += f"  シグナル: {info['signal']}\n"
        body += f"  RSI: {info['rsi']:.2f}\n"
        body += f"  終値: {info['close']:.2f}\n"
        body += f"  移動平均(50日): {info['moving_avg']:.2f}\n"
        body += f"  期待値スコア: {info['expected_value']:.2f}\n"

        # Bランク注意書き
        if rank == "B":
            body += "  ※Bランクは信頼度が低いため、参考程度にご利用ください\n"

        # 手じまいガイド
        body += "  ▶ 手じまいガイド（期待値ベース）\n"
        body += f"     利確ライン: {take_profit}\n"
        body += f"     損切りライン: {stop_loss}\n"
        body += "--------------------\n\n"

    # 勝率サマリー
    body += "【過去シグナルの成績（1日後）】\n"
    body += f"BUY 勝率: {buy_win}%\n"
    body += f"SELL 勝率: {sell_win}%\n"
    body += f"平均反発率: +{win_rates['buy_avg_gain']}%\n"
    body += f"平均下落率: {win_rates['sell_avg_drop']}%\n\n"

    # ランク別成績（動的）
    ranked = calculate_ranked_win_rates()

    body += "【ランク別成績（1日後）】\n"
    body += f"Sランク BUY勝率: {ranked['S']['buy_win_rate']}% / 平均反発率: +{ranked['S']['buy_avg_gain']}%\n"
    body += f"Sランク SELL勝率: {ranked['S']['sell_win_rate']}% / 平均下落率: {ranked['S']['sell_avg_drop']}%\n\n"

    body += f"Aランク BUY勝率: {ranked['A']['buy_win_rate']}% / 平均反発率: +{ranked['A']['buy_avg_gain']}%\n"
    body += f"Aランク SELL勝率: {ranked['A']['sell_win_rate']}% / 平均下落率: {ranked['A']['sell_avg_drop']}%\n\n"

    body += f"Bランク BUY勝率: {ranked['B']['buy_win_rate']}% / 平均反発率: +{ranked['B']['buy_avg_gain']}%\n"
    body += f"Bランク SELL勝率: {ranked['B']['sell_win_rate']}% / 平均下落率: {ranked['B']['sell_avg_drop']}%\n"

    return body

def rank_signal(expected_value, signal_type):
    """
    expected_value と 過去の勝率データ を使ってランクを判定する。
    signal_type は "BUY" または "SELL"
    """

    # 全体の勝率
    win_rates = calculate_win_rates()

    # BUY/SELL の全体勝率
    if signal_type == "BUY":
        base_win = win_rates["buy_win_rate"]
    else:
        base_win = win_rates["sell_win_rate"]

    # 期待値と勝率の複合スコア
    score = (expected_value * 0.7) + (base_win * 0.3)

    # ランク判定ロジック（調整可能）
    if score >= 120:
        return "S"
    elif score >= 80:
        return "A"
    else:
        return "B"
        
def calculate_exit_levels(close, expected_value, signal):
    """
    期待値ベースの利確・損切りラインを計算する。
    expected_value が大きいほど利確幅を広げる動的モデル。
    """

    # 係数（調整可能）
    take_profit_factor = expected_value / 50000
    stop_loss_factor = expected_value / 80000

    if signal == "BUY":
        take_profit = close * (1 + take_profit_factor)
        stop_loss = close * (1 - stop_loss_factor)

    elif signal == "SELL":
        take_profit = close * (1 - take_profit_factor)
        stop_loss = close * (1 + stop_loss_factor)

    else:
        return None, None

    return round(take_profit, 2), round(stop_loss, 2)

def load_tickers_from_csv(path):
    df = pd.read_csv(path)
    return df["symbol"].tolist()

# 🔥 銘柄リスト読み込み（ここがベスト）
TICKERS, NAMES = load_tickers()

def main():
    print("main: START")
    signals = {}
    api_limited = False
    run_timestamp = datetime.utcnow().isoformat()

    for ticker in TICKERS:
        try:
            df = get_price(ticker)

            # データ不足
            if df.empty or len(df) < 15:
                print(f"{ticker} はデータ不足のためスキップ")
                signals[ticker] = {
                    "signal": "HOLD",
                    "rsi": None,
                    "close": None,
                    "moving_avg": None,
                    "expected_value": None,
                    "rank": None,
                    "timestamp": run_timestamp
                }
                continue

            # RSI 計算
            df["rsi"] = calculate_rsi(df)

            # 最新行
            latest = df.iloc[-1]

            close = latest["close"]
            rsi = latest["rsi"]

            # 移動平均（50本）
            moving_avg = df["close"].rolling(50).mean().iloc[-1]

            # シグナル判定
            signal = check_signal(latest)

            # 期待値
            expected_value = calculate_expected_value(latest)

            # ランク
            rank = rank_signal(expected_value, signal)

            # 履歴保存
            history_entry = {
                "ticker": ticker,
                "signal": signal,
                "rsi": rsi,
                "close": close,
                "expected_value": expected_value,
                "rank": rank,
                "timestamp": run_timestamp
            }
            append_signal_history(history_entry)

            # メール用
            signals[ticker] = {
                "signal": signal,
                "rsi": rsi,
                "close": close,
                "moving_avg": moving_avg,
                "expected_value": expected_value,
                "rank": rank,
                "timestamp": run_timestamp
            }

            print(ticker, signal)

        except Exception as e:
            print(f"[エラー] {ticker}: {e}")
            api_limited = True
            continue

    # BUY/SELL 抽出
    filtered = filter_alerts(signals)

    if filtered:
        sorted_signals = sorted(
            filtered.items(),
            key=lambda x: x[1]["expected_value"],
            reverse=True
        )
        top_signals = dict(sorted_signals[:3])
        email_body = format_alerts_for_email(top_signals)
    else:
        email_body = "本日は高確度のシグナルは検出されませんでした。焦らず、チャンスを待ちましょう。"

    if api_limited:
        email_body += "\n\n※一部銘柄はAPI制限により分析できませんでした。ご了承ください。"

    send_email("Aurora Signal: ハイコンフィデンス・シグナル", email_body)
    print("main: END")

# 🔥 ここに置く（main の外）
import smtplib
from email.mime.text import MIMEText   # ← 修正ポイント

def send_email(subject, body):
    try:
        print("[メール送信開始]")

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = SEND_TO

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, SEND_TO, msg.as_string())
        server.quit()

        print("[メール送信完了]")

    except Exception as e:
        print(f"[メール送信エラー] {e}")

if __name__ == "__main__":
    main()
