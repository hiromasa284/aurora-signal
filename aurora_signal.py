import os
import json
import pandas as pd
from datetime import datetime

HISTORY_FILE = "signal_history.json"


# ============================
#  過去シグナル履歴の読み込み
# ============================
def load_signal_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[読み込みエラー] signal_history.json: {e}")
        return []


# ============================
#  履歴の追記保存
# ============================
def append_signal_history(entry):
    history = load_signal_history()
    history.append(entry)

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[保存エラー] signal_history.json: {e}")


# ============================
#  CSV からティッカー読み込み
# ============================
def load_tickers():
    jp = pd.read_csv("tickers_jp.csv")
    us = pd.read_csv("tickers_us.csv")

    tickers = {}
    for _, row in jp.iterrows():
        tickers[row["symbol"]] = row["name"]
    for _, row in us.iterrows():
        tickers[row["symbol"]] = row["name"]

    return tickers

import yfinance as yf
import pandas as pd
import numpy as np

# ============================
#  株価データ取得（日足）
# ============================
def get_price(symbol):
    print(f"[取得開始] {symbol}")

    try:
        # ★ yfinance のフリーズ対策として threads=False を追加
        df = yf.download(
            symbol,
            period="90d",
            interval="1d",
            timeout=10,
            threads=False  # ← これが超重要
        )

        if df is None or df.empty:
            print(f"[データなし] {symbol}")
            return pd.DataFrame()

        # MultiIndex を解除
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]]

        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        print(f"[取得エラー] {symbol}: {e}")
        return pd.DataFrame()

# ============================
#  RSI 計算
# ============================
def calculate_rsi(df, window=14):
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


# ============================
#  シグナル判定
# ============================
def check_signal(row):
    rsi = row["rsi"]
    price = row["close"]
    moving_avg = row.get("moving_avg", 150)

    if rsi <= 30 and price < moving_avg:
        return "BUY"

    if rsi >= 70 and price > moving_avg:
        return "SELL"

    return "HOLD"


# ============================
#  期待値スコア
# ============================
def calculate_expected_value(row):
    rsi = row["rsi"]
    price = row["close"]

    edge = (abs(50 - rsi) / 50) ** 2
    expected_value = edge * price

    return expected_value


# ============================
#  ランク判定
# ============================
def rank_signal(expected_value, signal_type):
    """
    expected_value と 過去勝率を組み合わせてランクを決定。
    """
    win_rates = calculate_win_rates()

    if signal_type == "BUY":
        base_win = win_rates["buy_win_rate"]
    else:
        base_win = win_rates["sell_win_rate"]

    score = (expected_value * 0.7) + (base_win * 0.3)

    if score >= 120:
        return "S"
    elif score >= 80:
        return "A"
    else:
        return "B"

from datetime import datetime, timedelta
import pandas as pd

from datetime import datetime

# ============================
#  追跡日数を計算
# ============================
def calculate_tracking_days(entry):
    ts = entry["timestamp"]
    date_str = ts.split("T")[0]
    signal_date = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.utcnow()
    delta = today - signal_date
    return delta.days

# ============================
#  シグナルの勝敗判定（タッチするまで追跡）
# ============================
def evaluate_signal_outcome(entry):
    ticker = entry["ticker"]
    signal = entry["signal"]
    tp = entry["take_profit"]
    sl = entry["stop_loss"]
    date = entry["timestamp"][:10]

    df = get_price(ticker)
    df_after = df[df.index.date > datetime.strptime(date, "%Y-%m-%d").date()]

    for _, row in df_after.iterrows():
        high = row["high"]
        low = row["low"]

        if signal == "BUY":
            if high >= tp:
                return "win"
            if low <= sl:
                return "lose"

        if signal == "SELL":
            if low <= tp:
                return "win"
            if high >= sl:
                return "lose"

    return None  # まだ決着していない

# ============================
#  ランク別累積勝率
# ============================
def calculate_rank_stats(history):
    stats = {"S": {"win": 0, "lose": 0},
             "A": {"win": 0, "lose": 0},
             "B": {"win": 0, "lose": 0}}

    for e in history:
        if e.get("resolved", False):
            r = e["rank"]
            if e["result"] == "win":
                stats[r]["win"] += 1
            elif e["result"] == "lose":
                stats[r]["lose"] += 1

    win_rates = {}
    for r, v in stats.items():
        total = v["win"] + v["lose"]
        win_rates[r] = round((v["win"] / total) * 100, 1) if total > 0 else 0

    return stats, win_rates


# ============================
#  追跡中件数 + 平均追跡日数
# ============================
def count_unresolved_by_rank_with_days(history):
    counts = {"S": 0, "A": 0, "B": 0}
    days_sum = {"S": 0, "A": 0, "B": 0}

    for e in history:
        if not e.get("resolved", False):
            r = e["rank"]
            counts[r] += 1
            days_sum[r] += calculate_tracking_days(e)

    avg_days = {}
    for r in counts:
        avg_days[r] = round(days_sum[r] / counts[r], 1) if counts[r] > 0 else 0

    total = counts["S"] + counts["A"] + counts["B"]
    return counts, avg_days, total


# ============================
#  本日決着した銘柄のテキスト生成
# ============================
def format_resolved_today(resolved_today):
    if not resolved_today:
        return "【本日決着したシグナル】\n本日は決着した銘柄はありませんでした。\n"

    lines = ["【本日決着したシグナル】\n"]

    for e in resolved_today:
        days = calculate_tracking_days(e)
        result_text = "勝ち（利確ラインにタッチ）" if e["result"] == "win" else "負け（損切りラインにタッチ）"

        lines.append(
            f"■ {e['ticker']} / {e.get('name','')}（{e['rank']}ランク）\n"
            f"  シグナル: {e['signal']}\n"
            f"  終値（シグナル時）: {e['close']}\n"
            f"  利確ライン: {e['take_profit']}\n"
            f"  損切りライン: {e['stop_loss']}\n"
            f"  → 結果: {result_text}\n"
            f"  → 追跡日数: {days}日\n"
            "--------------------"
        )

    return "\n".join(lines)

# ============================
#  メイン：過去シグナルの評価
# ============================
def evaluate_past_signals():
    history = load_signal_history()
    resolved_today = []

    for entry in history:
        if entry.get("resolved", False):
            continue

        outcome = evaluate_signal_outcome(entry)

        if outcome in ["win", "lose"]:
            entry["result"] = outcome
            entry["resolved"] = True
            entry["score"] = 1 if outcome == "win" else -1
            resolved_today.append(entry)

    save_signal_history(history)

    # ランク別累積勝率
    stats, win_rates = calculate_rank_stats(history)

    # 追跡中件数
    counts, avg_days, total = count_unresolved_by_rank_with_days(history)

    # 本日決着した銘柄
    resolved_text = format_resolved_today(resolved_today)

    # ===== 出力テキスト =====
    print("\n" + resolved_text)

    print("【ランク別累積成績】")
    print(f"Sランク： +{stats['S']['win']} / -{stats['S']['lose']}  → 勝率 {win_rates['S']}%")
    print(f"Aランク： +{stats['A']['win']} / -{stats['A']['lose']}  → 勝率 {win_rates['A']}%")
    print(f"Bランク： +{stats['B']['win']} / -{stats['B']['lose']}  → 勝率 {win_rates['B']}%")

    print("\n【追跡中の銘柄数】")
    print(f"Sランク： {counts['S']}件（平均 {avg_days['S']}日）")
    print(f"Aランク： {counts['A']}件（平均 {avg_days['A']}日）")
    print(f"Bランク： {counts['B']}件（平均 {avg_days['B']}日）")
    print(f"計： {total}件\n")

    print("evaluate_past_signals: END")

# ============================
#  全体勝率の集計
# ============================
def calculate_win_rates():
    history = load_signal_history()

    buy_total = sell_total = 0
    buy_win = sell_win = 0
    buy_gain_sum = sell_drop_sum = 0.0

    for entry in history:
        signal = entry.get("signal")
        r1 = entry.get("result_1d")

        if r1 not in ["WIN", "LOSE"]:
            continue

        price_0d = entry.get("close")
        price_1d = entry.get("price_1d")

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

    return {
        "buy_win_rate": round((buy_win / buy_total * 100), 1) if buy_total else 0,
        "sell_win_rate": round((sell_win / sell_total * 100), 1) if sell_total else 0,
        "buy_avg_gain": round((buy_gain_sum / buy_total), 2) if buy_total else 0,
        "sell_avg_drop": round((sell_drop_sum / sell_total), 2) if sell_total else 0
    }


# ============================
#  ランク別勝率の集計（S/A/B）
# ============================
def calculate_ranked_win_rates():
    history = load_signal_history()

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
        rank = entry.get("rank")

        if rank not in ["S", "A", "B"]:
            continue
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

# ============================
#  利確・損切りラインの計算
# ============================
def calculate_exit_levels(close, expected_value, signal):
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

# ============================
#  メール本文生成
# ============================
def format_alerts_for_email(signals):
    body = "【Aurora Signal: ハイコンフィデンス・シグナル】\n\n"

    # 全体勝率
    win_rates = calculate_win_rates()
    buy_win = win_rates["buy_win_rate"]
    sell_win = win_rates["sell_win_rate"]

    # 銘柄ごとの表示
    for ticker, info in signals.items():
        win_rate = buy_win if info["signal"] == "BUY" else sell_win
        rank = info["rank"]

        take_profit, stop_loss = calculate_exit_levels(
            info["close"],
            info["expected_value"],
            info["signal"]
        )

        body += f"■ {ticker} / {info['name']}（{rank}ランク）\n"
        body += f"  シグナル: {info['signal']}\n"
        body += f"  RSI: {info['rsi']:.2f}\n"
        body += f"  終値: {info['close']:.2f}\n"
        body += f"  移動平均(50日): {info['moving_avg']:.2f}\n"
        body += f"  期待値スコア: {info['expected_value']:.2f}\n"

        if rank == "B":
            body += "  ※Bランクは信頼度が低いため、参考程度にご利用ください\n"

        body += "  ▶ 手じまいガイド（期待値ベース）\n"
        body += f"     利確ライン: {take_profit}\n"
        body += f"     損切りライン: {stop_loss}\n"
        body += "--------------------\n\n"

    # 全体勝率
    body += "【過去シグナルの成績（1日後）】\n"
    body += f"BUY 勝率: {buy_win}%\n"
    body += f"SELL 勝率: {sell_win}%\n"
    body += f"平均反発率: +{win_rates['buy_avg_gain']}%\n"
    body += f"平均下落率: {win_rates['sell_avg_drop']}%\n\n"

    # ランク別勝率
    ranked = calculate_ranked_win_rates()

    body += "【ランク別成績（1日後）】\n"
    body += f"Sランク BUY勝率: {ranked['S']['buy_win_rate']}% / 平均反発率: +{ranked['S']['buy_avg_gain']}%\n"
    body += f"Sランク SELL勝率: {ranked['S']['sell_win_rate']}% / 平均下落率: {ranked['S']['sell_avg_drop']}%\n\n"

    body += f"Aランク BUY勝率: {ranked['A']['buy_win_rate']}% / 平均反発率: +{ranked['A']['buy_avg_gain']}%\n"
    body += f"Aランク SELL勝率: {ranked['A']['sell_win_rate']}% / 平均下落率: {ranked['A']['sell_avg_drop']}%\n\n"

    body += f"Bランク BUY勝率: {ranked['B']['buy_win_rate']}% / 平均反発率: +{ranked['B']['buy_avg_gain']}%\n"
    body += f"Bランク SELL勝率: {ranked['B']['sell_win_rate']}% / 平均下落率: {ranked['B']['sell_avg_drop']}%\n"

    return body

# ============================
#  メイン処理（完全版）
# ============================
def main():
    print("main: START")

    TICKERS = load_tickers()
    signals = {}
    run_timestamp = datetime.utcnow().isoformat()

    for ticker, name in TICKERS.items():
        try:
            df = get_price(ticker)

            if df.empty or len(df) < 15:
                print(f"{ticker} はデータ不足のためスキップ")
                continue

            df["rsi"] = calculate_rsi(df)
            latest = df.iloc[-1]

            close = latest["close"]
            rsi = latest["rsi"]
            moving_avg = df["close"].rolling(50).mean().iloc[-1]
            signal = check_signal(latest)
            expected_value = calculate_expected_value(latest)
            rank = rank_signal(expected_value, signal)

            # ★ HOLD は保存しない（重要）
            if signal == "HOLD":
                continue

            # 利確・損切りラインを計算
            take_profit, stop_loss = calculate_exit_levels(
                close,
                expected_value,
                signal
            )

            # 履歴保存
            history_entry = {
                "ticker": ticker,
                "signal": signal,
                "rsi": rsi,
                "close": close,
                "expected_value": expected_value,
                "rank": rank,
                "timestamp": run_timestamp,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "resolved": False,
                "result": None,
                "score": 0
            }
            append_signal_history(history_entry)

            # 今日のシグナル保存
            signals[ticker] = {
                "name": name,
                "signal": signal,
                "rsi": rsi,
                "close": close,
                "moving_avg": moving_avg,
                "expected_value": expected_value,
                "rank": rank,
                "timestamp": run_timestamp,
                "take_profit": take_profit,
                "stop_loss": stop_loss
            }

            print(f"{ticker}（{name}） {signal}")

        except Exception as e:
            print(f"[エラー] {ticker}: {e}")
            continue

    # ============================
    #  BUY/SELL のみ抽出
    # ============================
    filtered = {t: s for t, s in signals.items() if s["signal"] in ["BUY", "SELL"]}

    print("\n[DEBUG] 今日の BUY/SELL シグナル一覧")
    if filtered:
        for t, s in filtered.items():
            print(f"  {t}: {s['signal']} | EV={s['expected_value']:.2f} | Rank={s['rank']}")
    else:
        print("  BUY/SELL シグナルなし")

    # ============================
    #  通知テキスト生成
    # ============================
    if filtered:
        sorted_signals = sorted(filtered.items(), key=lambda x: x[1]["expected_value"], reverse=True)
        email_body = format_alerts_for_email(dict(sorted_signals))
    else:
        email_body = "本日は高確度のシグナルは検出されませんでした。焦らず、チャンスを待ちましょう。"

    print("main: END")

    return email_body


# ============================
#  実行
# ============================
if __name__ == "__main__":
    email_body = main()
    evaluate_past_signals()

    # ★★★ 最後に通知内容を出す（ここが最終位置） ★★★
    print("\n===== AuroraSignal 通知内容 =====")
    print(email_body)
    print("================================\n")
