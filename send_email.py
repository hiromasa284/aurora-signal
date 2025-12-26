def main():
    try:
        # --- CSV から symbol → name の辞書を作成 ---
        import pandas as pd
        jp = pd.read_csv("tickers_jp.csv")
        us = pd.read_csv("tickers_us.csv")
        name_map = pd.concat([jp, us]).set_index("symbol")["name"].to_dict()

        # --- signal.json を読み込む ---
        with open("signal.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        signals = data.get("signals", [])

        # --- シグナルが無い場合 ---
        if not signals:
            body = "本日のシグナルはありませんでした。\n\n" + data["stats"]
            subject = "Aurora Signal: シグナルなし"
            send_email(subject, body)
            return

        # --- 銘柄一覧（冒頭に表示） ---
        ticker_list = []
        for s in signals:
            ticker = s["ticker"]
            name = name_map.get(ticker, "名称不明")
            ticker_list.append(f"{ticker}（{name}）")

        ticker_list_text = ", ".join(ticker_list)

        # --- 銘柄ごとの詳細 ---
        detail_lines = []
        for s in signals:
            ticker = s["ticker"]
            name = name_map.get(ticker, "名称不明")

            detail_lines.append(
                f"■ {ticker}（{name} / {s['rank']}ランク）\n"
                f"  シグナル: {s['signal']}\n"
                f"  RSI: {s['rsi']}\n"
                f"  終値: {s['price']}\n"
                f"  移動平均(50日): {s['ma50']}\n"
                f"  期待値スコア: {s['score']}\n"
                f"  ▶ 手じまいガイド（期待値ベース）\n"
                f"     利確ライン: {s['tp']}\n"
                f"     損切りライン: {s['sl']}\n"
                "--------------------\n"
            )

        detail_text = "".join(detail_lines)

        # --- メール本文 ---
        body = (
            f"【本日の対象銘柄】\n"
            f"{ticker_list_text}\n\n"
            f"{detail_text}"
            f"{data['stats']}\n"
        )

        subject = "Aurora Signal: ハイコンフィデンス・シグナル"
        send_email(subject, body)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise
