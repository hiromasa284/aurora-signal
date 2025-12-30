# AuroraSignal

スイングトレード向けの株式シグナル通知システム。

## 特徴

- RSI21・ボリンジャーバンドによるシグナル判定
- Slack通知
- バックテスト機能あり

## セットアップ

```bash
pip install -r requirements.txt
cp config_sample.json config.json
# config.json に Slack Webhook を設定
python aurora_signal.py


> ※ VS Code で `.md` ファイルを作成すれば、プレビュー表示もできるよ（右上の「Open Preview」）
