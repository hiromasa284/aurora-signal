import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json

def send_email(subject, body, to_email=None):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = to_email or os.getenv("SEND_TO", smtp_user)

    # デバッグ用ログ
    print(f"SMTP_USER: {smtp_user}")
    print(f"SMTP_PASS: {'***' if smtp_pass else 'None'}")
    print(f"TO_EMAIL: {to_email}")

    # 必須項目の確認
    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP_USER または SMTP_PASS が設定されていません")

    # メールの作成
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # GmailのSMTPサーバーに接続
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # 暗号化を開始
            server.login(smtp_user, smtp_pass)  # ログイン
            server.sendmail(smtp_user, to_email, msg.as_string())  # メール送信
            print("メール送信に成功しました！")
    except Exception as e:
        print(f"メール送信中にエラーが発生しました: {e}")
        raise  # エラーを再スローしてGitHub Actionsに通知

# メインロジック（テスト用）
def main():
    try:
        # テスト用のダミーアラートデータ
        signal_data = {
            "AAPL": {"signal": "BUY", "rsi": 28.5, "price": 147.23},
            "MSFT": {"signal": "SELL", "rsi": 72.8, "price": 310.45},
        }

        subject = "Aurora Signal 最新アラート"
        body = f"以下は最新のアラート情報です：\n\n{json.dumps(signal_data, indent=4, ensure_ascii=False)}"
        send_email(subject, body)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
