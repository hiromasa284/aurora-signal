import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json

def send_email(subject, body, to_email):
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
            print("メール送信に成功しました！")
    except Exception as e:
        print(f"メール送信中にエラーが発生しました: {e}")

def main():
    with open("signal.json", "r", encoding="utf-8") as file:
        signal_data = json.load(file)

    subject = "Aurora Signal 最新アラート"
    body = f"以下は最新のアラート情報です：\n\n{json.dumps(signal_data, indent=4, ensure_ascii=False)}"
    send_email(subject, body, os.environ["SEND_TO"])

if __name__ == "__main__":
    main()
