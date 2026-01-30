import pandas as pd
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
import os

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_USER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    df = pd.read_csv("predictions.csv")

    trend = "FAVORABLE 📈" if df["prediction"].iloc[-1] > df["prediction"].iloc[0] else "DEFAVORABLE 📉"

    body = f"""
    Résumé prédiction AAPL (21 jours)

    Tendance : {trend}
    Prix initial : {df['prediction'].iloc[0]:.2f}
    Prix final : {df['prediction'].iloc[-1]:.2f}
    """

    send_email("AAPL – Résultat prédiction", body)
    print("[OK] Email envoyé")
