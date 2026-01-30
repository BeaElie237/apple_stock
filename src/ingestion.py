import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_stock_data(ticker="AAPL", period="4y"):
    data = yf.download(ticker, period=period)
    if data.empty:
        raise ValueError("Aucune donnée téléchargée")
    return data

if __name__ == "__main__":
    df = download_stock_data()
    output_path = DATA_DIR / "aapl.csv"
    df.to_csv(output_path)
    print(f"[OK] Données sauvegardées : {output_path}")
